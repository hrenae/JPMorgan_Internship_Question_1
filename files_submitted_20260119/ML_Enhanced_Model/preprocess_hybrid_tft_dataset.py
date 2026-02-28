# preprocess_hybrid_tft_dataset.py
# -*- coding: utf-8 -*-
"""
Build dataset for Hybrid TFT + Differentiable Accounting model.

Enhancements vs earlier draft:
- freq supports A / Q / MIX (annual + quarterly merged by date; keep annual if same date).
- Allows padding if history length < lookback -> outputs hist_mask.
- Adds period_days (annual=365, quarterly=365/4) and uses it for implied turnover features.
- Financials handling: if COGS missing -> set 0; also masks Inv/AP losses if COGS ~ 0.
"""

from __future__ import annotations
import argparse, json, os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import prepare_data_tsrl_baseline_v3 as prep

try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError("yfinance is required. Please `pip install yfinance`.") from e


PERIOD_DAYS_ANNUAL = 365.0
PERIOD_DAYS_QUARTER = 365.0 / 4.0


def safe_div(a, b, default=0.0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, default, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-12)
    out[m] = a[m] / b[m]
    return out


def year_quarter(dt: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    dt = pd.to_datetime(dt)
    year = dt.dt.year.values.astype(int)
    q = ((dt.dt.month.values - 1) // 3 + 1).astype(int)
    return year, q


def build_universe(args) -> pd.DataFrame:
    if args.universe_csv and os.path.exists(args.universe_csv):
        df = pd.read_csv(args.universe_csv)
        if "ticker" not in df.columns:
            raise ValueError("universe_csv must include a 'ticker' column.")
        if "label" not in df.columns:
            df["label"] = "train"
        if "sector" not in df.columns:
            df["sector"] = ""
        df["ticker"] = df["ticker"].astype(str).str.strip()
        df["sector"] = df["sector"].astype(str).fillna("").str.strip()
        df["label"] = df["label"].astype(str).fillna("train").str.strip().str.lower()
        return df[["ticker", "sector", "label"]].copy()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise ValueError("Provide --tickers or --universe_csv.")
    return pd.DataFrame({"ticker": tickers, "sector": "", "label": "train"})


def sector_from_yf(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return str(info.get("sector", "") or "")
    except Exception:
        return ""


def _fetch_one(ticker: str, one_freq: str) -> pd.DataFrame:
    one_freq = one_freq.upper()
    if one_freq not in ("A", "Q"):
        raise ValueError("one_freq must be A or Q")

    if one_freq == "A":
        BS_ATTRS = ["balance_sheet"]
        INC_ATTRS = ["financials"]
        CF_ATTRS = ["cashflow", "cash_flow", "cashflow_stmt"]
    else:
        BS_ATTRS = ["quarterly_balance_sheet"]
        INC_ATTRS = ["quarterly_financials"]
        CF_ATTRS = ["quarterly_cashflow"]

    bs_df, _ = prep.fetch_with_retries(ticker, BS_ATTRS, retries=2, pause_s=1.0)
    inc_df, _ = prep.fetch_with_retries(ticker, INC_ATTRS, retries=2, pause_s=1.0)
    cf_df, _ = prep.fetch_with_retries(ticker, CF_ATTRS, retries=2, pause_s=1.0)

    # IMPORTANT: correct argument order (freq is 2nd positional)
    bs  = prep.build_period_table_from_statement(bs_df,  one_freq, prep.extract_balance_sheet_one_period)
    inc = prep.build_period_table_from_statement(inc_df, one_freq, prep.extract_income_one_period)
    cf  = prep.build_period_table_from_statement(cf_df,  one_freq, prep.extract_cashflow_one_period)

    df = prep.merge_statements(bs, inc, cf)
    df = prep.compute_residuals_and_audits(df)
    df = prep.add_time_features(df)
    df["ticker"] = ticker
    df["freq_tag"] = one_freq
    return df


def fetch_period_table(ticker: str, freq: str) -> pd.DataFrame:
    """
    freq: A / Q / MIX
    MIX: merge A+Q by date; keep annual if same date.
    """
    freq = freq.upper()
    if freq in ("A", "Q"):
        df = _fetch_one(ticker, freq)
        return df.sort_values("date").reset_index(drop=True)

    if freq != "MIX":
        raise ValueError("freq must be 'A', 'Q', or 'MIX'")

    df_a = _fetch_one(ticker, "A")
    df_q = _fetch_one(ticker, "Q")
    df = pd.concat([df_a, df_q], ignore_index=True)

    # keep annual when same date exists
    df["freq_rank"] = df["freq_tag"].map({"A": 0, "Q": 1})
    df = df.sort_values(["date", "freq_rank"]).drop_duplicates(subset=["date"], keep="first")
    df = df.drop(columns=["freq_rank"]).sort_values("date").reset_index(drop=True)
    return df


def add_derived_features(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    df = df.copy()
    df["sector"] = sector

    # period_days depends on freq_tag
    if "freq_tag" in df.columns:
        df["period_days"] = df["freq_tag"].map({"A": PERIOD_DAYS_ANNUAL, "Q": PERIOD_DAYS_QUARTER}).fillna(PERIOD_DAYS_ANNUAL)
    else:
        df["period_days"] = PERIOD_DAYS_ANNUAL

    # Financials: COGS is structurally missing sometimes; treat as 0 (do NOT drop)
    if isinstance(sector, str) and ("Financial" in sector):
        if "COGS" in df.columns:
            df["COGS"] = df["COGS"].fillna(0.0)

    # ensure columns exist
    need = ["C","AR","Inv","K","AP","STD","LTD","TA","TL","E_implied",
            "S","COGS","OPEX","CapEx","DA_cf","EquityIssues","NI","Div","date","period_days"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan

    pdays = df["period_days"].astype(float).values
    pdays = np.where(np.isfinite(pdays) & (pdays > 0), pdays, PERIOD_DAYS_ANNUAL)

    # annualized log sales growth w.r.t. varying period length
    S = df["S"].astype(float).values
    S_prev = np.roll(S, 1)
    g = np.zeros_like(S, dtype=float)
    m = np.isfinite(S) & np.isfinite(S_prev) & (S > 0) & (S_prev > 0)
    # normalize by current period length to make growth more comparable across A/Q/MIX
    g[m] = np.log(S[m] / S_prev[m]) * (PERIOD_DAYS_ANNUAL / pdays[m])
    df["sales_growth"] = g

    df["cogs_margin"] = safe_div(df["COGS"].values, df["S"].values, default=0.0)
    df["opex_margin"] = safe_div(df["OPEX"].values, df["S"].values, default=0.0)

    # implied turnover (use period_days of that row)
    df["implied_dso"] = safe_div(pdays * df["AR"].values, df["S"].values, default=0.0)
    df["implied_dio"] = safe_div(pdays * df["Inv"].values, df["COGS"].values, default=0.0)
    df["implied_dpo"] = safe_div(pdays * df["AP"].values, df["COGS"].values, default=0.0)

    df["capex_to_sales_implied"] = safe_div(df["CapEx"].values, df["S"].values, default=0.0)

    K_prev = np.roll(df["K"].astype(float).values, 1)
    df["dep_to_ppe_implied"] = safe_div(df["DA_cf"].values, K_prev, default=0.0)

    df["cash_to_sales"] = safe_div(df["C"].values, df["S"].values, default=0.0)
    df["leverage"] = safe_div((df["STD"].values + df["LTD"].values), np.maximum(df["TA"].values, 1e-6), default=0.0)

    year, q = year_quarter(df["date"])
    df["q"] = q
    df["year_norm"] = (year - year.min()) / max(1.0, (year.max() - year.min()))
    df["q_sin"] = np.sin(2.0 * np.pi * q / 4.0)
    df["q_cos"] = np.cos(2.0 * np.pi * q / 4.0)

    # normalize period length as known feature
    df["period_days_norm"] = pdays / PERIOD_DAYS_ANNUAL

    return df


def make_samples_for_ticker(df: pd.DataFrame, lookback: int, horizon: int):
    df = df.sort_values("date").reset_index(drop=True).copy()

    state_cols = ["C","AR","Inv","K","AP","STD","LTD","E_implied"]
    out_cols = ["C","AR","Inv","K","AP","STD","LTD","TA","TL","E_implied"]
    x_cols = ["S","COGS","OPEX","EquityIssues","NI","Div"]

    hist_feat_cols = [
        "sales_growth","cogs_margin","opex_margin",
        "implied_dso","implied_dio","implied_dpo",
        "capex_to_sales_implied","dep_to_ppe_implied",
        "cash_to_sales","leverage",
        "year_norm","q_sin","q_cos",
        "period_days_norm",
    ]
    fut_feat_cols = ["year_norm","q_sin","q_cos","period_days_norm"]

    for c in state_cols + out_cols + x_cols + hist_feat_cols + fut_feat_cols + ["period_days"]:
        if c not in df.columns:
            df[c] = np.nan

    ta0_series = df["TA"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    ta0 = float(ta0_series.iloc[0]) if len(ta0_series) else 0.0
    size_log_ta = np.array([np.log(max(ta0, 1.0))], dtype=np.float32)

    samples = []
    n = len(df)
    if n < 1 + horizon:
        return samples

    for t in range(0, n - horizon):
        hist_start = max(0, t - lookback + 1)
        hist = df.iloc[hist_start : t + 1]
        fut = df.iloc[t + 1 : t + 1 + horizon]
        if len(fut) < horizon:
            continue

        y0 = hist[state_cols].iloc[-1].astype(float).values
        if not np.isfinite(y0).any():
            continue

        hist_feats_raw = hist[hist_feat_cols].astype(float).values
        fut_feats = fut[fut_feat_cols].astype(float).values
        x_future = fut[x_cols].astype(float).values
        y_true = fut[out_cols].astype(float).values
        mask_y = np.isfinite(y_true).astype(np.float32)

        # Financials-like periods: if COGS ~ 0, Inv/AP are not meaningful -> mask their losses
        cogs_future = np.abs(fut["COGS"].astype(float).fillna(0.0).values)
        if np.nanmedian(cogs_future) < 1e-9:
            inv_idx = out_cols.index("Inv")
            ap_idx = out_cols.index("AP")
            mask_y[:, inv_idx] = 0.0
            mask_y[:, ap_idx] = 0.0

        # padding to fixed lookback
        L_eff = hist_feats_raw.shape[0]
        pad_len = lookback - L_eff
        if pad_len > 0:
            pad = np.zeros((pad_len, hist_feats_raw.shape[1]), dtype=float)
            hist_feats = np.vstack([pad, hist_feats_raw])
            hist_mask = np.concatenate([np.zeros((pad_len,), dtype=np.float32),
                                        np.ones((L_eff,), dtype=np.float32)])
        else:
            hist_feats = hist_feats_raw[-lookback:, :]
            hist_mask = np.ones((lookback,), dtype=np.float32)

        # period_days for future steps
        pd_future = fut["period_days"].astype(float).values.reshape(horizon, 1)
        pd_future = np.where(np.isfinite(pd_future) & (pd_future > 0), pd_future, PERIOD_DAYS_ANNUAL).astype(np.float32)

        # NaN -> 0 for inputs, but keep mask_y for targets
        hist_feats = np.nan_to_num(hist_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        fut_feats = np.nan_to_num(fut_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y0 = np.nan_to_num(y0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        x_future = np.nan_to_num(x_future, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        samples.append(dict(
            hist_feats=hist_feats,
            hist_mask=hist_mask.astype(np.float32),
            future_feats=fut_feats,
            y0=y0,
            x_future=x_future,
            period_days_future=pd_future,
            y_true=y_true,
            mask_y=mask_y,
            size_log_ta=size_log_ta,
            date0=str(pd.to_datetime(hist["date"].iloc[-1]).date()),
            date_end=str(pd.to_datetime(fut["date"].iloc[-1]).date()),
        ))
    return samples


def fit_scaler(arr: np.ndarray) -> Dict[str, np.ndarray]:
    x = arr.reshape([-1, arr.shape[-1]])
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return {"mean": mu.astype(np.float32), "std": sd.astype(np.float32)}


def apply_scaler(arr: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    mu = scaler["mean"][None, None, :]
    sd = scaler["std"][None, None, :]
    return (arr - mu) / sd


def stack_samples(samples, ticker_id: int, sector_id: int):
    for s in samples:
        s["ticker_id"] = np.int32(ticker_id)
        s["sector_id"] = np.int32(sector_id)

    keys = [
        "hist_feats","hist_mask","future_feats","y0","x_future","period_days_future",
        "y_true","mask_y","size_log_ta",
        "ticker_id","sector_id","date0","date_end"
    ]
    out = {}
    for k in keys:
        if k in ("date0","date_end"):
            out[k] = np.array([s[k] for s in samples])
        else:
            out[k] = np.stack([s[k] for s in samples], axis=0)
    return out


def concat_parts(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_csv", type=str, default="")
    ap.add_argument("--tickers", type=str, default="0700.HK,9988.HK,GOOG,JPM,MSFT,VWAGY,XOM")
    ap.add_argument("--freq", type=str, default="A", choices=["A","Q","MIX"])
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--out_dir", type=str, default="data_hybrid")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    uni = build_universe(args)

    # fill missing sectors
    for i in range(len(uni)):
        if not str(uni.loc[i, "sector"]).strip():
            uni.loc[i, "sector"] = sector_from_yf(str(uni.loc[i, "ticker"]))

    tickers = sorted(uni["ticker"].unique().tolist())
    sectors = sorted(uni["sector"].unique().tolist())
    ticker_to_id = {t:i for i,t in enumerate(tickers)}
    sector_to_id = {s:i for i,s in enumerate(sectors)}

    # if user didn't define splits: random val split by ticker
    if (uni["label"] == "train").all():
        tks = tickers.copy()
        rng.shuffle(tks)
        n_val = max(1, int(len(tks) * args.val_ratio))
        val_set = set(tks[:n_val])
        uni["label"] = uni["ticker"].apply(lambda t: "val" if t in val_set else "train")

    TEST_TICKERS = set([
        "0700.HK", "9988.HK", "GOOG", "JPM", "MSFT", "VWAGY", "XOM"
    ])

    uni["ticker"] = uni["ticker"].astype(str).str.strip()
    uni.loc[uni["ticker"].isin(TEST_TICKERS), "label"] = "test"

    splits = {"train": [], "val": [], "test": []}

    for _, r in uni.iterrows():
        tkr = str(r["ticker"])
        sector = str(r["sector"])
        label = str(r["label"]).lower()
        label = label if label in splits else "train"

        print(f"[fetch] {tkr} | sector={sector} | split={label}")
        try:
            df = fetch_period_table(tkr, freq=args.freq)
            df = add_derived_features(df, sector=sector)
        except Exception as e:
            print(f"  !! failed: {tkr}: {type(e).__name__}: {e}")
            continue

        samp = make_samples_for_ticker(df, lookback=args.lookback, horizon=args.horizon)
        if not samp:
            print(f"  !! no samples for {tkr} (try smaller --lookback/--horizon or freq=MIX/A)")
            continue

        part = stack_samples(samp, ticker_id=ticker_to_id[tkr], sector_id=sector_to_id[sector])
        splits[label].append(part)

    if not splits["train"]:
        raise RuntimeError("No training samples created. Try freq=A or freq=MIX and smaller lookback/horizon.")

    train = concat_parts(splits["train"])
    val = concat_parts(splits["val"]) if splits["val"] else None
    test = concat_parts(splits["test"]) if splits["test"] else None

    # scalers for hist/future features only
    hist_scaler = fit_scaler(train["hist_feats"])
    fut_scaler = fit_scaler(train["future_feats"])

    for split in [train, val, test]:
        if split is None:
            continue
        split["hist_feats"] = apply_scaler(split["hist_feats"], hist_scaler).astype(np.float32)
        split["future_feats"] = apply_scaler(split["future_feats"], fut_scaler).astype(np.float32)

    meta = {
        "freq": args.freq,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "tickers": tickers,
        "sectors": sectors,
        "ticker_to_id": ticker_to_id,
        "sector_to_id": sector_to_id,
        "state_cols": ["C","AR","Inv","K","AP","STD","LTD","E_implied"],
        "x_cols": ["S","COGS","OPEX","EquityIssues","NI","Div"],
        "out_cols": ["C","AR","Inv","K","AP","STD","LTD","TA","TL","E_implied"],
        "hist_feat_dim": int(train["hist_feats"].shape[-1]),
        "fut_feat_dim": int(train["future_feats"].shape[-1]),
        "has_hist_mask": True,
        "has_period_days_future": True,
    }

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    np.savez(os.path.join(args.out_dir, "scaler_hist.npz"), **hist_scaler)
    np.savez(os.path.join(args.out_dir, "scaler_fut.npz"), **fut_scaler)

    np.savez(os.path.join(args.out_dir, "train.npz"), **train)
    if val is not None:
        np.savez(os.path.join(args.out_dir, "val.npz"), **val)
    if test is not None:
        np.savez(os.path.join(args.out_dir, "test.npz"), **test)

    uni_out = uni.copy()
    uni_out["ticker_id"] = uni_out["ticker"].map(ticker_to_id)
    uni_out["sector_id"] = uni_out["sector"].map(sector_to_id)
    uni_out.to_csv(os.path.join(args.out_dir, "universe_used.csv"), index=False)

    print(f"Saved dataset to: {args.out_dir}")
    print(f"Train: {train['hist_feats'].shape[0]}")
    if val is not None:
        print(f"Val:   {val['hist_feats'].shape[0]}")
    if test is not None:
        print(f"Test:  {test['hist_feats'].shape[0]}")


if __name__ == "__main__":
    main()
