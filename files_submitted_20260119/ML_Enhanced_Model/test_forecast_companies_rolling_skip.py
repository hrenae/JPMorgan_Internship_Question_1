# test_forecast_companies_rolling.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from hybrid_tft_accounting_model import HybridTFTAccounting
import preprocess_hybrid_tft_dataset as pre


# ---------- model loading (reuse your existing settings) ----------
def build_model(meta: dict, model_dir: str) -> HybridTFTAccounting:
    model = HybridTFTAccounting(
        n_tickers=len(meta["tickers"]),
        n_sectors=len(meta["sectors"]),
        hist_feat_dim=int(meta["hist_feat_dim"]),
        fut_feat_dim=int(meta["fut_feat_dim"]),
        horizon=int(meta["horizon"]),
        taus=[0.1, 0.5, 0.9],
        d_model=64,
        n_heads=4,
        lstm_units=64,
        dropout=0.1,
    )

    # build variables (dummy forward)
    B, L, H = 2, int(meta["lookback"]), int(meta["horizon"])
    dummy = {
        "hist_feats": tf.zeros([B, L, int(meta["hist_feat_dim"])]),
        "hist_mask": tf.ones([B, L]),
        "future_feats": tf.zeros([B, H, int(meta["fut_feat_dim"])]),
        "y0": tf.zeros([B, 8]),
        "x_future": tf.zeros([B, H, 6]),
        "period_days_future": tf.ones([B, H, 1]) * 365.0,
        "ticker_id": tf.zeros([B], tf.int32),
        "sector_id": tf.zeros([B], tf.int32),
        "size_log_ta": tf.zeros([B, 1]),
    }
    _ = model(dummy, training=False)

    w_best = os.path.join(model_dir, "weights_best.weights.h5")
    w_last = os.path.join(model_dir, "weights_last.weights.h5")
    model.load_weights(w_best if os.path.exists(w_best) else w_last)
    return model


def load_scalers(data_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    hist_scaler = dict(np.load(os.path.join(data_dir, "scaler_hist.npz")))
    fut_scaler = dict(np.load(os.path.join(data_dir, "scaler_fut.npz")))
    return hist_scaler, fut_scaler


# ---------- features consistent with your dataset builder ----------
HIST_FEAT_COLS = [
    "sales_growth","cogs_margin","opex_margin",
    "implied_dso","implied_dio","implied_dpo",
    "capex_to_sales_implied","dep_to_ppe_implied",
    "cash_to_sales","leverage",
    "year_norm","q_sin","q_cos",
    "period_days_norm",
]
FUT_FEAT_COLS = ["year_norm","q_sin","q_cos","period_days_norm"]


def enforce_monotone_quantiles(q10, q50, q90):
    """Monotone rearrangement to avoid quantile crossing (for plotting and yerr)."""
    q = np.stack([q10, q50, q90], axis=-1)
    q = np.sort(q, axis=-1)
    return q[..., 0], q[..., 1], q[..., 2]


def _filter_missing_period_rows(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """Drop placeholder periods where *all* key statement columns are NaN.

    Motivation:
    Some data pipelines expand to a regular quarterly grid and leave missing quarters as all-NaN rows.
    Replacing these NaNs with zeros (via np.nan_to_num) injects artificial signals into the model.
    Here we remove such rows and recompute period_days (and period_days_norm) so the model can
    perceive irregular time gaps through the time-delta features.
    """
    key_cols = [c for c in key_cols if c in df.columns]
    if not key_cols:
        return df

    keep = df[key_cols].notna().any(axis=1)
    df = df.loc[keep].copy().reset_index(drop=True)

    # Recompute time gaps after filtering so missing quarters become longer intervals.
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce").diff().dt.days.astype(float)

        # reasonable default for the first row (or degenerate cases)
        if len(df) >= 2 and np.isfinite(np.nanmedian(d.iloc[1:])):
            default_pd = float(np.nanmedian(d.iloc[1:]))
        else:
            default_pd = 90.0  # ~ one quarter

        df["period_days"] = d.fillna(default_pd).clip(lower=1.0)
        df["period_days_norm"] = df["period_days"] / 365.0

    return df


def rolling_backtest_one_ticker(
    ticker: str,
    meta: dict,
    model: HybridTFTAccounting,
    data_dir: str,
    seed_points: int = 4,
    use_step: int = 1,   # 1-step ahead by default
    skip_missing_periods: bool = False,
) -> pd.DataFrame:
    """
    Rolling/walk-forward backtest:
    use first seed_points observations to start, then roll forward.
    At each anchor t, predict horizon H; we keep only 'use_step' (1 or 2) to avoid duplicate dates.
    """
    lookback, H = int(meta["lookback"]), int(meta["horizon"])
    assert 1 <= use_step <= H

    hist_scaler, fut_scaler = load_scalers(data_dir)

    # sector id consistent with training universe (if available)
    uni_path = os.path.join(data_dir, "universe_used.csv")
    ticker_to_sector = {}
    if os.path.exists(uni_path):
        uni = pd.read_csv(uni_path)
        for _, r in uni.iterrows():
            ticker_to_sector[str(r["ticker"]).strip()] = str(r["sector"]).strip()

    sector = ticker_to_sector.get(ticker, "") or pre.sector_from_yf(ticker)

    # load statement series
    df = pre.fetch_period_table(ticker, freq=meta["freq"])
    df = pre.add_derived_features(df, sector=sector).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    # Optionally remove all-NaN placeholder periods (do NOT treat them as zeros).
    if skip_missing_periods:
        key_cols = list(dict.fromkeys(list(meta.get("state_cols", []))
                                      + list(meta.get("x_cols", []))
                                      + list(meta.get("out_cols", []))))
        df = _filter_missing_period_rows(df, key_cols)

    if len(df) < seed_points + H:
        raise RuntimeError(f"{ticker}: not enough points. len(df)={len(df)}, seed_points={seed_points}, H={H}")

    state_cols = meta["state_cols"]  # 8 dims
    x_cols = meta["x_cols"]          # 6 dims
    out_cols = meta["out_cols"]      # 10 dims

    # embeddings
    tid = int(meta["ticker_to_id"].get(ticker, 0))
    sid = int(meta["sector_to_id"].get(sector, 0))

    # size feature
    ta0_series = df["TA"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    ta0 = float(ta0_series.iloc[0]) if len(ta0_series) else 0.0
    size_log_ta = np.array([[np.log(max(ta0, 1.0))]], dtype=np.float32)

    rows = []
    start_t = seed_points - 1
    end_t = len(df) - use_step - 1  # last anchor such that t+use_step exists
    for t in range(start_t, end_t + 1):
        hist_start = max(0, t - lookback + 1)
        hist = df.iloc[hist_start:t+1]
        fut = df.iloc[t+1:t+1+H]
        fut_len = len(fut)
        # We only need the first 'use_step' target(s) to exist for evaluation,
        # but the model expects a fixed horizon H. If fut is shorter than H near
        # the end of the series, we pad future inputs by repeating the last
        # available future features, and we pad period_days with a typical value.
        if fut_len < use_step:
            continue


        # y0
        y0 = np.nan_to_num(hist[state_cols].iloc[-1].astype(float).values, nan=0.0).astype(np.float32)

        # hist feats + padding + mask
        hist_raw = np.nan_to_num(hist[HIST_FEAT_COLS].astype(float).values, nan=0.0).astype(np.float32)
        L_eff = hist_raw.shape[0]
        pad_len = lookback - L_eff
        if pad_len > 0:
            pad = np.zeros((pad_len, hist_raw.shape[1]), dtype=np.float32)
            hist_feats = np.vstack([pad, hist_raw])
            hist_mask = np.concatenate([np.zeros((pad_len,), dtype=np.float32),
                                        np.ones((L_eff,), dtype=np.float32)])
        else:
            hist_feats = hist_raw[-lookback:, :]
            hist_mask = np.ones((lookback,), dtype=np.float32)

        # future known features (pad to fixed horizon H if needed)
        fut_feats = np.nan_to_num(fut[FUT_FEAT_COLS].astype(float).values, nan=0.0).astype(np.float32)
        x_future = np.nan_to_num(fut[x_cols].astype(float).values, nan=0.0).astype(np.float32)
        period_days_future = fut["period_days"].astype(float).values.reshape(fut_len, 1).astype(np.float32)

        if fut_len < H:
            pad_n = H - fut_len

            # Repeat last available row for future features to keep shapes consistent.
            fut_feats = np.vstack([fut_feats, np.repeat(fut_feats[-1:, :], pad_n, axis=0)])
            x_future = np.vstack([x_future, np.repeat(x_future[-1:, :], pad_n, axis=0)])

            # Typical quarter length (in days); fall back to 90 days.
            pd_med = float(np.nanmedian(df["period_days"].astype(float).values)) if "period_days" in df.columns else 90.0
            if (not np.isfinite(pd_med)) or (pd_med <= 0):
                pd_med = 90.0
            period_days_future = np.vstack([period_days_future, np.full((pad_n, 1), pd_med, dtype=np.float32)])

        # final safety: ensure fixed horizon length
        if fut_feats.shape[0] != H:
            fut_feats = fut_feats[:H, :]
        if x_future.shape[0] != H:
            x_future = x_future[:H, :]
        if period_days_future.shape[0] != H:
            period_days_future = period_days_future[:H, :]

        # scale
        hist_feats = pre.apply_scaler(hist_feats[None, ...], hist_scaler).astype(np.float32)
        fut_feats = pre.apply_scaler(fut_feats[None, ...], fut_scaler).astype(np.float32)

        inp = {
            "hist_feats": hist_feats,
            "hist_mask": hist_mask[None, :].astype(np.float32),
            "future_feats": fut_feats,
            "y0": y0[None, :],
            "x_future": x_future[None, :, :],
            "period_days_future": period_days_future[None, :, :],
            "ticker_id": np.array([tid], np.int32),
            "sector_id": np.array([sid], np.int32),
            "size_log_ta": size_log_ta,
        }

        y_pred_q = model.predict(inp, verbose=0)[0]  # (H, 10, 3)
        k = use_step - 1  # 0-based

        # actual for that step
        actual = fut[out_cols].astype(float).values[k, :]
        date_k = pd.to_datetime(fut["date"].iloc[k])

        # enforce monotone quantiles per variable for stability
        q10 = y_pred_q[k, :, 0]
        q50 = y_pred_q[k, :, 1]
        q90 = y_pred_q[k, :, 2]
        q10, q50, q90 = enforce_monotone_quantiles(q10, q50, q90)

        row = {"ticker": ticker, "anchor_date": str(pd.to_datetime(df["date"].iloc[t]).date()),
               "date": str(date_k.date()), "step": use_step}
        for j, col in enumerate(out_cols):
            row[f"{col}_true"] = float(actual[j]) if np.isfinite(actual[j]) else np.nan
            row[f"{col}_p10"] = float(q10[j])
            row[f"{col}_p50"] = float(q50[j])
            row[f"{col}_p90"] = float(q90[j])
        rows.append(row)

        # --- also record driver vector x at the evaluation step (ground truth / used input) ---
        x_actual = fut[x_cols].astype(float).values[k, :]  # this is what we fed as x_future for that date
        for j, xname in enumerate(x_cols):
            row[f"{xname}_true"] = float(x_actual[j]) if np.isfinite(x_actual[j]) else np.nan

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def compute_mae_table(df_pred: pd.DataFrame, out_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in out_cols:
        yt = pd.to_numeric(df_pred[f"{col}_true"], errors="coerce")
        yp = pd.to_numeric(df_pred[f"{col}_p50"], errors="coerce")
        m = yt.notna() & yp.notna()
        mae = (yt[m] - yp[m]).abs().mean() if m.any() else np.nan
        rows.append({"var": col, "MAE(p50)": float(mae) if np.isfinite(mae) else np.nan, "n": int(m.sum())})
    return pd.DataFrame(rows)


def plot_company_grid(
    df_hist: pd.DataFrame,
    df_pred: pd.DataFrame,
    out_cols: List[str],
    out_path: str,
    scale: float = 1.0,
):
    import matplotlib.dates as mdates
    from matplotlib.ticker import FixedLocator, FuncFormatter
    from collections import OrderedDict

    def yq_formatter(x, pos=None):
        dt = mdates.num2date(x)
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}\nQ{q}"

    # ---- 防御性处理：日期列 ----
    if "date" in df_hist.columns:
        df_hist = df_hist.copy()
        df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
    if "date" in df_pred.columns:
        df_pred = df_pred.copy()
        df_pred["date"] = pd.to_datetime(df_pred["date"], errors="coerce")

    n = len(out_cols)
    nrows, ncols = (5, 2) if n == 10 else (int(np.ceil(n / 2)), 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 18), squeeze=False)
    axes_flat = axes.flatten()

    # ---- 统一x轴tick：最多显示约12个季度刻度 ----
    all_dates = pd.concat([df_hist.get("date", pd.Series(dtype="datetime64[ns]")),
                           df_pred.get("date", pd.Series(dtype="datetime64[ns]"))],
                          axis=0).dropna()
    all_dates = pd.to_datetime(sorted(pd.unique(all_dates)))
    if len(all_dates) > 12:
        step = int(np.ceil(len(all_dates) / 12))
        tick_dates = all_dates[::step]
    else:
        tick_dates = all_dates
    tick_locs = mdates.date2num(tick_dates) if len(tick_dates) > 0 else []

    for i, col in enumerate(out_cols):
        ax = axes_flat[i]

        # 1) 历史观测：灰色线 + 圆点（与你参考保持一致）
        if col in df_hist.columns and "date" in df_hist.columns:
            y = pd.to_numeric(df_hist[col], errors="coerce") / scale
            m_hist = df_hist["date"].notna() & y.notna()
            if m_hist.any():
                ax.plot(
                    df_hist.loc[m_hist, "date"],
                    y.loc[m_hist],
                    marker="o",
                    markersize=4,
                    linewidth=1.0,
                    color="gray",
                    alpha=0.5,
                    label="Historical Obs",
                )

        # 2) 预测分位点：P10/P50/P90 分别用不同 marker
        #    并强制保证分位不交叉（排序）
        p50_key = f"{col}_p50"
        p10_key = f"{col}_p10"
        p90_key = f"{col}_p90"

        if ("date" in df_pred.columns) and (p50_key in df_pred.columns) and (p10_key in df_pred.columns) and (p90_key in df_pred.columns):
            y50 = pd.to_numeric(df_pred[p50_key], errors="coerce") / scale
            y10 = pd.to_numeric(df_pred[p10_key], errors="coerce") / scale
            y90 = pd.to_numeric(df_pred[p90_key], errors="coerce") / scale

            q = np.stack([y10.to_numpy(), y50.to_numpy(), y90.to_numpy()], axis=-1)
            q = np.sort(q, axis=-1)

            m_pred = df_pred["date"].notna() & pd.Series(q[:, 1]).notna()
            x_pred = df_pred.loc[m_pred, "date"]
            p10_vals = q[m_pred.to_numpy(), 0]
            p50_vals = q[m_pred.to_numpy(), 1]
            p90_vals = q[m_pred.to_numpy(), 2]

            # P10: 下三角，红色
            ax.scatter(
                x_pred, p10_vals,
                marker="v", color="red", s=40,
                alpha=0.7, label="Pred P10 (Lower)"
            )
            # P50: 正方形，蓝色 + 黑边
            ax.scatter(
                x_pred, p50_vals,
                marker="s", color="blue", s=50,
                edgecolor="black", label="Pred P50 (Median)"
            )
            # P90: 上三角，绿色
            ax.scatter(
                x_pred, p90_vals,
                marker="^", color="green", s=40,
                alpha=0.7, label="Pred P90 (Upper)"
            )

        # 3) 真实值：黑色 X
        true_key = f"{col}_true"
        if ("date" in df_pred.columns) and (true_key in df_pred.columns):
            yt = pd.to_numeric(df_pred[true_key], errors="coerce") / scale
            m_true = df_pred["date"].notna() & yt.notna()
            if m_true.any():
                ax.scatter(
                    df_pred.loc[m_true, "date"],
                    yt.loc[m_true],
                    marker="x",
                    s=60,
                    color="black",
                    linewidths=2.0,
                    zorder=5,
                    label="Actual Truth",
                )

        ax.set_title(f"Variable: {col}", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        if len(tick_locs) > 0:
            ax.xaxis.set_major_locator(FixedLocator(tick_locs))
            ax.xaxis.set_major_formatter(FuncFormatter(yq_formatter))
            ax.tick_params(axis="x", labelsize=7)

    # 多余子图关闭
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    # ---- Legend：从所有子图汇总并去重（更稳健）----
    uniq = OrderedDict()
    for ax in axes_flat[:n]:
        handles, labels = ax.get_legend_handles_labels()
        for h, lab in zip(handles, labels):
            if lab not in uniq:
                uniq[lab] = h

    if len(uniq) > 0:
        fig.legend(
            list(uniq.values()),
            list(uniq.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=5,
            frameon=True,
            fontsize=9,
        )

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.965])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_hybrid_MIX")
    ap.add_argument("--model_dir", type=str, default="model_hybrid_tft")
    ap.add_argument("--tickers", type=str, default="0700.HK,9988.HK,GOOG,JPM,MSFT,VWAGY,XOM")
    ap.add_argument("--seed_points", type=int, default=4, help="How many initial time points to start rolling backtest.")
    ap.add_argument("--use_step", type=int, default=1, help="Which horizon step to evaluate/plot (1..H).")
    ap.add_argument("--skip_missing_periods", action="store_true", help="Drop placeholder periods where all key statement columns are NaN (avoid feeding zeros for missing quarters).")
    ap.add_argument("--out_dir", type=str, default="rolling_results")
    ap.add_argument("--plot_dir", type=str, default="rolling_figures")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.data_dir, "meta.json"), "r", encoding="utf-8"))
    model = build_model(meta, args.model_dir)
    out_cols = meta["out_cols"]
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    for tkr in tickers:
        print(f"\n=== Rolling backtest: {tkr} | seed_points={args.seed_points} | step={args.use_step} ===")
        df_pred = rolling_backtest_one_ticker(
            ticker=tkr, meta=meta, model=model, data_dir=args.data_dir,
            seed_points=args.seed_points, use_step=args.use_step,
            skip_missing_periods=args.skip_missing_periods
        )

        # also fetch history for plotting
        uni_path = os.path.join(args.data_dir, "universe_used.csv")
        sector = ""
        if os.path.exists(uni_path):
            uni = pd.read_csv(uni_path)
            m = uni["ticker"].astype(str).str.strip() == tkr
            if m.any():
                sector = str(uni.loc[m, "sector"].iloc[0]).strip()
        sector = sector or pre.sector_from_yf(tkr)
        df_hist = pre.fetch_period_table(tkr, freq=meta["freq"])
        df_hist = pre.add_derived_features(df_hist, sector=sector).sort_values("date").reset_index(drop=True)
        df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
        df_hist = df_hist.dropna(subset=["date"]).reset_index(drop=True)
        if args.skip_missing_periods:
            key_cols = list(dict.fromkeys(list(meta.get("state_cols", []))
                                          + list(meta.get("x_cols", []))
                                          + list(meta.get("out_cols", []))))
            df_hist = _filter_missing_period_rows(df_hist, key_cols)

        # metrics
        df_mae = compute_mae_table(df_pred, out_cols)

        # save excel
        os.makedirs(args.out_dir, exist_ok=True)
        xlsx_path = os.path.join(args.out_dir, f"rolling_{tkr.replace('/','_')}.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            df_pred.to_excel(w, index=False, sheet_name="predictions")
            df_mae.to_excel(w, index=False, sheet_name="metrics")
        print(f"Saved: {xlsx_path}")

        # plot
        os.makedirs(args.plot_dir, exist_ok=True)
        fig_path = os.path.join(args.plot_dir, f"rolling_{tkr.replace('/','_')}.pdf")
        plot_company_grid(df_hist, df_pred, out_cols, fig_path, scale=args.scale)
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
