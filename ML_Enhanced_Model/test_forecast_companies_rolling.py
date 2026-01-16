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


def rolling_backtest_one_ticker(
    ticker: str,
    meta: dict,
    model: HybridTFTAccounting,
    data_dir: str,
    seed_points: int = 4,
    use_step: int = 1,   # 1-step ahead by default
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
    end_t = len(df) - H - 1  # last anchor such that t+H exists
    for t in range(start_t, end_t + 1):
        hist_start = max(0, t - lookback + 1)
        hist = df.iloc[hist_start:t+1]
        fut = df.iloc[t+1:t+1+H]

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

        # future known features
        fut_feats = np.nan_to_num(fut[FUT_FEAT_COLS].astype(float).values, nan=0.0).astype(np.float32)
        x_future = np.nan_to_num(fut[x_cols].astype(float).values, nan=0.0).astype(np.float32)
        period_days_future = fut["period_days"].astype(float).values.reshape(H, 1).astype(np.float32)

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


def plot_company_grid(df_hist: pd.DataFrame, df_pred: pd.DataFrame, out_cols: List[str],
                      out_path: str, scale: float = 1.0):
    import matplotlib.dates as mdates
    from matplotlib.ticker import FixedLocator
    from matplotlib.ticker import FuncFormatter

    def yq_formatter(x, pos=None):
        dt = mdates.num2date(x)
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}\nQ{q}"

    n = len(out_cols)
    nrows, ncols = (5, 2) if n == 10 else (int(np.ceil(n / 2)), 2)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 18), squeeze=False)
    axes_flat = axes.flatten()

    all_dates = pd.concat([df_hist["date"], df_pred["date"]], axis=0).dropna()
    all_dates = pd.to_datetime(sorted(pd.unique(all_dates)))
    if len(all_dates) > 12:
        step = int(np.ceil(len(all_dates) / 12))
        tick_dates = all_dates[::step]
    else:
        tick_dates = all_dates
    tick_locs = mdates.date2num(tick_dates)

    for i, col in enumerate(out_cols):
        ax = axes_flat[i]

        # 1. 绘制历史观测值 (蓝实线)
        if col in df_hist.columns:
            y = pd.to_numeric(df_hist[col], errors="coerce") / scale
            ax.plot(df_hist["date"], y, marker="o", markersize=4, linewidth=1.0, 
                    color="gray", alpha=0.5, label="Historical Obs")

        # 2. 准备预测数据
        y50 = pd.to_numeric(df_pred[f"{col}_p50"], errors="coerce") / scale
        y10 = pd.to_numeric(df_pred[f"{col}_p10"], errors="coerce") / scale
        y90 = pd.to_numeric(df_pred[f"{col}_p90"], errors="coerce") / scale

        # 确保分位数不交叉
        q = np.stack([y10.values, y50.values, y90.values], axis=-1)
        q = np.sort(q, axis=-1)
        
        m = df_pred["date"].notna() & pd.Series(q[:, 1]).notna()
        x_pred = df_pred.loc[m, "date"]
        p10_vals = q[m, 0]
        p50_vals = q[m, 1]
        p90_vals = q[m, 2]

        # --- 修改部分：将 Errorbar 替换为三个独立的点 ---
        # P10: 下三角，红色
        ax.scatter(x_pred, p10_vals, marker="v", color="red", s=40, 
                   alpha=0.7, label="Pred P10 (Lower)")
        
        # P50: 正方形，蓝色
        ax.scatter(x_pred, p50_vals, marker="s", color="blue", s=50, 
                   edgecolor='black', label="Pred P50 (Median)")
        
        # P90: 上三角，绿色
        ax.scatter(x_pred, p90_vals, marker="^", color="green", s=40, 
                   alpha=0.7, label="Pred P90 (Upper)")
        # ----------------------------------------------

        # 3. 绘制真实值 (黑色 X，用于对比预测准确度)
        yt = pd.to_numeric(df_pred[f"{col}_true"], errors="coerce") / scale
        mt = df_pred["date"].notna() & yt.notna()
        ax.scatter(df_pred.loc[mt, "date"], yt.loc[mt], marker="x", s=60, 
                   color="black", linewidths=2.0, zorder=5, label="Actual Truth")

        ax.set_title(f"Variable: {col}", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        ax.xaxis.set_major_locator(FixedLocator(tick_locs))
        ax.xaxis.set_major_formatter(FuncFormatter(yq_formatter))
        ax.tick_params(axis="x", labelsize=7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    # 更新图例提取逻辑
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), 
               ncol=5, frameon=True, fontsize=9)

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
            seed_points=args.seed_points, use_step=args.use_step
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
