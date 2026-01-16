#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast & plot annual balance-sheet trajectories using calibrated θ (company/sector/global).

This script implements the evaluation protocol described in the prompt:
  For each ticker, provide an observed prefix of length L (L=1,2,3,...),
  then roll forward predictions for all subsequent years using observed drivers x_t.

Outputs (written under OUT_DIR)
--------------------------------
OUT_DIR/
  predictions/
    <TICKER>__<SCOPE>__predictions.csv   (true + predicted series for L=1..L_MAX)
  metrics/
    <TICKER>__<SCOPE>__metrics.csv       (MAE/MAPE for each L)
  plots/
    <TICKER>__<SCOPE>__overview.png      (multi-panel overview figure)

Run:
  python forecast_plot_pareja_baseline_v3.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pareja_baseline_core import Params, step_numpy, mae, mape


# ---------------------------
# Configuration (edit here)
# ---------------------------
DATA_DIR = "data_tsrl_annual_v3"   # produced by prepare_data_tsrl_baseline_v3.py
THETA_DIR = "theta_pareja_v3"      # produced by fit_theta_pareja_baseline_v3.py
OUT_DIR = "results_pareja_v3"

FREQ = "A"
TEST_LABEL = "test"

# Forecast settings
L_MAX = 3   # evaluate L=1..L_MAX

# Variables to plot/evaluate (keep small & interpretable for PPT)
PLOT_VARS = ["C", "AR", "Inv", "K", "AP", "STD", "LTD", "TA", "TL", "E_implied"]

SCOPES = ["company", "sector", "global"]


# ---------------------------
# Helpers
# ---------------------------
def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, p)
    if os.path.exists(cand):
        return cand
    return os.path.join(os.getcwd(), p)


def _safe_sector_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    for ch in ["/", "\\", " ", "-", ":", ",", "."]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _load_theta(scope: str, ticker: str, sector: str) -> Params:
    if scope == "global":
        p = _resolve_path(os.path.join(THETA_DIR, "global.json"))
    elif scope == "sector":
        p = _resolve_path(os.path.join(THETA_DIR, "sector", f"{_safe_sector_name(sector)}.json"))
    elif scope == "company":
        p = _resolve_path(os.path.join(THETA_DIR, "company", f"{ticker.replace('/', '_')}.json"))
    else:
        raise ValueError(f"Unknown scope: {scope}")

    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing θ file: {p}")

    payload = json.loads(Path(p).read_text(encoding="utf-8"))
    th = payload.get("theta", payload)  # allow bare dict
    return Params(**{k: float(v) for k, v in th.items()})


def _load_periods(ticker: str) -> pd.DataFrame:
    p = _resolve_path(os.path.join(DATA_DIR, "periods", f"{ticker.replace('/', '_')}.csv"))
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "freq" in df.columns:
        df = df[df["freq"] == FREQ].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _load_transitions(ticker: str) -> pd.DataFrame:
    p = _resolve_path(os.path.join(DATA_DIR, "transitions", f"{ticker.replace('/', '_')}.csv"))
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "freq" in df.columns:
        df = df[df["freq"] == FREQ].copy()
    df["date_prev"] = pd.to_datetime(df["date_prev"])
    df["date_next"] = pd.to_datetime(df["date_next"])
    df = df.sort_values("date_next").reset_index(drop=True)
    return df


def _roll_forecast(
    df_periods: pd.DataFrame,
    df_trans: pd.DataFrame,
    theta: Params,
    L: int,
) -> pd.DataFrame:
    """Return a dataframe aligned to df_periods['date'] with predictions for all modeled vars.

    Interpretation of L:
      * You observe y_0, ..., y_{L-1} (levels), then predict y_L, ..., y_T.
    """
    if df_periods.empty or df_trans.empty:
        return pd.DataFrame()

    dates = df_periods["date"].to_list()
    n = len(dates)

    # Map transitions by date_next (end-of-period date)
    x_by_date = {pd.to_datetime(r["date_next"]): r for _, r in df_trans.iterrows()}

    # True series (levels)
    true = {v: df_periods.get(v, pd.Series([np.nan] * n)).to_numpy(dtype=float) for v in PLOT_VARS}

    # Pred series
    pred = {v: np.full((n,), np.nan, dtype=float) for v in PLOT_VARS}
    # For observed prefix, copy truth (useful for plotting continuity)
    for v in PLOT_VARS:
        pred[v][:min(L, n)] = true[v][:min(L, n)]

    # Rolling simulation
    for i in range(1, n):
        # Only start predicting at i >= L
        if i < L:
            continue

        dt = dates[i]
        if dt not in x_by_date:
            continue
        r = x_by_date[dt]

        # previous state: observed if i-1 < L, else predicted
        if (i - 1) < L:
            y_prev = {
                "C": true["C"][i - 1],
                "AR": true["AR"][i - 1],
                "Inv": true["Inv"][i - 1],
                "K": true["K"][i - 1],
                "AP": true["AP"][i - 1],
                "STD": true["STD"][i - 1],
                "LTD": true["LTD"][i - 1],
                "E_gross_report": df_periods.get("E_gross_report", pd.Series([np.nan]*n)).to_numpy(dtype=float)[i - 1],
                "E_implied": true["E_implied"][i - 1],
            }
        else:
            y_prev = {
                "C": pred["C"][i - 1],
                "AR": pred["AR"][i - 1],
                "Inv": pred["Inv"][i - 1],
                "K": pred["K"][i - 1],
                "AP": pred["AP"][i - 1],
                "STD": pred["STD"][i - 1],
                "LTD": pred["LTD"][i - 1],
                "E_gross_report": pred["E_implied"][i - 1],
                "E_implied": pred["E_implied"][i - 1],
            }

        x_t = {
            "S": r.get("S"),
            "COGS": r.get("COGS"),
            "OPEX": r.get("OPEX"),
            "EquityIssues": r.get("EquityIssues"),
            "NI": r.get("NI"),
            "Div": r.get("Div"),
        }

        yhat = step_numpy(y_prev=y_prev, x_t=x_t, theta=theta)

        pred["C"][i] = yhat["C"]
        pred["AR"][i] = yhat["AR"]
        pred["Inv"][i] = yhat["Inv"]
        pred["K"][i] = yhat["K"]
        pred["AP"][i] = yhat["AP"]
        pred["STD"][i] = yhat["STD"]
        pred["LTD"][i] = yhat["LTD"]
        pred["TA"][i] = yhat["TA"]
        pred["TL"][i] = yhat["TL"]
        pred["E_implied"][i] = yhat["E_implied"]

    out = pd.DataFrame({"date": dates})
    for v in PLOT_VARS:
        out[f"{v}_true"] = true[v]
        out[f"{v}_pred"] = pred[v]
    return out


def _metrics_for_L(df_pred: pd.DataFrame, L: int) -> Dict[str, float]:
    """Compute MAE/MAPE on the forecast segment starting at index L."""
    out: Dict[str, float] = {"L": float(L)}
    if df_pred.empty:
        for v in PLOT_VARS:
            out[f"{v}_MAE"] = np.nan
            out[f"{v}_MAPE"] = np.nan
        return out

    # Determine forecast mask
    mask = np.arange(len(df_pred)) >= int(L)
    for v in PLOT_VARS:
        a = df_pred.loc[mask, f"{v}_true"].to_numpy(dtype=float)
        b = df_pred.loc[mask, f"{v}_pred"].to_numpy(dtype=float)
        out[f"{v}_MAE"] = mae(a, b)
        out[f"{v}_MAPE"] = mape(a, b)

    # Accounting residual check for predictions
    ta = df_pred.loc[mask, "TA_pred"].to_numpy(float)
    tl = df_pred.loc[mask, "TL_pred"].to_numpy(float)
    e = df_pred.loc[mask, "E_implied_pred"].to_numpy(float)
    resid = ta - (tl + e)
    if np.isfinite(resid).any():
        out["max_abs_identity_resid"] = float(np.nanmax(np.abs(resid)))
    else:
        out["max_abs_identity_resid"] = np.nan
    return out


def _plot_overview(ticker: str, scope: str, df_true: pd.DataFrame, dfs_L: List[Tuple[int, pd.DataFrame]], out_path: str):
    """Multi-panel figure: each panel shows true series and predicted series for L=1..L_MAX."""
    nvars = len(PLOT_VARS)
    ncol = 3
    nrow = int(np.ceil(nvars / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
    axes = axes.flatten()

    for j, v in enumerate(PLOT_VARS):
        ax = axes[j]
        ax.plot(df_true["date"], df_true[f"{v}_true"], label="true")
        for L, dfp in dfs_L:
            ax.plot(dfp["date"], dfp[f"{v}_pred"], label=f"pred (L={L})", linestyle="--")
        ax.set_title(v)
        ax.set_xlabel("date")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    for k in range(nvars, len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"{ticker} | scope={scope} | annual roll-forward forecasts", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    os.makedirs(_resolve_path(OUT_DIR), exist_ok=True)
    for sub in ["predictions", "metrics", "plots"]:
        os.makedirs(_resolve_path(os.path.join(OUT_DIR, sub)), exist_ok=True)

    df_u = pd.read_csv(_resolve_path(os.path.join(DATA_DIR, "universe.csv")))
    report = df_u[df_u["label"].astype(str).str.lower().eq(TEST_LABEL)].copy()
    report_tickers = report["ticker"].tolist()

    if len(report_tickers) == 0:
        raise RuntimeError(f"No tickers with label='{TEST_LABEL}' found in {DATA_DIR}/universe.csv")

    for _, r in report.iterrows():
        ticker = str(r["ticker"])
        sector = str(r["sector"])

        df_p = _load_periods(ticker)
        df_t = _load_transitions(ticker)

        if df_p.empty or df_t.empty or len(df_p) < 3:
            print(f"[WARN] Skip {ticker}: insufficient annual data.")
            continue

        for scope in SCOPES:
            try:
                theta = _load_theta(scope=scope, ticker=ticker, sector=sector)
            except Exception as e:
                print(f"[WARN] Skip {ticker} scope={scope}: {e}")
                continue

            dfs_L: List[Tuple[int, pd.DataFrame]] = []
            metrics_rows: List[Dict[str, float]] = []

            # Build predictions for each L and store side-by-side
            df_pred_all = None
            for L in range(1, int(L_MAX) + 1):
                df_pred = _roll_forecast(df_p, df_t, theta, L=L)
                if df_pred.empty:
                    continue
                dfs_L.append((L, df_pred))
                metrics_rows.append(_metrics_for_L(df_pred, L=L))

                # Merge into a single wide table
                if df_pred_all is None:
                    df_pred_all = df_pred[["date"] + [f"{v}_true" for v in PLOT_VARS]].copy()
                for v in PLOT_VARS:
                    df_pred_all[f"{v}_pred_L{L}"] = df_pred[f"{v}_pred"].to_numpy(dtype=float)

            if df_pred_all is None:
                continue

            # Save predictions
            pred_path = _resolve_path(os.path.join(OUT_DIR, "predictions", f"{ticker.replace('/', '_')}__{scope}__predictions.csv"))
            df_pred_all.to_csv(pred_path, index=False)

            # Save metrics
            df_met = pd.DataFrame(metrics_rows)
            met_path = _resolve_path(os.path.join(OUT_DIR, "metrics", f"{ticker.replace('/', '_')}__{scope}__metrics.csv"))
            df_met.to_csv(met_path, index=False)

            # Plot
            plot_path = _resolve_path(os.path.join(OUT_DIR, "plots", f"{ticker.replace('/', '_')}__{scope}__overview.png"))
            _plot_overview(ticker, scope, dfs_L[0][1], dfs_L, plot_path)

            print(f"[OK] {ticker} scope={scope} -> saved predictions/metrics/plots")

    print(f"[DONE] Forecasting & plotting complete. OUT_DIR={OUT_DIR}")


if __name__ == "__main__":
    main()
