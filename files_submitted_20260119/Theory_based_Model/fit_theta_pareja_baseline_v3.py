#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate θ for the Pareja-style baseline balance-sheet simulator using annual transitions.

This script fits three parameter granularities:
  (1) company-level θ  : each report ticker independently (uses only that ticker's history)
  (2) sector-level θ   : one θ per sector (uses all TRAIN tickers in that sector)
  (3) global θ         : one θ for the whole universe (uses all TRAIN tickers)

Outputs (written under THETA_DIR)
--------------------------------
THETA_DIR/
  company/<TICKER>.json
  sector/<SECTOR_SAFE>.json
  global.json
  summary.csv           one row per fitted θ with training/holdout metrics
  metadata.json         config snapshot for reproducibility

Run:
  python fit_theta_pareja_baseline_v3.py

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pareja_baseline_core import (
    Params,
    clean_transitions,
    fit_theta_auto,
    load_transitions_csv,
    step_numpy,
    mae,
    mape,
)

# ---------------------------
# Configuration (edit here)
# ---------------------------
DATA_DIR = "data_tsrl_annual_v3"     # produced by prepare_data_tsrl_baseline_v3.py
TRANS_DIR = os.path.join(DATA_DIR, "transitions")
UNIVERSE_CSV = os.path.join(DATA_DIR, "universe.csv")

THETA_DIR = "theta_pareja_v3"

# Only annual transitions used.
FREQ = "A"

# Label logic: in DataPrepare.csv, report tickers are labeled "test"; others are train.
TRAIN_LABEL = "train"
TEST_LABEL = "test"

# Training mask definition (within each ticker's time-ordered transitions).
# Example: 0.8 means use first 80% years for calibration, hold out last 20% (within each ticker).
TRAIN_FRAC_WITHIN_TICKER = 0.80

# Calibration mode: "auto" -> heuristic init + TF fine-tune if enough data & TF installed.
FIT_MODE = "auto"          # "auto" | "heuristic" | "tf"
TF_EPOCHS = 2500
TF_LR = 5e-3
MIN_TF_TRAIN = 30
TF_LOSS_MODE = "relative"  # recommended for sector/global fits

# Optional: exclude "test" tickers when training sector/global.
EXCLUDE_TEST_FROM_TRAIN = True

RANDOM_SEED = 0


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


def _load_one_ticker_transitions(ticker: str) -> pd.DataFrame:
    p = os.path.join(_resolve_path(TRANS_DIR), f"{ticker.replace('/', '_')}.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = load_transitions_csv(p, freq=FREQ)
    df = clean_transitions(df)
    df["ticker"] = ticker
    return df


def _build_train_mask_within_ticker(df: pd.DataFrame, frac: float) -> np.ndarray:
    """Train mask using first `frac` transitions per ticker in chronological order."""
    if df.empty:
        return np.zeros((0,), dtype=bool)
    df = df.sort_values(["ticker", "date_next"]).copy()
    g = df.groupby("ticker", sort=False).cumcount()
    n = df.groupby("ticker", sort=False)["date_next"].transform("count")
    cutoff = np.floor(frac * n).astype(int) - 1
    # train if index <= cutoff (at least 1 transition if possible)
    train = g <= np.maximum(cutoff, 0)
    return train.to_numpy(dtype=bool)


def _one_step_eval(df: pd.DataFrame, theta: Params, mask: np.ndarray) -> Dict[str, float]:
    if df.empty or mask.sum() == 0:
        return {"n": int(mask.sum()), "C_MAE": np.nan, "C_MAPE": np.nan, "E_MAE": np.nan, "E_MAPE": np.nan}

    # Compute predictions row-wise (small datasets; clarity > micro-optimizations).
    cols_true = {
        "C": "C_next",
        "AR": "AR_next",
        "Inv": "Inv_next",
        "K": "K_next",
        "AP": "AP_next",
        "STD": "STD_next",
        "LTD": "LTD_next",
        "E": "E_next",
    }

    pred = {k: [] for k in cols_true.keys()}
    true = {k: [] for k in cols_true.keys()}

    for _, r in df.loc[mask].iterrows():
        y_prev = {
            "C": r.get("C_prev"),
            "AR": r.get("AR_prev"),
            "Inv": r.get("Inv_prev"),
            "K": r.get("K_prev"),
            "AP": r.get("AP_prev"),
            "STD": r.get("STD_prev"),
            "LTD": r.get("LTD_prev"),
            "E_gross_report": r.get("E_prev"),
            "E_implied": r.get("E_prev"),
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

        pred["C"].append(yhat["C"])
        pred["AR"].append(yhat["AR"])
        pred["Inv"].append(yhat["Inv"])
        pred["K"].append(yhat["K"])
        pred["AP"].append(yhat["AP"])
        pred["STD"].append(yhat["STD"])
        pred["LTD"].append(yhat["LTD"])
        pred["E"].append(yhat["E_implied"])

        for k, c in cols_true.items():
            true[k].append(r.get(c))

    out = {"n": int(mask.sum())}
    for k in cols_true.keys():
        a = np.asarray(true[k], dtype=float)
        b = np.asarray(pred[k], dtype=float)
        out[f"{k}_MAE"] = mae(a, b)
        out[f"{k}_MAPE"] = mape(a, b)
    return out


def _fit_and_save(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    scope: str,
    key: str,
    out_path: str,
) -> Tuple[Params, Dict[str, object]]:
    theta, info = fit_theta_auto(
        df_trans=df,
        train_mask=train_mask.astype(int),
        mode=FIT_MODE,
        min_train=MIN_TF_TRAIN,
        tf_epochs=TF_EPOCHS,
        tf_lr=TF_LR,
        seed=RANDOM_SEED,
        loss_mode=TF_LOSS_MODE,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"scope": scope, "key": key, "theta": theta.to_dict(), "fit_info": info}
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return theta, info


def main():
    uni_path = _resolve_path(UNIVERSE_CSV)
    df_u = pd.read_csv(uni_path)

    # Report tickers (for PPT/report)
    report = df_u[df_u["label"].astype(str).str.lower().eq(TEST_LABEL)].copy()
    report_tickers = report["ticker"].tolist()

    # Training universe
    if EXCLUDE_TEST_FROM_TRAIN:
        train_u = df_u[df_u["label"].astype(str).str.lower().eq(TRAIN_LABEL)].copy()
    else:
        train_u = df_u.copy()

    os.makedirs(_resolve_path(THETA_DIR), exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    # -----------------------
    # GLOBAL θ (train tickers)
    # -----------------------
    df_glob_list = []
    for t in train_u["ticker"].tolist():
        dft = _load_one_ticker_transitions(t)
        if not dft.empty:
            df_glob_list.append(dft)
    df_glob = pd.concat(df_glob_list, ignore_index=True) if df_glob_list else pd.DataFrame()
    df_glob = clean_transitions(df_glob) if not df_glob.empty else df_glob

    train_mask_glob = _build_train_mask_within_ticker(df_glob, TRAIN_FRAC_WITHIN_TICKER) if not df_glob.empty else np.zeros((0,), bool)
    hold_mask_glob = ~train_mask_glob

    theta_g, info_g = _fit_and_save(
        df=df_glob,
        train_mask=train_mask_glob,
        scope="global",
        key="global",
        out_path=_resolve_path(os.path.join(THETA_DIR, "global.json")),
    )
    m_train = _one_step_eval(df_glob, theta_g, train_mask_glob)
    m_hold = _one_step_eval(df_glob, theta_g, hold_mask_glob)
    summary_rows.append(
        {"scope": "global", "key": "global", "n_total": int(len(df_glob)), "n_train": int(train_mask_glob.sum()), "n_hold": int(hold_mask_glob.sum()),
         **{f"train_{k}": v for k, v in m_train.items()}, **{f"hold_{k}": v for k, v in m_hold.items()}, **theta_g.to_dict(), **info_g}
    )

    # -----------------------
    # SECTOR θ (train tickers)
    # -----------------------
    # Fit only sectors that appear in report tickers (for downstream evaluation/plots).
    report_sectors = sorted(set(report["sector"].astype(str).tolist()))
    for sec in report_sectors:
        sec_train_tickers = train_u[train_u["sector"].astype(str).eq(sec)]["ticker"].tolist()
        if len(sec_train_tickers) == 0:
            # fallback: include all tickers in this sector (including test)
            sec_train_tickers = df_u[df_u["sector"].astype(str).eq(sec)]["ticker"].tolist()

        dfl = []
        for t in sec_train_tickers:
            dft = _load_one_ticker_transitions(t)
            if not dft.empty:
                dfl.append(dft)
        df_sec = pd.concat(dfl, ignore_index=True) if dfl else pd.DataFrame()
        df_sec = clean_transitions(df_sec) if not df_sec.empty else df_sec

        train_mask_sec = _build_train_mask_within_ticker(df_sec, TRAIN_FRAC_WITHIN_TICKER) if not df_sec.empty else np.zeros((0,), bool)
        hold_mask_sec = ~train_mask_sec

        sec_key = _safe_sector_name(sec)
        theta_s, info_s = _fit_and_save(
            df=df_sec,
            train_mask=train_mask_sec,
            scope="sector",
            key=sec,
            out_path=_resolve_path(os.path.join(THETA_DIR, "sector", f"{sec_key}.json")),
        )
        m_train = _one_step_eval(df_sec, theta_s, train_mask_sec)
        m_hold = _one_step_eval(df_sec, theta_s, hold_mask_sec)
        summary_rows.append(
            {"scope": "sector", "key": sec, "n_total": int(len(df_sec)), "n_train": int(train_mask_sec.sum()), "n_hold": int(hold_mask_sec.sum()),
             **{f"train_{k}": v for k, v in m_train.items()}, **{f"hold_{k}": v for k, v in m_hold.items()}, **theta_s.to_dict(), **info_s}
        )

    # -----------------------
    # COMPANY θ (report tickers)
    # -----------------------
    for t in report_tickers:
        dft = _load_one_ticker_transitions(t)
        dft = clean_transitions(dft) if not dft.empty else dft
        train_mask = _build_train_mask_within_ticker(dft, TRAIN_FRAC_WITHIN_TICKER) if not dft.empty else np.zeros((0,), bool)
        hold_mask = ~train_mask

        theta_c, info_c = _fit_and_save(
            df=dft,
            train_mask=train_mask,
            scope="company",
            key=t,
            out_path=_resolve_path(os.path.join(THETA_DIR, "company", f"{t.replace('/', '_')}.json")),
        )
        m_train = _one_step_eval(dft, theta_c, train_mask)
        m_hold = _one_step_eval(dft, theta_c, hold_mask)
        summary_rows.append(
            {"scope": "company", "key": t, "n_total": int(len(dft)), "n_train": int(train_mask.sum()), "n_hold": int(hold_mask.sum()),
             **{f"train_{k}": v for k, v in m_train.items()}, **{f"hold_{k}": v for k, v in m_hold.items()}, **theta_c.to_dict(), **info_c}
        )

    # Save summary
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(_resolve_path(os.path.join(THETA_DIR, "summary.csv")), index=False)

    meta = {
        "DATA_DIR": DATA_DIR,
        "FREQ": FREQ,
        "TRAIN_FRAC_WITHIN_TICKER": TRAIN_FRAC_WITHIN_TICKER,
        "FIT_MODE": FIT_MODE,
        "TF_EPOCHS": TF_EPOCHS,
        "TF_LR": TF_LR,
        "MIN_TF_TRAIN": MIN_TF_TRAIN,
        "TF_LOSS_MODE": TF_LOSS_MODE,
        "EXCLUDE_TEST_FROM_TRAIN": EXCLUDE_TEST_FROM_TRAIN,
        "RANDOM_SEED": RANDOM_SEED,
        "report_tickers": report_tickers,
    }
    Path(_resolve_path(os.path.join(THETA_DIR, "metadata.json"))).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] θ calibration complete. THETA_DIR={THETA_DIR}")
    print(f"     report_tickers={len(report_tickers)} | sectors_fitted={len(report_sectors)}")


if __name__ == "__main__":
    main()
