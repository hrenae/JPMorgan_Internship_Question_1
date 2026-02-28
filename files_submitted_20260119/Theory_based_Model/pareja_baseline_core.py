#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core utilities for the Pareja-style baseline balance-sheet simulator / forecaster.

This module centralizes:
  * The structural transition (NumPy).
  * Heuristic calibration of θ from one-step transitions.
  * Optional TensorFlow calibration with robust, scale-normalized loss.
  * Loading/cleaning transition tables produced by the data-preparation script.

Intended workflow
-----------------
1) Run `prepare_data_tsrl_baseline_v3.py` to generate annual transition CSVs.
2) Run `fit_theta_pareja_baseline_v3.py` to calibrate θ (company / sector / global).
3) Run `forecast_plot_pareja_baseline_v3.py` to produce multi-step forecasts and plots.

Notes
-----
* Accounting identities are enforced by construction:
      TA = sum(assets), TL = sum(liabilities), E = TA - TL.
* The model is stock-flow consistent (end-of-period convention):
      y_t = f(y_{t-1}, x_t).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def _is_finite(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _safe(x, default: float = 0.0) -> float:
    """Convert to float; if missing/non-finite -> default."""
    if _is_finite(x):
        return float(x)
    return float(default)


def _pos(x: float) -> float:
    """Non-negative projection (ReLU)."""
    if not _is_finite(x):
        return float("nan")
    return float(max(x, 0.0))


def _robust_median(arr, default: float) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float(default)
    return float(np.median(a))


def _robust_median_pos(arr, default: float) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return float(default)
    return float(np.median(a))


# -----------------------------
# Parameters (θ)
# -----------------------------
@dataclass
class Params:
    # Working capital (days)
    dso: float = 60.0
    dio: float = 30.0
    dpo: float = 60.0

    # Residual categories (scaled by Sales)
    oca_to_sales: float = 0.02
    onca_to_sales: float = 0.10
    ocl_to_sales: float = 0.02
    oncl_to_sales: float = 0.10

    # PPE dynamics
    capex_to_sales: float = 0.08
    dep_to_ppe: float = 0.06

    # Cash policy
    cash_min_to_sales: float = 0.02

    # Financing cost (end-of-period convention)
    r_st: float = 0.05
    r_lt: float = 0.05

    # Taxes & payout
    tax_rate: float = 0.20
    payout: float = 0.25

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


# -----------------------------
# Core transition (NumPy)
# -----------------------------
def step_numpy(y_prev: Dict[str, float], x_t: Dict[str, float], theta: Params) -> Dict[str, float]:
    """One-step transition y_t = f(y_{t-1}, x_t; θ).

    y_prev: levels at t-1
    x_t: flows over the period ending at t
    """
    # --- previous state (levels at t-1) ---
    C0 = _safe(y_prev.get("C"))
    AR0 = _safe(y_prev.get("AR"))
    Inv0 = _safe(y_prev.get("Inv"))
    AP0 = _safe(y_prev.get("AP"))
    K0 = _safe(y_prev.get("K"))
    STD0 = _safe(y_prev.get("STD"))
    LTD0 = _safe(y_prev.get("LTD"))

    E0 = _safe(y_prev.get("E_implied"), _safe(y_prev.get("E_gross_report"), 0.0))

    # --- drivers (flows over period t) ---
    S = _safe(x_t.get("S"))
    COGS = _safe(x_t.get("COGS"))
    OPEX = _safe(x_t.get("OPEX"))
    equity_issues = _safe(x_t.get("EquityIssues"))

    # Optional observed payouts/NI (used only in diagnostic E_flow when present)
    NI_obs = x_t.get("NI")
    Div_obs = x_t.get("Div")

    # --- forecast working capital from turnover-day policies ---
    AR = _pos(theta.dso / 365.0 * S)
    Inv = _pos(theta.dio / 365.0 * COGS)
    AP = _pos(theta.dpo / 365.0 * COGS)

    # --- other categories (scale with sales) ---
    OCA = _pos(theta.oca_to_sales * S)
    ONCA = _pos(theta.onca_to_sales * S)
    OCL = _pos(theta.ocl_to_sales * S)
    ONCL = _pos(theta.oncl_to_sales * S)

    # --- PPE dynamics ---
    dep = _pos(theta.dep_to_ppe * K0)
    capex = _pos(theta.capex_to_sales * S)
    K = _pos(K0 + capex - dep)

    # --- interest (end-of-period convention) ---
    interest = _pos(theta.r_st * STD0 + theta.r_lt * LTD0)

    # --- earnings (simple) ---
    ebit = (S - COGS - OPEX) - dep
    taxable_income = max(ebit - interest, 0.0)
    taxes = _pos(theta.tax_rate * taxable_income)
    net_income = ebit - interest - taxes

    # --- NWC change and CFO (indirect) ---
    d_nwc = (AR - AR0) + (Inv - Inv0) - (AP - AP0)
    cfo = net_income + dep - d_nwc
    cfi = -capex

    # payout policy: if no observed Div supplied, use payout ratio
    dividends = _pos(theta.payout * max(net_income, 0.0))
    if _is_finite(Div_obs):
        dividends = _pos(float(Div_obs))

    cash_pre = C0 + cfo + cfi + equity_issues - dividends

    # --- financing policy to maintain minimum cash buffer ---
    cash_min = _pos(theta.cash_min_to_sales * S)

    borrow = max(cash_min - cash_pre, 0.0)
    cash_after_borrow = cash_pre + borrow

    # repay with excess cash: repay STD then LTD
    excess = max(cash_after_borrow - cash_min, 0.0)
    repay_std = min(STD0, excess)
    excess2 = excess - repay_std
    repay_ltd = min(LTD0, excess2)

    STD = _pos(STD0 + borrow - repay_std)
    LTD = _pos(LTD0 - repay_ltd)
    C = _pos(cash_min + (excess2 - repay_ltd))

    # --- accounting identities by construction ---
    TA = C + AR + Inv + K + OCA + ONCA
    TL = AP + OCL + ONCL + STD + LTD
    E_implied = TA - TL

    # --- retained-earnings flow identity (diagnostic) ---
    NI_for_flow = float(NI_obs) if _is_finite(NI_obs) else net_income
    E_flow = E0 + NI_for_flow - dividends + equity_issues

    return {
        "C": C,
        "AR": AR,
        "Inv": Inv,
        "K": K,
        "AP": AP,
        "STD": STD,
        "LTD": LTD,
        "OCA": OCA,
        "ONCA": ONCA,
        "OCL": OCL,
        "ONCL": ONCL,
        "TA": TA,
        "TL": TL,
        "E_implied": E_implied,
        "E_flow": E_flow,
        "NetIncome_model": net_income,
        "Interest_model": interest,
        "Tax_model": taxes,
        "Capex_model": capex,
        "Dep_model": dep,
    }


# -----------------------------
# Loading & cleaning
# -----------------------------
def load_transitions_csv(path: str, freq: str = "A") -> pd.DataFrame:
    """Load a per-ticker transition CSV (produced by data prep) and filter by freq."""
    df = pd.read_csv(path)
    if "freq" in df.columns:
        df = df[df["freq"] == freq].copy()
    df["date_next"] = pd.to_datetime(df["date_next"])
    df["date_prev"] = pd.to_datetime(df["date_prev"])
    df = df.sort_values("date_next").reset_index(drop=True)
    return df


def clean_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop obviously invalid rows and enforce numeric types."""
    df = df.copy()

    # Required drivers
    for c in ["S", "COGS", "OPEX"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Required prev/next levels for our modeled block
    req_levels = [
        "C_prev", "AR_prev", "Inv_prev", "K_prev", "AP_prev", "STD_prev", "LTD_prev",
        "C_next", "AR_next", "Inv_next", "K_next", "AP_next", "STD_next", "LTD_next",
    ]
    for c in req_levels:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep rows where core drivers exist
    df = df[df["S"].notna() & df["COGS"].notna() & df["OPEX"].notna()].copy()

    # Drop rows where all prev levels missing
    prev_cols = ["C_prev", "AR_prev", "Inv_prev", "K_prev", "AP_prev", "STD_prev", "LTD_prev"]
    if set(prev_cols).issubset(df.columns):
        prev_all_missing = df[prev_cols].isna().all(axis=1)
        df = df[~prev_all_missing].copy()

    return df.reset_index(drop=True)


# -----------------------------
# Heuristic calibration (robust)
# -----------------------------
def estimate_params_heuristic(df_trans: pd.DataFrame, train_mask: np.ndarray) -> Params:
    """Heuristic calibration based on robust medians of observable accounting ratios."""
    base = Params()
    df = df_trans.loc[train_mask.astype(bool)].copy()
    if df.empty:
        return base

    S = df["S"].to_numpy(dtype=float)
    COGS = df["COGS"].to_numpy(dtype=float)

    # Days policies from observed levels (use NEXT levels)
    dso = (df["AR_next"].to_numpy(float) * 365.0) / np.where(S > 0, S, np.nan)
    dio = (df["Inv_next"].to_numpy(float) * 365.0) / np.where(COGS > 0, COGS, np.nan)
    dpo = (df["AP_next"].to_numpy(float) * 365.0) / np.where(COGS > 0, COGS, np.nan)

    # Capex ratio from observed CapEx
    capex = df["CapEx"].to_numpy(float) if "CapEx" in df.columns else np.full_like(S, np.nan)
    capex_to_sales = capex / np.where(S > 0, S, np.nan)

    # Depreciation ratio: infer from PPE reconciliation when possible
    K_prev = df["K_prev"].to_numpy(float)
    K_next = df["K_next"].to_numpy(float)
    dep_implied = (K_prev + capex - K_next)
    dep_to_ppe = np.where(K_prev > 0, np.clip(dep_implied, 0, np.inf) / K_prev, np.nan)

    # Effective interest rate
    debt_prev = df["STD_prev"].to_numpy(float) + df["LTD_prev"].to_numpy(float)
    I = df["I"].to_numpy(float) if "I" in df.columns else np.full_like(S, np.nan)
    r_eff = I / np.where(debt_prev > 0, debt_prev, np.nan)

    # Tax rate from reported tax and pre-tax income proxy (NI + Tax)
    Tax = df["Tax"].to_numpy(float) if "Tax" in df.columns else np.full_like(S, np.nan)
    NI = df["NI"].to_numpy(float) if "NI" in df.columns else np.full_like(S, np.nan)
    ebt = NI + Tax
    tax_rate = Tax / np.where(ebt > 0, ebt, np.nan)

    # Payout from dividends / NI (only when NI > 0)
    Div = df["Div"].to_numpy(float) if "Div" in df.columns else np.full_like(S, np.nan)
    payout = Div / np.where(NI > 0, NI, np.nan)

    # Residual categories from reported aggregates (if available)
    if {"TCA_next", "TA_next", "C_next", "AR_next", "Inv_next", "K_next"}.issubset(df.columns):
        oca = (df["TCA_next"].to_numpy(float) - (df["C_next"].to_numpy(float) + df["AR_next"].to_numpy(float) + df["Inv_next"].to_numpy(float)))
        onca = (df["TA_next"].to_numpy(float) - df["TCA_next"].to_numpy(float) - df["K_next"].to_numpy(float))
        oca_to_sales = np.clip(oca, 0, np.inf) / np.where(S > 0, S, np.nan)
        onca_to_sales = np.clip(onca, 0, np.inf) / np.where(S > 0, S, np.nan)
    else:
        oca_to_sales = np.array([np.nan])
        onca_to_sales = np.array([np.nan])

    if {"TCL_next", "TL_next", "AP_next", "STD_next", "LTD_next"}.issubset(df.columns):
        ocl = (df["TCL_next"].to_numpy(float) - (df["AP_next"].to_numpy(float) + df["STD_next"].to_numpy(float)))
        oncl = (df["TL_next"].to_numpy(float) - df["TCL_next"].to_numpy(float) - df["LTD_next"].to_numpy(float))
        ocl_to_sales = np.clip(ocl, 0, np.inf) / np.where(S > 0, S, np.nan)
        oncl_to_sales = np.clip(oncl, 0, np.inf) / np.where(S > 0, S, np.nan)
    else:
        ocl_to_sales = np.array([np.nan])
        oncl_to_sales = np.array([np.nan])

    # Cash buffer: lower quantile of observed cash / sales
    cash_min_to_sales = base.cash_min_to_sales
    if "C_next" in df.columns:
        cash_ratio = df["C_next"].to_numpy(float) / np.where(S > 0, S, np.nan)
        if np.isfinite(cash_ratio).any():
            cash_min_to_sales = float(np.nanquantile(cash_ratio, 0.20))

    return Params(
        dso=_robust_median_pos(dso, base.dso),
        dio=_robust_median_pos(dio, base.dio),
        dpo=_robust_median_pos(dpo, base.dpo),
        oca_to_sales=float(np.clip(_robust_median_pos(oca_to_sales, base.oca_to_sales), 0.0, 2.0)),
        onca_to_sales=float(np.clip(_robust_median_pos(onca_to_sales, base.onca_to_sales), 0.0, 5.0)),
        ocl_to_sales=float(np.clip(_robust_median_pos(ocl_to_sales, base.ocl_to_sales), 0.0, 2.0)),
        oncl_to_sales=float(np.clip(_robust_median_pos(oncl_to_sales, base.oncl_to_sales), 0.0, 5.0)),
        capex_to_sales=float(np.clip(_robust_median_pos(capex_to_sales, base.capex_to_sales), 0.0, 1.0)),
        dep_to_ppe=float(np.clip(_robust_median_pos(dep_to_ppe, base.dep_to_ppe), 0.0, 0.5)),
        cash_min_to_sales=float(np.clip(_safe(cash_min_to_sales, base.cash_min_to_sales), 0.0, 0.3)),
        r_st=float(np.clip(_robust_median_pos(r_eff, base.r_st), 0.0, 0.2)),
        r_lt=float(np.clip(_robust_median_pos(r_eff, base.r_lt), 0.0, 0.2)),
        tax_rate=float(np.clip(_robust_median_pos(tax_rate, base.tax_rate), 0.0, 0.5)),
        payout=float(np.clip(_robust_median_pos(payout, base.payout), 0.0, 1.0)),
    )


# -----------------------------
# TensorFlow fitting (optional)
# -----------------------------
def _tf_import():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except Exception:
        return None


def fit_params_tf(
    df_trans: pd.DataFrame,
    train_mask: np.ndarray,
    n_epochs: int = 2000,
    lr: float = 5e-3,
    seed: int = 0,
    init_theta: Optional[Params] = None,
    loss_mode: str = "relative",  # "relative" or "mse"
) -> Tuple[Optional[Params], Dict[str, object]]:
    """Fit θ by minimizing one-step-ahead loss on selected balance-sheet variables.

    loss_mode="relative" is recommended when fitting across many companies (scale robustness).
    """
    tf = _tf_import()
    if tf is None:
        return None, {"loss": float("nan"), "epoch": -1, "note": "TensorFlow not installed."}

    tf.random.set_seed(int(seed))

    y_cols = ["C_next", "AR_next", "Inv_next", "K_next", "AP_next", "STD_next", "LTD_next"]
    for c in y_cols:
        if c not in df_trans.columns:
            return None, {"loss": float("nan"), "epoch": -1, "note": f"Missing column {c}."}

    # Masks: use y masks if exist, else y notna
    m_cols = []
    for c in y_cols:
        m = f"mask_{c.replace('_next', '')}"
        if m not in df_trans.columns:
            df_trans[m] = df_trans[c].notna().astype(int)
        m_cols.append(m)

    Xcols = ["C_prev", "AR_prev", "Inv_prev", "K_prev", "AP_prev", "STD_prev", "LTD_prev",
             "E_prev", "S", "COGS", "OPEX", "EquityIssues"]

    for c in Xcols:
        if c not in df_trans.columns:
            df_trans[c] = np.nan

    X = {k: tf.convert_to_tensor(df_trans.loc[:, k].fillna(0.0).to_numpy(dtype=np.float32)) for k in Xcols}
    Y = tf.convert_to_tensor(df_trans.loc[:, y_cols].fillna(0.0).to_numpy(dtype=np.float32))
    M = tf.convert_to_tensor(df_trans.loc[:, m_cols].to_numpy(dtype=np.float32))

    # Train mask: (T,1)
    train_mask_tf = tf.convert_to_tensor(train_mask.astype(np.float32))[:, None]

    # Initialize around heuristic θ (stability)
    if init_theta is None:
        init_theta = Params()

    def inv_sigmoid(y):
        y = np.clip(y, 1e-6, 1 - 1e-6)
        return np.log(y / (1 - y))

    def inv_softplus(y):
        y = max(float(y), 1e-6)
        return np.log(np.expm1(y))

    # Unconstrained parameters p mapped to constrained θ
    p = {
        "dso": tf.Variable(inv_softplus(max(init_theta.dso - 1.0, 1e-3)), dtype=tf.float32),
        "dio": tf.Variable(inv_softplus(max(init_theta.dio - 1.0, 1e-3)), dtype=tf.float32),
        "dpo": tf.Variable(inv_softplus(max(init_theta.dpo - 1.0, 1e-3)), dtype=tf.float32),
        "oca": tf.Variable(inv_sigmoid(min(init_theta.oca_to_sales / 0.20, 0.999999)), dtype=tf.float32),
        "onca": tf.Variable(inv_sigmoid(min(init_theta.onca_to_sales / 1.00, 0.999999)), dtype=tf.float32),
        "ocl": tf.Variable(inv_sigmoid(min(init_theta.ocl_to_sales / 0.20, 0.999999)), dtype=tf.float32),
        "oncl": tf.Variable(inv_sigmoid(min(init_theta.oncl_to_sales / 1.00, 0.999999)), dtype=tf.float32),
        "capex": tf.Variable(inv_sigmoid(min(init_theta.capex_to_sales / 0.30, 0.999999)), dtype=tf.float32),
        "dep": tf.Variable(inv_sigmoid(min(init_theta.dep_to_ppe / 0.40, 0.999999)), dtype=tf.float32),
        "cashmin": tf.Variable(inv_sigmoid(min(init_theta.cash_min_to_sales / 0.20, 0.999999)), dtype=tf.float32),
        "r_st": tf.Variable(inv_softplus(max(init_theta.r_st / 0.20, 1e-6)), dtype=tf.float32),
        "r_lt": tf.Variable(inv_softplus(max(init_theta.r_lt / 0.20, 1e-6)), dtype=tf.float32),
        "tax": tf.Variable(inv_sigmoid(min(init_theta.tax_rate / 0.50, 0.999999)), dtype=tf.float32),
        "payout": tf.Variable(inv_sigmoid(min(init_theta.payout / 0.80, 0.999999)), dtype=tf.float32),
    }

    def to_theta():
        softplus = tf.nn.softplus
        sigmoid = tf.nn.sigmoid
        return {
            "dso": softplus(p["dso"]) + 1.0,
            "dio": softplus(p["dio"]) + 1.0,
            "dpo": softplus(p["dpo"]) + 1.0,
            "oca": sigmoid(p["oca"]) * 0.20,
            "onca": sigmoid(p["onca"]) * 1.00,
            "ocl": sigmoid(p["ocl"]) * 0.20,
            "oncl": sigmoid(p["oncl"]) * 1.00,
            "capex": sigmoid(p["capex"]) * 0.30,
            "dep": sigmoid(p["dep"]) * 0.40,
            "cashmin": sigmoid(p["cashmin"]) * 0.20,
            "r_st": softplus(p["r_st"]) * 0.20,
            "r_lt": softplus(p["r_lt"]) * 0.20,
            "tax": sigmoid(p["tax"]) * 0.50,
            "payout": sigmoid(p["payout"]) * 0.80,
        }

    def one_step_pred(theta):
        C0, AR0, Inv0, K0 = X["C_prev"], X["AR_prev"], X["Inv_prev"], X["K_prev"]
        AP0 = X["AP_prev"]
        STD0, LTD0 = X["STD_prev"], X["LTD_prev"]
        S, COGS, OPEX = X["S"], X["COGS"], X["OPEX"]
        equity_issues = X["EquityIssues"]

        AR = tf.nn.relu(theta["dso"] / 365.0 * S)
        Inv = tf.nn.relu(theta["dio"] / 365.0 * COGS)
        AP = tf.nn.relu(theta["dpo"] / 365.0 * COGS)

        OCA = tf.nn.relu(theta["oca"] * S)
        ONCA = tf.nn.relu(theta["onca"] * S)
        OCL = tf.nn.relu(theta["ocl"] * S)
        ONCL = tf.nn.relu(theta["oncl"] * S)

        dep = tf.nn.relu(theta["dep"] * K0)
        capex = tf.nn.relu(theta["capex"] * S)
        K = tf.nn.relu(K0 + capex - dep)

        interest = theta["r_st"] * STD0 + theta["r_lt"] * LTD0

        ebit = (S - COGS - OPEX) - dep
        taxable = tf.nn.relu(ebit - interest)
        taxes = theta["tax"] * taxable
        ni = ebit - interest - taxes

        d_nwc = (AR - AR0) + (Inv - Inv0) - (AP - AP0)
        cfo = ni + dep - d_nwc
        cfi = -capex

        dividends = theta["payout"] * tf.nn.relu(ni)

        cash_pre = C0 + cfo + cfi + equity_issues - dividends
        cash_min = theta["cashmin"] * S

        borrow = tf.nn.relu(cash_min - cash_pre)
        cash_after_borrow = cash_pre + borrow

        excess = tf.nn.relu(cash_after_borrow - cash_min)
        repay_std = tf.minimum(STD0, excess)
        excess2 = excess - repay_std
        repay_ltd = tf.minimum(LTD0, excess2)

        STD = STD0 + borrow - repay_std
        LTD = LTD0 - repay_ltd
        C = cash_min + (excess2 - repay_ltd)

        return tf.stack([C, AR, Inv, K, AP, STD, LTD], axis=1)

    opt = tf.keras.optimizers.Adam(learning_rate=float(lr))

    best = {"loss": float("inf"), "epoch": -1}
    best_weights = None

    # Precompute a reasonable scale for "relative" loss:
    # denom = |y_true| + |y_prev| + 1, per variable.
    if loss_mode == "relative":
        Y_abs = tf.abs(Y)
        Yprev = tf.stack(
            [tf.abs(X["C_prev"]), tf.abs(X["AR_prev"]), tf.abs(X["Inv_prev"]), tf.abs(X["K_prev"]),
             tf.abs(X["AP_prev"]), tf.abs(X["STD_prev"]), tf.abs(X["LTD_prev"])],
            axis=1,
        )
        denom = Y_abs + Yprev + 1.0
    else:
        denom = tf.ones_like(Y)

    for epoch in range(int(n_epochs)):
        with tf.GradientTape() as tape:
            th = to_theta()
            Yhat = one_step_pred(th)
            diff2 = tf.square((Yhat - Y) / denom)
            w = M * train_mask_tf
            loss_data = tf.reduce_sum(diff2 * w) / tf.maximum(tf.reduce_sum(w), 1.0)
            reg = 1e-4 * tf.add_n([tf.square(v) for v in p.values()])
            loss = loss_data + reg

        l = float(loss.numpy())
        if not np.isfinite(l):
            return None, {"loss": float("nan"), "epoch": epoch, "note": "TF loss became non-finite."}

        grads = tape.gradient(loss, list(p.values()))
        opt.apply_gradients(zip(grads, list(p.values())))

        if l < best["loss"]:
            best = {"loss": l, "epoch": epoch}
            best_weights = {k: v.numpy().copy() for k, v in p.items()}

    # Restore best weights
    if best_weights is not None:
        for k, v in p.items():
            v.assign(best_weights[k])

    th = to_theta()
    fitted = Params(
        dso=float(th["dso"].numpy()),
        dio=float(th["dio"].numpy()),
        dpo=float(th["dpo"].numpy()),
        oca_to_sales=float(th["oca"].numpy()),
        onca_to_sales=float(th["onca"].numpy()),
        ocl_to_sales=float(th["ocl"].numpy()),
        oncl_to_sales=float(th["oncl"].numpy()),
        capex_to_sales=float(th["capex"].numpy()),
        dep_to_ppe=float(th["dep"].numpy()),
        cash_min_to_sales=float(th["cashmin"].numpy()),
        r_st=float(th["r_st"].numpy()),
        r_lt=float(th["r_lt"].numpy()),
        tax_rate=float(th["tax"].numpy()),
        payout=float(th["payout"].numpy()),
    )

    if any(not np.isfinite(v) for v in fitted.to_dict().values()):
        return None, {"loss": float("nan"), "epoch": best["epoch"], "note": "TF produced non-finite parameters."}

    return fitted, {"loss": best["loss"], "epoch": best["epoch"], "note": "", "loss_mode": loss_mode}


def fit_theta_auto(
    df_trans: pd.DataFrame,
    train_mask: np.ndarray,
    mode: str = "auto",  # "auto" | "heuristic" | "tf"
    min_train: int = 30,
    tf_epochs: int = 2000,
    tf_lr: float = 5e-3,
    seed: int = 0,
    loss_mode: str = "relative",
) -> Tuple[Params, Dict[str, object]]:
    """Convenience wrapper: heuristic init + optional TF fine-tune."""
    theta_h = estimate_params_heuristic(df_trans, train_mask)
    info: Dict[str, object] = {"fit_mode_used": "heuristic", "loss": float("nan"), "epoch": -1, "note": ""}

    if mode not in {"auto", "heuristic", "tf"}:
        mode = "auto"

    do_tf = (mode == "tf") or (mode == "auto" and int(train_mask.sum()) >= int(min_train))
    if do_tf:
        theta_tf, tf_info = fit_params_tf(
            df_trans=df_trans,
            train_mask=train_mask,
            n_epochs=tf_epochs,
            lr=tf_lr,
            seed=seed,
            init_theta=theta_h,
            loss_mode=loss_mode,
        )
        if theta_tf is not None:
            info = {"fit_mode_used": "tf", **tf_info}
            return theta_tf, info
        # TF failure -> fall back
        info = {"fit_mode_used": "heuristic", **tf_info}

    return theta_h, info


# -----------------------------
# Metrics
# -----------------------------
def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(a[m] - b[m])))


def mape(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 1e-12)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[m] - b[m]) / a[m])))


def compute_metrics_one_step(df_pred: pd.DataFrame, idx: np.ndarray) -> Dict[str, float]:
    """Metrics for one-step predictions: assumes columns *_true and *_pred exist."""
    out: Dict[str, float] = {}
    cols = [
        ("C", "C_true", "C_pred"),
        ("AR", "AR_true", "AR_pred"),
        ("Inv", "Inv_true", "Inv_pred"),
        ("K", "K_true", "K_pred"),
        ("AP", "AP_true", "AP_pred"),
        ("STD", "STD_true", "STD_pred"),
        ("LTD", "LTD_true", "LTD_pred"),
        ("E", "E_true", "E_pred"),
    ]
    for name, tcol, pcol in cols:
        a = df_pred.loc[idx, tcol].to_numpy(dtype=float)
        b = df_pred.loc[idx, pcol].to_numpy(dtype=float)
        out[f"{name}_MAE"] = mae(a, b)
        out[f"{name}_MAPE"] = mape(a, b)
    out["E_pred_vs_E_flow_MAE"] = mae(
        df_pred.loc[idx, "E_pred"].to_numpy(float),
        df_pred.loc[idx, "E_flow"].to_numpy(float),
    )
    return out
