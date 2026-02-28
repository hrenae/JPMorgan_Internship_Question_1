#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theory_uncond_baseline_rolling.py

Theory baseline for the unconditional setting.

Two modes are supported:

(1) backtest (default):
    Given the first `--warmup` observed periods, perform *rolling* one-step-ahead
    forecasts for all subsequent periods that exist in the cached panel
    (data_dir/panels/<ticker>.csv). This aligns each prediction with an
    *existing* timestamp so it can be compared against ground truth.

    - Sales forecast: AR(1) on log S estimated from the *available* historical
      observations up to the current step (expanding window).
    - Policy (theta): sector-wise medians estimated on the training set
      (sector_theta_medians.json).
    - Accounting simulator: simulate_step() produces the next-period balance-sheet state.

    Missing ground truth (e.g., due to yfinance limitations) is handled as:
      - true value is written as 0.0
      - corresponding mask column is 0
    The rolling forecast continues through gaps by using the model-predicted
    state (and, if needed, predicted logS) as the new starting point.

(2) future:
    Legacy behavior: start from the last observed period and roll forward
    `--rollout_steps` steps into the future (no ground truth).

Example (backtest):
  python theory_uncond_baseline_rolling.py \
    --data_dir data_uncond \
    --out_dir results_theory \
    --mode backtest \
    --warmup 3 \
    --disable_interest_for_banks

Example (future rollout):
  python theory_uncond_baseline_rolling.py \
    --data_dir data_uncond \
    --out_dir results_theory \
    --mode future \
    --rollout_steps 8 \
    --disable_interest_for_banks
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd



# ----------------------------
# Numerical safeguards (baseline stability)
# ----------------------------
# Clip log-sales before exponentiation to avoid extreme S when logS is corrupted.
LOGS_CLIP = (-40.0, 40.0)   # exp(40) ≈ 2e17
# Hard cap on state magnitudes to prevent overflow (especially if later cast to float32 elsewhere).
STATE_ABS_CAP = 1e30
# Reasonable bounds on period length (days) used in working-capital mappings.
DT_MIN = 1.0
DT_MAX = 400.0


# The simulator uses a 12D state vector.
STATE_COLS = ["C","AR","Inv","OCA","K","ONCA","AP","OCL","STD","LTD","ONCL","E_flow"]


# ----------------------------
# Core simulator
# ----------------------------

def simulate_step(
    state: np.ndarray,
    logS: float,
    theta: Dict[str, float],
    period_days: float,
    is_financial: bool = False,
    disable_interest: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """One-step accounting transition.

    Notes on implementation choices (aligned with the TeX write-up):
      - Interest uses lagged debt stocks with potentially different ST/LT rates.
      - Period length dt is treated as the *time between consecutive state dates*.
      - Numerical guards (logS clipping, state clipping) are included to keep the
        baseline robust when the input panel has missing/corrupted entries.
    """
    st = np.asarray(state, dtype=float).reshape(-1)
    if st.size != 12:
        raise ValueError(f"state must be 12D, got {st.size}")

    C, AR_prev, Inv_prev, OCA_prev, K_prev, ONCA_prev, AP_prev, OCL_prev, STD_prev, LTD_prev, ONCL_prev, E_prev = st.tolist()

    # ---- safe dt and sales ----
    dt = float(period_days) if np.isfinite(period_days) else 365.0
    if dt <= 0:
        dt = 91.25
    dt = float(np.clip(dt, DT_MIN, DT_MAX))

    logS_in = float(logS) if np.isfinite(logS) else 0.0
    logS_safe = float(np.clip(logS_in, LOGS_CLIP[0], LOGS_CLIP[1]))
    S = float(np.exp(logS_safe))

    # ---- load theta (NaN-safe + domain-clipped; bounds aligned with TARGET_SPECS) ----
    def _finite_or(x: Any, default: float) -> float:
        try:
            x = float(x)
        except Exception:
            return float(default)
        return float(x) if np.isfinite(x) else float(default)

    def _clip(x: Any, lo: float, hi: float, default: float) -> float:
        x = _finite_or(x, default)
        return float(np.clip(x, lo, hi))

    m_gross = _clip(theta.get("m_gross", 0.5), 0.0, 1.0, 0.5)
    m_opex  = _clip(theta.get("m_opex", 0.2), 0.0, 1.0, 0.2)

    DSO     = _clip(theta.get("DSO", 60.0), 0.0, 720.0, 60.0)
    DIO     = _clip(theta.get("DIO", 60.0), 0.0, 720.0, 60.0)
    DPO     = _clip(theta.get("DPO", 60.0), 0.0, 720.0, 60.0)

    a_OCA   = _clip(theta.get("alpha_OCA", 0.05), 0.0, 0.50, 0.05)
    a_ONCA  = _clip(theta.get("alpha_ONCA", 0.20), 0.0, 1.00, 0.20)
    a_OCL   = _clip(theta.get("alpha_OCL", 0.05), 0.0, 0.50, 0.05)
    a_ONCL  = _clip(theta.get("alpha_ONCL", 0.20), 0.0, 1.00, 0.20)

    kappa   = _clip(theta.get("kappa", 0.05), 0.0, 0.80, 0.05)
    delta   = _clip(theta.get("delta", 0.05), 0.0, 0.50, 0.05)

    payout  = _clip(theta.get("payout", 0.0), 0.0, 1.00, 0.0)
    neteq_to_sales = _clip(theta.get("neteq_to_sales", 0.0), -1.0, 1.0, 0.0)
    phi     = _clip(theta.get("phi", 0.05), 0.0, 0.50, 0.05)

    # Interest rates: prefer r_ST/r_LT; fall back to r_debt if present.
    r_debt  = theta.get("r_debt", 0.05)
    r_ST    = _clip(theta.get("r_ST", r_debt), 0.0, 0.50, 0.05)
    r_LT    = _clip(theta.get("r_LT", r_debt), 0.0, 0.50, 0.05)

    tau     = _clip(theta.get("tau", 0.2), 0.0, 0.50, 0.2)
    # Banks/financials convention: ignore gross margin + working-capital channels.
    if is_financial:
        m_gross = 1.0
        DSO = DIO = DPO = 0.0

    # ---- income statement ----
    COGS = (1.0 - m_gross) * S
    OPEX = m_opex * S
    Int = 0.0 if disable_interest else (r_ST * STD_prev + r_LT * LTD_prev)

    # ---- map policy to stocks ----
    AR  = (DSO / dt) * S
    Inv = (DIO / dt) * COGS if COGS != 0 else 0.0
    AP  = (DPO / dt) * COGS if COGS != 0 else 0.0

    OCA  = a_OCA * S
    ONCA = a_ONCA * S
    OCL  = a_OCL * S
    ONCL = a_ONCL * S

    Dep = delta * K_prev
    CapEx = kappa * S
    K = K_prev + CapEx - Dep

    EBIT = (S - COGS - OPEX) - Dep
    TaxBase = max(EBIT - Int, 0.0)
    Tax = tau * TaxBase
    NI = EBIT - Int - Tax

    Div  = payout * max(NI, 0.0)

    # Net equity issuance/buyback:
    # - If the policy provides an absolute NetEq, use it.
    # - Otherwise interpret neteq_to_sales as a scale-free control and convert via sales.
    NetEq_raw = theta.get("NetEq", np.nan)
    if isinstance(NetEq_raw, (int, float, np.floating)) and np.isfinite(float(NetEq_raw)):
        NetEq = float(NetEq_raw)
        neteq_mode = 1  # absolute
    else:
        NetEq = neteq_to_sales * S
        neteq_mode = 0  # ratio-to-sales

    # ---- cash flow statements ----
    dAR   = AR   - AR_prev
    dInv  = Inv  - Inv_prev
    dOCA  = OCA  - OCA_prev
    dAP   = AP   - AP_prev
    dOCL  = OCL  - OCL_prev
    dONCA = ONCA - ONCA_prev
    dONCL = ONCL - ONCL_prev

    dNWC = dAR + dInv + dOCA - dAP - dOCL
    CFO = NI + Dep - dNWC
    CFI = -CapEx - dONCA

    # ---- financing closure via cash minimum ----
    C_min = phi * S
    C_pre = C + CFO + CFI + dONCL + NetEq - Div

    Borrow = max(C_min - C_pre, 0.0)
    Excess = max(C_pre - C_min, 0.0)

    Repay_st = min(STD_prev, Excess)
    Repay_lt = min(LTD_prev, max(Excess - Repay_st, 0.0))

    STD = STD_prev + Borrow - Repay_st
    LTD = LTD_prev - Repay_lt
    C_new = C_pre + Borrow - Repay_st - Repay_lt

    dDebt = (STD - STD_prev) + (LTD - LTD_prev)
    CFF = dDebt + dONCL + NetEq - Div

    # ---- equity flow state ----
    E = E_prev + NI - Div + NetEq

    # ---- diagnostics ----
    TA = C_new + AR + Inv + OCA + K + ONCA
    TL = AP + OCL + STD + LTD + ONCL

    bs_resid = TA - TL - E
    cash_resid = C_new - (C + CFO + CFI + CFF)

    next_state = np.asarray([C_new, AR, Inv, OCA, K, ONCA, AP, OCL, STD, LTD, ONCL, E], dtype=float)

    # Numerical sanitization (recorded via diag flags)
    clipped = 0
    nonfinite = 0
    if not np.all(np.isfinite(next_state)):
        nonfinite = 1
        next_state = np.nan_to_num(next_state, nan=0.0, posinf=STATE_ABS_CAP, neginf=-STATE_ABS_CAP)
        clipped = 1
    if np.any(np.abs(next_state) > STATE_ABS_CAP):
        next_state = np.clip(next_state, -STATE_ABS_CAP, STATE_ABS_CAP)
        clipped = 1

    diag = dict(
        # key primitives
        dt=dt, logS_in=logS_in, logS_safe=logS_safe, S=S,
        # controls
        m_gross=m_gross, m_opex=m_opex, DSO=DSO, DIO=DIO, DPO=DPO,
        alpha_OCA=a_OCA, alpha_ONCA=a_ONCA, alpha_OCL=a_OCL, alpha_ONCL=a_ONCL,
        kappa=kappa, delta=delta, payout=payout, neteq_to_sales=neteq_to_sales, phi=phi,
        r_ST=r_ST, r_LT=r_LT, tau=tau, neteq_mode=float(neteq_mode),
        # statements
        COGS=COGS, OPEX=OPEX, Int=Int, Tax=Tax, NI=NI, Div=Div, NetEq=NetEq,
        CFO=CFO, CFI=CFI, CFF=CFF,
        # balance / checks
        TA=TA, TL=TL, E=E,
        bs_resid=bs_resid, cash_resid=cash_resid,
        Borrow=Borrow, Repay_st=Repay_st, Repay_lt=Repay_lt, C_min=C_min,
        # stability flags
        nonfinite_state=float(nonfinite), clipped_state=float(clipped),
    )
    return next_state, diag


# ----------------------------
# AR(1) helpers
# ----------------------------

def fit_ar1(logS: np.ndarray) -> Tuple[float, float]:
    """Fit y_t = c + phi y_{t-1} in least squares."""
    y = logS[1:]
    x = logS[:-1]
    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    c, phi = float(beta[0]), float(beta[1])
    phi = max(min(phi, 0.99), -0.99)
    return c, phi

def ar1_next(logS_t: float, c: float, phi: float) -> float:
    return float(c + phi * logS_t)

def ar1_path(logS_t: float, c: float, phi: float, H: int) -> np.ndarray:
    out = np.zeros((H,), dtype=float)
    cur = logS_t
    for k in range(H):
        cur = c + phi * cur
        out[k] = cur
    return out


# ----------------------------
# Panel IO + robust field access
# ----------------------------

def _safe_name(ticker: str) -> str:
    return str(ticker).replace("/", "_").replace(":", "_")

def load_panel(data_dir: str, ticker: str) -> pd.DataFrame:
    p = os.path.join(data_dir, "panels", f"{_safe_name(ticker)}.csv")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def _is_financial_sector(sector: str) -> bool:
    s = (sector or "").lower()
    return ("financial" in s) or ("bank" in s)

def _get_first_finite(row: pd.Series, candidates: List[str]) -> float:
    for c in candidates:
        if c in row.index:
            v = row.get(c, np.nan)
            try:
                v = float(v)
            except Exception:
                continue
            if np.isfinite(v):
                return v
    return float("nan")

def _get_logS(row: pd.Series) -> float:
    v = _get_first_finite(row, ["logS"])
    if np.isfinite(v):
        return v
    s = _get_first_finite(row, ["S", "Sales", "revenue"])
    if np.isfinite(s) and s > 0:
        return float(np.log(max(s, 1e-12)))
    return float("nan")

def _truth_state_from_row(row: pd.Series) -> np.ndarray:
    """Extract the 12D state from a panel row. Missing entries become NaN."""
    C   = _get_first_finite(row, ["C"])
    AR  = _get_first_finite(row, ["AR"])
    Inv = _get_first_finite(row, ["Inv"])

    # Various preprocessing versions use implied working-capital accounts.
    OCA = _get_first_finite(row, ["OCA", "OCA_implied"])
    K   = _get_first_finite(row, ["K"])
    ONCA= _get_first_finite(row, ["ONCA", "ONCA_implied"])

    AP  = _get_first_finite(row, ["AP"])
    OCL = _get_first_finite(row, ["OCL", "OCL_implied"])
    STD = _get_first_finite(row, ["STD"])
    LTD = _get_first_finite(row, ["LTD"])
    ONCL= _get_first_finite(row, ["ONCL", "ONCL_implied"])

    # For simulator state we prefer identity-consistent equity when component blocks are available.
    assets = [C, AR, Inv, OCA, K, ONCA]
    liabs  = [AP, OCL, STD, LTD, ONCL]
    if all(np.isfinite(x) for x in assets) and all(np.isfinite(x) for x in liabs):
        E = float(sum(assets) - sum(liabs))
    else:
        E = _get_first_finite(row, ["E_flow", "E_gross_report", "E_report", "E_implied", "E0"])
        if not np.isfinite(E):
            TA = _get_first_finite(row, ["TA"])
            TL = _get_first_finite(row, ["TL"])
            if np.isfinite(TA) and np.isfinite(TL):
                E = TA - TL
    return np.asarray([C, AR, Inv, OCA, K, ONCA, AP, OCL, STD, LTD, ONCL, E], dtype=np.float32)

def _has_full_truth_state(row: pd.Series) -> bool:
    st = _truth_state_from_row(row)
    return bool(np.all(np.isfinite(st)))

def _nan_to_zero_with_mask(x: float) -> Tuple[float, int]:
    if np.isfinite(x):
        return float(x), 1
    return 0.0, 0


# ----------------------------
# Mode (1): Backtest
# ----------------------------

def run_backtest_one_ticker(
    df: pd.DataFrame,
    ticker: str,
    sector: str,
    theta: Dict[str, float],
    warmup: int = 3,
    min_ar1_points: int = 3,
    disable_interest: bool = False,
) -> pd.DataFrame:
    """Rolling one-step-ahead backtest aligned to existing panel rows."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    n = len(df)
    if n <= warmup:
        raise ValueError(f"{ticker}: need > warmup rows; got n={n}, warmup={warmup}")

    # initialize at the (warmup-1)-th observed row
    idx0 = warmup - 1
    row0 = df.iloc[idx0]

    st = _truth_state_from_row(row0)
    # If some warmup truth fields are missing, fill with zeros so simulation can start.
    st = np.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # Enforce identity-consistent initial equity for the internal state.
    TA0 = float(st[0] + st[1] + st[2] + st[3] + st[4] + st[5])
    TL0 = float(st[6] + st[7] + st[8] + st[9] + st[10])
    st[11] = float(TA0 - TL0)

    logS_t = _get_logS(row0)
    if not np.isfinite(logS_t):
        # fall back to logS computed from the latest finite S before idx0
        hist_logS = np.asarray([_get_logS(df.iloc[i]) for i in range(idx0 + 1)], dtype=float)
        hist_logS = hist_logS[np.isfinite(hist_logS)]
        logS_t = float(hist_logS[-1]) if hist_logS.size > 0 else 0.0

    rows: List[Dict[str, Any]] = []

    # --- also output the warmup ground-truth points for convenient plotting ---
    # These rows are *not* forecasts; we store the observed truth (with masks),
    # and leave forecast columns as NaN automatically.
    for i in range(0, idx0 + 1):
        r = df.iloc[i]

        logS_true_i = _get_logS(r)
        logS_true_val_i, mask_logS_i = _nan_to_zero_with_mask(logS_true_i)

        S_true_i = _get_first_finite(r, ["S"])
        S_true_val_i, mask_S_i = _nan_to_zero_with_mask(S_true_i)

        st_true_i = _truth_state_from_row(r)
        st_true_vals_i: List[float] = []
        st_true_masks_i: List[int] = []
        for v in st_true_i.tolist():
            vv, mm = _nan_to_zero_with_mask(float(v) if np.isfinite(v) else float("nan"))
            st_true_vals_i.append(vv)
            st_true_masks_i.append(mm)

        period_days_i = _get_first_finite(r, ["period_days"])
        if not np.isfinite(period_days_i):
            period_days_i = 365.0

        rec_warm: Dict[str, Any] = dict(
            ticker=str(ticker),
            sector=str(sector),
            date=str(pd.to_datetime(r["date"]).date()),
            idx=int(i),
            step=int(i - idx0),  # negative .. 0 for warmup
            period_days=float(period_days_i),
            logS_pred=float("nan"),
            logS_true=float(logS_true_val_i),
            mask_logS=int(mask_logS_i),
            S_true=float(S_true_val_i),
            mask_S=int(mask_S_i),
            **{f"theta_{k}": float(v) for k, v in theta.items()},
        )
        # --- warmup truth for theta + statement flows (for evaluation/plotting) ---
        THETA_KEYS_EVAL = [
            "m_gross","m_opex","DSO","DIO","DPO",
            "alpha_OCA","alpha_ONCA","alpha_OCL","alpha_ONCL",
            "kappa","delta","payout","neteq_to_sales","phi",
            "r_ST","r_LT","tau",
        ]
        theta_true = {}
        theta_mask = {}
        for k in THETA_KEYS_EVAL:
            v = _get_first_finite(r, [k])
            vv, mm = _nan_to_zero_with_mask(v)
            if _is_financial_sector(sector) and (k in ["m_gross","DSO","DIO","DPO"]):
                vv, mm = 0.0, 0
            theta_true[k] = vv
            theta_mask[k] = mm
        rec_warm.update({f"theta_true_{k}": float(v) for k, v in theta_true.items()})
        rec_warm.update({f"mask_theta_{k}": int(m) for k, m in theta_mask.items()})

        def _abs_if_finite(x: float) -> float:
            return float(abs(x)) if np.isfinite(x) else float("nan")

        flow_specs = {
            "COGS": (["COGS"], None),
            "OPEX": (["OPEX"], None),
            "Tax":  (["Tax"], None),
            "NI":   (["NI"], None),
            "Div":  (["Div"], _abs_if_finite),
            "Int":  (["I"], _abs_if_finite),   # panel uses I; model uses Int >= 0
            "TA":   (["TA"], None),
            "TL":   (["TL"], None),
        }
        for name, (cols, fn) in flow_specs.items():
            v = _get_first_finite(r, cols)
            if fn is not None:
                v = fn(v)
            vv, mm = _nan_to_zero_with_mask(v)
            rec_warm[f"{name}_true"] = float(vv)
            rec_warm[f"mask_{name}"] = int(mm)

        # NetEq truth: EquityIssues - Buyback
        eq = _get_first_finite(r, ["EquityIssues"])
        bb = _get_first_finite(r, ["Buyback"])
        mm = int(np.isfinite(eq) or np.isfinite(bb))
        eqv = float(eq) if np.isfinite(eq) else 0.0
        bbv = float(bb) if np.isfinite(bb) else 0.0
        rec_warm["NetEq_true"] = float(eqv - bbv) if mm else 0.0
        rec_warm["mask_NetEq"] = int(mm)

        rec_warm.update({f"state_true_{STATE_COLS[j]}": float(st_true_vals_i[j]) for j in range(len(STATE_COLS))})
        rec_warm.update({f"mask_state_{STATE_COLS[j]}": int(st_true_masks_i[j]) for j in range(len(STATE_COLS))})
        rows.append(rec_warm)

    # rolling one-step ahead: predict idx -> idx+1 for idx in [idx0, n-2]
    for idx in range(idx0, n - 1):
        row_cur = df.iloc[idx]
        row_nxt = df.iloc[idx + 1]
        # Synchronize logS to truth whenever available; synchronize the full state only when complete.
        logS_cur_truth = _get_logS(row_cur)
        if np.isfinite(logS_cur_truth):
            logS_t = float(logS_cur_truth)
        if _has_full_truth_state(row_cur):
            st = _truth_state_from_row(row_cur).astype(float)
        # Fit AR(1) on *available* historical observations up to the current index.
        hist_logS = np.asarray([_get_logS(df.iloc[i]) for i in range(idx + 1)], dtype=float)
        hist_logS = hist_logS[np.isfinite(hist_logS)]
        if hist_logS.size >= min_ar1_points:
            c, phi = fit_ar1(hist_logS)
        else:
            # Too few points: random-walk baseline.
            c, phi = 0.0, 1.0

        logS_pred = ar1_next(float(logS_t), c, phi)
        # Period length (days):
        # Prefer calendar delta between consecutive statement dates. This prevents
        # spurious 365-day jumps in MIX mode when annual rows override year-end quarters.
        period_days_panel = _get_first_finite(row_nxt, ["period_days"])
        if not np.isfinite(period_days_panel):
            period_days_panel = _get_first_finite(row_cur, ["period_days"])
        if not np.isfinite(period_days_panel):
            period_days_panel = 365.0

        try:
            dcur = pd.to_datetime(row_cur["date"])
            dnxt = pd.to_datetime(row_nxt["date"])
            period_days_date = float((dnxt - dcur).days)
        except Exception:
            period_days_date = float("nan")

        if np.isfinite(period_days_date) and period_days_date > 0:
            period_days = period_days_date
        else:
            period_days = float(period_days_panel)

        period_days = float(np.clip(period_days, DT_MIN, DT_MAX))
        st_pred_next, diag = simulate_step(
            st, float(logS_pred), theta, float(period_days),
            is_financial=_is_financial_sector(sector),
            disable_interest=disable_interest,
        )

        # --- ground truth at idx+1 (may be missing) ---
        logS_true = _get_logS(row_nxt)
        logS_true_val, mask_logS = _nan_to_zero_with_mask(logS_true)

        # raw S (optional; useful for diagnostics)
        S_true = _get_first_finite(row_nxt, ["S"])
        S_true_val, mask_S = _nan_to_zero_with_mask(S_true)

        st_true = _truth_state_from_row(row_nxt)
        st_true_vals: List[float] = []
        st_true_masks: List[int] = []
        for v in st_true.tolist():
            vv, mm = _nan_to_zero_with_mask(float(v) if np.isfinite(v) else float("nan"))
            st_true_vals.append(vv)
            st_true_masks.append(mm)

        # record
        rec: Dict[str, Any] = dict(
            ticker=str(ticker),
            sector=str(sector),
            date=str(pd.to_datetime(row_nxt["date"]).date()),
            idx=int(idx + 1),
            step=int((idx + 1) - idx0),   # 1,2,3,...
            period_days=float(period_days),
            period_days_panel=float(period_days_panel),
            period_days_date=float(period_days_date) if np.isfinite(period_days_date) else float("nan"),
            logS_pred=float(logS_pred),
            logS_true=float(logS_true_val),
            mask_logS=int(mask_logS),
            S_true=float(S_true_val),
            mask_S=int(mask_S),
            **{f"theta_{k}": float(v) for k, v in theta.items()},
        )


        # --- optional: ground truth for theta channels and statement flows at idx+1 ---
        # These are *not* required to run the simulator; they are saved only for evaluation/plotting.
        # Missing values are written as 0 with an explicit mask=0 (same convention as the dataset builder).
        THETA_KEYS_EVAL = [
            "m_gross","m_opex","DSO","DIO","DPO",
            "alpha_OCA","alpha_ONCA","alpha_OCL","alpha_ONCL",
            "kappa","delta","payout","neteq_to_sales","phi",
            "r_ST","r_LT","tau",
        ]
        theta_true = {}
        theta_mask = {}
        for k in THETA_KEYS_EVAL:
            v = _get_first_finite(row_nxt, [k])
            vv, mm = _nan_to_zero_with_mask(v)
            # Follow the preprocessing convention: ignore gross-margin + working-capital channels for banks/financials.
            if _is_financial_sector(sector) and (k in ["m_gross","DSO","DIO","DPO"]):
                vv, mm = 0.0, 0
            theta_true[k] = vv
            theta_mask[k] = mm
        rec.update({f"theta_true_{k}": float(v) for k, v in theta_true.items()})
        rec.update({f"mask_theta_{k}": int(m) for k, m in theta_mask.items()})

        # Statement items that often exist in the panel (subset of diag); saved for direct comparison with pred_*.
        def _abs_if_finite(x: float) -> float:
            return float(abs(x)) if np.isfinite(x) else float("nan")

        flow_specs = {
            "COGS": (["COGS"], None),
            "OPEX": (["OPEX"], None),
            "Tax":  (["Tax"], None),
            "NI":   (["NI"], None),
            "Div":  (["Div"], _abs_if_finite),
            "Int":  (["I"], _abs_if_finite),   # panel uses I; model uses Int >= 0
            "TA":   (["TA"], None),
            "TL":   (["TL"], None),
        }

        for name, (cols, fn) in flow_specs.items():
            v = _get_first_finite(row_nxt, cols)
            if fn is not None:
                v = fn(v)
            vv, mm = _nan_to_zero_with_mask(v)
            rec[f"{name}_true"] = float(vv)
            rec[f"mask_{name}"] = int(mm)

        # Net equity issuance / buyback: NetEq = EquityIssues - Buyback (same definition as preprocessing).
        eq = _get_first_finite(row_nxt, ["EquityIssues"])
        bb = _get_first_finite(row_nxt, ["Buyback"])
        mm = int(np.isfinite(eq) or np.isfinite(bb))
        eqv = float(eq) if np.isfinite(eq) else 0.0
        bbv = float(bb) if np.isfinite(bb) else 0.0
        rec["NetEq_true"] = float(eqv - bbv) if mm else 0.0
        rec["mask_NetEq"] = int(mm)

        # predicted diag/state
        rec.update({f"pred_{k}": float(v) for k, v in diag.items()})
        rec.update({f"state_pred_{STATE_COLS[i]}": float(st_pred_next[i]) for i in range(len(STATE_COLS))})

        # true state + mask
        rec.update({f"state_true_{STATE_COLS[i]}": float(st_true_vals[i]) for i in range(len(STATE_COLS))})
        rec.update({f"mask_state_{STATE_COLS[i]}": int(st_true_masks[i]) for i in range(len(STATE_COLS))})

        rows.append(rec)

        # advance: default to predicted state/logS; if next truth is fully available,
        # the next iteration will re-sync automatically.
        st = st_pred_next
        logS_t = float(logS_pred)

    out = pd.DataFrame(rows)
    return out


# ----------------------------
# Mode (2): Future rollout (legacy)
# ----------------------------

def advance_date(date: pd.Timestamp, period_days: float) -> pd.Timestamp:
    if period_days >= 300:
        return date + pd.DateOffset(years=1)
    return date + pd.DateOffset(months=3)

def initial_state_legacy(df: pd.DataFrame) -> Tuple[np.ndarray, float, pd.Timestamp, str]:
    """Legacy: start from the last row, fill missing with zeros."""
    last = df.iloc[-1]

    def g(name, alt: Optional[str] = None):
        if name in last.index:
            v = float(last.get(name, np.nan))
            if np.isfinite(v):
                return v
        if alt is not None and alt in last.index:
            v = float(last.get(alt, np.nan))
            if np.isfinite(v):
                return v
        return 0.0

    C = g("C")
    AR = g("AR")
    Inv = g("Inv")
    OCA = g("OCA", "OCA_implied")
    K = g("K")
    ONCA = g("ONCA", "ONCA_implied")
    AP = g("AP")
    OCL = g("OCL", "OCL_implied")
    STD = g("STD")
    LTD = g("LTD")
    ONCL = g("ONCL", "ONCL_implied")

    TA = _get_first_finite(last, ["TA"])
    TL = _get_first_finite(last, ["TL"])
    E_flow = (TA - TL) if (np.isfinite(TA) and np.isfinite(TL)) else 0.0

    pdays = float(_get_first_finite(last, ["period_days"])) if np.isfinite(_get_first_finite(last, ["period_days"])) else 365.0
    date = pd.to_datetime(last.get("date"))
    sector = str(last.get("sector", ""))
    st = np.array([C, AR, Inv, OCA, K, ONCA, AP, OCL, STD, LTD, ONCL, E_flow], dtype=np.float32)
    return st, pdays, date, sector


# ----------------------------
# Package configuration and entry points
# ----------------------------

@dataclass
class TheoryConfig:
    """Configuration for the theory baseline pipeline."""

    data_dir: str
    out_dir: str
    mode: str = "backtest"
    warmup: int = 3
    min_ar1_points: int = 3
    save_one_file: bool = False
    rollout_steps: int = 8
    disable_interest_for_banks: bool = False
    tickers: Optional[list[str] | str] = None


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the theory baseline."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--mode",
        type=str,
        default="backtest",
        choices=["backtest", "future"],
        help=(
            "backtest: rolling predictions aligned to existing dates (default); "
            "future: roll into unseen future (legacy)."
        ),
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of initial observed periods given before rolling prediction.",
    )
    ap.add_argument(
        "--min_ar1_points",
        type=int,
        default=3,
        help="Minimum number of observed logS points to fit AR(1); otherwise use random-walk.",
    )
    ap.add_argument(
        "--save_one_file",
        action="store_true",
        help="If set, also save a concatenated CSV over all tickers in out_dir/backtest_all.csv.",
    )
    ap.add_argument("--rollout_steps", type=int, default=8)
    ap.add_argument("--disable_interest_for_banks", action="store_true")
    ap.add_argument("--tickers", type=str, default="", help="Optional comma-separated tickers to run.")
    return ap


def _normalize_ticker_selection(tickers: Optional[Sequence[str] | str]) -> Optional[List[str]]:
    """Normalize ticker selection from either a string or a sequence."""
    if tickers is None:
        return None
    if isinstance(tickers, str):
        normalized = [tok.strip() for tok in tickers.split(",") if tok.strip()]
    else:
        normalized = [str(tok).strip() for tok in tickers if str(tok).strip()]
    return normalized or None


def _load_sector_theta_metadata(data_dir: str) -> Tuple[List[str], Dict[str, Any]]:
    """Load sector theta medians with backward-compatible schema handling."""
    with open(os.path.join(data_dir, "sector_theta_medians.json"), "r", encoding="utf-8") as f:
        med = json.load(f)

    if "theta_cols" in med:
        theta_cols = med["theta_cols"]
        sector_medians = med["sector_medians"]
    else:
        sector_medians = med
        if sector_medians:
            first_sector_values = next(iter(sector_medians.values()))
            theta_cols = list(first_sector_values.keys()) if isinstance(first_sector_values, dict) else []
        else:
            theta_cols = []
    return theta_cols, sector_medians


class TheoryBacktestRunner:
    """Package-style runner for the theory baseline.

    Parameters
    ----------
    config:
        Optional :class:`TheoryConfig` instance.
    **kwargs:
        Keyword arguments used to build :class:`TheoryConfig` directly.
    """

    def __init__(self, config: Optional[TheoryConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either 'config' or keyword arguments, not both.")
        self.config = config if config is not None else TheoryConfig(**kwargs)

    @staticmethod
    def load_test_tickers(data_dir: str) -> List[str]:
        """Load the canonical test ticker list from ``meta.json``."""
        with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta: Dict[str, Any] = json.load(f)
        return [str(t) for t in meta.get("test_tickers", [])]

    def selected_tickers(self, meta: Dict[str, Any]) -> List[str]:
        """Return the explicit ticker selection if provided, else all test tickers."""
        explicit = _normalize_ticker_selection(self.config.tickers)
        return explicit if explicit is not None else [str(t) for t in meta.get("test_tickers", [])]

    def run_one(self, *args, **kwargs):
        """Dispatch to :func:`run_backtest_one_ticker`."""
        return run_backtest_one_ticker(*args, **kwargs)

    def run(self) -> None:
        """Run the theory baseline using the stored configuration."""
        cfg = self.config
        os.makedirs(cfg.out_dir, exist_ok=True)

        with open(os.path.join(cfg.data_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta: Dict[str, Any] = json.load(f)

        theta_cols, sector_medians = _load_sector_theta_metadata(cfg.data_dir)
        test_tickers = self.selected_tickers(meta)
        all_frames: List[pd.DataFrame] = []

        for ticker in test_tickers:
            df = load_panel(cfg.data_dir, ticker)
            sector = str(df["sector"].iloc[-1]) if ("sector" in df.columns and len(df) > 0) else ""

            theta_raw = sector_medians.get(sector, {})
            if isinstance(theta_raw, list):
                theta = {theta_cols[i]: float(theta_raw[i]) for i in range(min(len(theta_cols), len(theta_raw)))}
            else:
                theta = {k: float(theta_raw.get(k, 0.0)) for k in theta_cols}

            disable_interest = bool(cfg.disable_interest_for_banks and _is_financial_sector(sector))

            if cfg.mode == "backtest":
                try:
                    out = run_backtest_one_ticker(
                        df=df,
                        ticker=ticker,
                        sector=sector,
                        theta=theta,
                        warmup=int(cfg.warmup),
                        min_ar1_points=int(cfg.min_ar1_points),
                        disable_interest=disable_interest,
                    )
                except Exception as e:
                    print(f"[WARN] {ticker}: backtest skipped ({e})")
                    continue

                out_path = os.path.join(cfg.out_dir, f"{_safe_name(ticker)}_theory_backtest.csv")
                out.to_csv(out_path, index=False)
                print(f"[OK] {ticker}: {out_path}")

                if "mask_logS" in out.columns and out["mask_logS"].sum() > 0:
                    valid = (out["mask_logS"] == 1) & (out["step"] >= 1) & np.isfinite(out["logS_pred"].to_numpy())
                    if bool(valid.any()):
                        mae = float(np.mean(np.abs(out.loc[valid, "logS_pred"] - out.loc[valid, "logS_true"])))
                        print(f"      logS MAE (masked) = {mae:.4f}")

                all_frames.append(out)
            else:
                st, period_days, cur_date, sector_legacy = initial_state_legacy(df)

                S = df["S"].astype(float).values if "S" in df.columns else np.array([], dtype=float)
                logS = np.log(np.maximum(S, 1e-12)) if S.size > 0 else np.array([], dtype=float)
                logS = logS[np.isfinite(logS)]
                if logS.size < 3:
                    print(f"[WARN] {ticker}: insufficient sales history")
                    continue
                c, phi = fit_ar1(logS)
                logS_t = float(logS[-1])

                is_financial = _is_financial_sector(sector_legacy)
                disable_interest2 = bool(cfg.disable_interest_for_banks and is_financial)

                H_chunk = 2
                steps_done = 0
                rows = []
                while steps_done < cfg.rollout_steps:
                    path = ar1_path(logS_t, c, phi, H_chunk)
                    for h in range(H_chunk):
                        if steps_done >= cfg.rollout_steps:
                            break
                        cur_date = advance_date(cur_date, period_days)
                        st_next, diag = simulate_step(
                            st,
                            float(path[h]),
                            theta,
                            period_days,
                            is_financial=is_financial,
                            disable_interest=disable_interest2,
                        )
                        rows.append(dict(
                            date=str(cur_date.date()),
                            step=int(steps_done + 1),
                            logS=float(path[h]),
                            **{f"theta_{k}": float(v) for k, v in theta.items()},
                            **diag,
                            **{f"state_{STATE_COLS[i]}": float(st_next[i]) for i in range(len(STATE_COLS))},
                        ))
                        st = st_next
                        logS_t = float(path[h])
                        steps_done += 1

                out_path = os.path.join(cfg.out_dir, f"{_safe_name(ticker)}_theory_rollout.csv")
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f"[OK] {ticker}: {out_path}")

        if cfg.mode == "backtest" and cfg.save_one_file and all_frames:
            big = pd.concat(all_frames, ignore_index=True)
            out_path = os.path.join(cfg.out_dir, "backtest_all.csv")
            big.to_csv(out_path, index=False)
            print(f"[OK] saved concatenated backtest: {out_path}")

    def run_cli(self) -> None:
        """Compatibility alias retained for existing wrappers."""
        self.run()


def main() -> None:
    """Command-line entry point preserving the original arguments."""
    args = build_arg_parser().parse_args()
    TheoryBacktestRunner(TheoryConfig(**vars(args))).run()


__all__ = [
    "TheoryConfig",
    "TheoryBacktestRunner",
    "run_backtest_one_ticker",
    "simulate_step",
    "load_panel",
    "main",
]


if __name__ == "__main__":
    main()
