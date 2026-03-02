#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_compare_theory_tft.py

Goal
-----
Given rolling backtest outputs from:
  - theory_uncond_baseline_rolling_patched.py  ->  *_theory_backtest.csv
  - tft_uncond_backtest_rolling.py            ->  *_tft_backtest.csv

produce multi-page PDF figures (per ticker) that overlay:
  - Historical observations (warmup region): truth with step <= 0
  - Actual truth in forecast region: truth with step >= 1
  - Theory baseline point forecasts
  - TFT forecasts (quantiles when available; otherwise median point forecasts)

Typical usage
-------------
python plot_compare_theory_tft.py \
  --theory_dir results_theory \
  --tft_dir results_tft \
  --out_dir figures_compare \
  --group all \
  --max_vars_per_page 10 \
  --max_xticks 12

Notes
-----
1) This script is intentionally "read-only" and does not depend on TensorFlow.
2) Missing truth in backtest CSVs follows the convention:
   truth value stored as 0 with a corresponding mask column == 0.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, FuncFormatter
from collections import OrderedDict


# ---------------------------------------------------------------------
# Constants (keep consistent with the rolling scripts)
# ---------------------------------------------------------------------

THETA_KEYS_EVAL: List[str] = [
    "m_gross", "m_opex", "DSO", "DIO", "DPO",
    "alpha_OCA", "alpha_ONCA", "alpha_OCL", "alpha_ONCL",
    "kappa", "delta", "payout", "neteq_to_sales", "phi",
    "r_ST", "r_LT", "tau",
]

# Prefer importing STATE_COLS from the theory baseline so naming stays in-sync.
try:
    from .theory import STATE_COLS as _STATE_COLS
    STATE_COLS: List[str] = list(_STATE_COLS)
except Exception:
    STATE_COLS = ["C", "AR", "Inv", "OCA", "K", "ONCA", "AP", "OCL", "STD", "LTD", "ONCL", "E_flow"]

# Optional: load panel-truth to fill warmup truth for theta/flows when the backtest CSVs did not write them.
# This keeps plotting robust without requiring modifications to rolling scripts.
try:
    from .theory import load_panel as _load_panel, _is_financial_sector as _is_fin_sector
except Exception:
    _load_panel = None  # type: ignore
    _is_fin_sector = None  # type: ignore

FLOW_KEYS_EVAL: List[str] = ["COGS", "OPEX", "Tax", "NI", "Div", "Int", "TA", "TL", "NetEq"]


def _normalize_ticker_selection(tickers: Optional[Sequence[str] | str]) -> Optional[List[str]]:
    """Normalize ticker selection from either a string or a sequence."""
    if tickers is None:
        return None
    if isinstance(tickers, str):
        normalized = [tok.strip() for tok in tickers.split(",") if tok.strip()]
    else:
        normalized = [str(tok).strip() for tok in tickers if str(tok).strip()]
    return normalized or None


def _safe_fs_name(name: str) -> str:
    """Return a filesystem-safe artifact name."""
    name = str(name).strip()
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name) or "UNKNOWN"


# ---------------------------------------------------------------------
# Variable specs
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class VarSpec:
    """A plotting variable with a canonical name and a type that defines column mapping."""
    name: str
    kind: str  # one of {"logS","theta","state","flow"}
    key: Optional[str] = None  # used when kind in {"theta","state","flow"}


def build_vars(group: str) -> List[VarSpec]:
    group = (group or "all").lower()
    out: List[VarSpec] = []

    if group in ("all", "logs_theta", "logs+theta", "logS_theta"):
        out.append(VarSpec(name="logS", kind="logS"))
        for k in THETA_KEYS_EVAL:
            out.append(VarSpec(name=f"theta_{k}", kind="theta", key=k))

    if group in ("all", "state", "states"):
        for s in STATE_COLS:
            out.append(VarSpec(name=f"state_{s}", kind="state", key=s))

    if group in ("all", "flow", "flows"):
        for f in FLOW_KEYS_EVAL:
            out.append(VarSpec(name=f"flow_{f}", kind="flow", key=f))

    # If an unknown group is given, fall back to "all"
    if not out:
        return build_vars("all")
    return out


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "idx" in df.columns:
        df["idx"] = pd.to_numeric(df["idx"], errors="coerce").astype("Int64")
    return df


def _prefix_except(df: pd.DataFrame, prefix: str, keep: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    ren = {c: f"{prefix}{c}" for c in df.columns if c not in keep}
    return df.rename(columns=ren)


def _merge_prefixed_frames(frames: Sequence[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Merge multiple backtest tables on (idx, date) after prefixing non-join columns."""
    join_cols = ["idx", "date"]
    prefixed: List[pd.DataFrame] = []
    for prefix, frame in frames:
        prefixed.append(_prefix_except(frame, prefix, keep=join_cols))

    if not prefixed:
        return pd.DataFrame(columns=join_cols)

    df = prefixed[0]
    for nxt in prefixed[1:]:
        df = pd.merge(df, nxt, on=join_cols, how="outer", sort=True)

    def pick(*cols: str) -> pd.Series:
        series: Optional[pd.Series] = None
        for col in cols:
            cur = df.get(col)
            if cur is None:
                continue
            series = cur if series is None else series.where(series.notna(), cur)
        if series is None:
            return pd.Series([np.nan] * len(df), index=df.index)
        return series

    df["ticker"] = pick("tft_ticker", "llm_ticker", "th_ticker")
    df["sector"] = pick("tft_sector", "llm_sector", "th_sector")
    df["step"] = pd.to_numeric(pick("tft_step", "llm_step", "th_step"), errors="coerce")

    if "idx" in df.columns:
        df = df.sort_values(["idx", "date"], kind="mergesort")
    else:
        df = df.sort_values(["date"], kind="mergesort")
    return df.reset_index(drop=True)


def load_pair(theory_csv: str, tft_csv: str) -> pd.DataFrame:
    """
    Merge theory and TFT backtest CSVs on (idx, date). All non-join columns get prefixed:
        th_* , tft_*
    """
    df_th = _read_csv(theory_csv)
    df_tft = _read_csv(tft_csv)
    return _merge_prefixed_frames([
        ("th_", df_th),
        ("tft_", df_tft),
    ])


def load_triple(theory_csv: str, tft_csv: str, llm_csv: str) -> pd.DataFrame:
    """
    Merge theory, TFT, and LLM backtest CSVs on (idx, date). All non-join columns get prefixed:
        th_* , tft_* , llm_*
    """
    df_th = _read_csv(theory_csv)
    df_tft = _read_csv(tft_csv)
    df_llm = _read_csv(llm_csv)
    return _merge_prefixed_frames([
        ("th_", df_th),
        ("tft_", df_tft),
        ("llm_", df_llm),
    ])

# ---------------------------------------------------------------------
# Optional: fill warmup truth from panels
# ---------------------------------------------------------------------

def fill_warmup_truth_from_panels(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """
    If theta_true_*/mask_theta_* and flow *_true/mask_* are missing in warmup rows (step <= 0),
    fill them from the original cached panel (data_dir/panels/<ticker>.csv) using the same
    masking convention (NaN -> value=0, mask=0).

    This is purely a plotting convenience: it does NOT change any model predictions.
    """
    if not data_dir:
        return df
    if _load_panel is None:
        return df
    if "step" not in df.columns or "idx" not in df.columns:
        return df

    m_warm = df["step"].notna() & (pd.to_numeric(df["step"], errors="coerce") <= 0)
    if not bool(m_warm.any()):
        return df

    ticker = str(df.get("ticker", pd.Series([""])).dropna().iloc[0]) if len(df) else ""
    sector = str(df.get("sector", pd.Series([""])).dropna().iloc[0]) if len(df) else ""

    try:
        panel = _load_panel(data_dir, ticker)  # type: ignore[misc]
    except Exception:
        return df

    if panel is None or len(panel) == 0:
        return df

    # Align panel by idx (positional index after date sort in load_panel()).
    panel = panel.sort_values("date").reset_index(drop=True).copy()
    panel.index.name = "idx"

    # Warmup indices in df, in the same order as df rows:
    idxs = pd.to_numeric(df.loc[m_warm, "idx"], errors="coerce").astype("Int64")
    # Reindex panel to df warmup idxs; rows out of range become NaN.
    idxs_np = idxs.fillna(-1).astype(int).to_numpy()
    panel_w = panel.reindex(idxs_np)

    is_fin = False
    if _is_fin_sector is not None:
        try:
            is_fin = bool(_is_fin_sector(sector))  # type: ignore[misc]
        except Exception:
            is_fin = False

    # Helper: convert a numeric array into (val, mask) with the repo convention.
    def _val_mask(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = arr.astype(float)
        mask = np.isfinite(arr).astype(int)
        val = np.where(mask == 1, arr, 0.0)
        return val, mask

    # --- theta truth ---
    for k in THETA_KEYS_EVAL:
        truth_col = f"tft_theta_true_{k}"
        mask_col = f"tft_mask_theta_{k}"
        if truth_col not in df.columns:
            df[truth_col] = np.nan
        if mask_col not in df.columns:
            df[mask_col] = np.nan

        need = m_warm & (pd.to_numeric(df[mask_col], errors="coerce").fillna(0.0) != 1.0)
        if not bool(need.any()):
            continue

        if is_fin and k in ["m_gross", "DSO", "DIO", "DPO"]:
            val = np.zeros(int(m_warm.sum()), dtype=float)
            mask = np.zeros(int(m_warm.sum()), dtype=int)
        else:
            arr = pd.to_numeric(panel_w.get(k, pd.Series([np.nan] * int(m_warm.sum()))), errors="coerce").to_numpy()
            val, mask = _val_mask(arr)

        # fill only warmup subset
        warm_index = df.index[m_warm]
        df.loc[warm_index, truth_col] = np.where(
            pd.to_numeric(df.loc[warm_index, mask_col], errors="coerce").fillna(0.0) == 1.0,
            df.loc[warm_index, truth_col],
            val,
        )
        df.loc[warm_index, mask_col] = np.where(
            pd.to_numeric(df.loc[warm_index, mask_col], errors="coerce").fillna(0.0) == 1.0,
            df.loc[warm_index, mask_col],
            mask,
        )

    # --- flow truth ---
    def _abs_if_finite_arr(a: np.ndarray) -> np.ndarray:
        a = a.astype(float)
        return np.where(np.isfinite(a), np.abs(a), a)

    flow_specs = {
        "COGS": (["COGS"], None),
        "OPEX": (["OPEX"], None),
        "Tax":  (["Tax"], None),
        "NI":   (["NI"], None),
        "Div":  (["Div"], _abs_if_finite_arr),
        "Int":  (["I"], _abs_if_finite_arr),  # panel uses I; output uses Int >= 0
        "TA":   (["TA"], None),
        "TL":   (["TL"], None),
    }

    for name, (cols, fn) in flow_specs.items():
        truth_col = f"tft_{name}_true"
        mask_col = f"tft_mask_{name}"
        if truth_col not in df.columns:
            df[truth_col] = np.nan
        if mask_col not in df.columns:
            df[mask_col] = np.nan

        # Only fill rows where warmup mask is missing/0
        if not bool((m_warm & (pd.to_numeric(df[mask_col], errors="coerce").fillna(0.0) != 1.0)).any()):
            continue

        # Use the first listed panel column; keep NaN if column not present.
        base = cols[0]
        arr = pd.to_numeric(panel_w.get(base, pd.Series([np.nan] * int(m_warm.sum()))), errors="coerce").to_numpy()
        if fn is not None:
            arr = fn(arr)

        val, mask = _val_mask(arr)

        warm_index = df.index[m_warm]
        df.loc[warm_index, truth_col] = np.where(
            pd.to_numeric(df.loc[warm_index, mask_col], errors="coerce").fillna(0.0) == 1.0,
            df.loc[warm_index, truth_col],
            val,
        )
        df.loc[warm_index, mask_col] = np.where(
            pd.to_numeric(df.loc[warm_index, mask_col], errors="coerce").fillna(0.0) == 1.0,
            df.loc[warm_index, mask_col],
            mask,
        )

    # NetEq truth: EquityIssues - Buyback (same as preprocessing / backtests)
    name = "NetEq"
    truth_col = f"tft_{name}_true"
    mask_col = f"tft_mask_{name}"
    if truth_col not in df.columns:
        df[truth_col] = np.nan
    if mask_col not in df.columns:
        df[mask_col] = np.nan

    if bool((m_warm & (pd.to_numeric(df[mask_col], errors="coerce").fillna(0.0) != 1.0)).any()):
        eq = pd.to_numeric(panel_w.get("EquityIssues", pd.Series([np.nan] * int(m_warm.sum()))), errors="coerce").to_numpy()
        bb = pd.to_numeric(panel_w.get("Buyback", pd.Series([np.nan] * int(m_warm.sum()))), errors="coerce").to_numpy()
        m_eq = np.isfinite(eq)
        m_bb = np.isfinite(bb)
        mask = (m_eq | m_bb).astype(int)
        eqv = np.where(m_eq, eq.astype(float), 0.0)
        bbv = np.where(m_bb, bb.astype(float), 0.0)
        val = np.where(mask == 1, eqv - bbv, 0.0)

        warm_index = df.index[m_warm]
        df.loc[warm_index, truth_col] = np.where(
            pd.to_numeric(df.loc[warm_index, mask_col], errors="coerce").fillna(0.0) == 1.0,
            df.loc[warm_index, truth_col],
            val,
        )
        df.loc[warm_index, mask_col] = np.where(
            pd.to_numeric(df.loc[warm_index, mask_col], errors="coerce").fillna(0.0) == 1.0,
            df.loc[warm_index, mask_col],
            mask,
        )

    return df


def discover_pairs(theory_dir: str, tft_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching (*_theory_backtest.csv, *_tft_backtest.csv) by shared basename prefix.
    Example:
      AAPL_theory_backtest.csv  <->  AAPL_tft_backtest.csv
    """
    theory_dir = os.path.abspath(theory_dir)
    tft_dir = os.path.abspath(tft_dir)

    th_files = [f for f in os.listdir(theory_dir) if f.endswith("_theory_backtest.csv")]
    tft_files = [f for f in os.listdir(tft_dir) if f.endswith("_tft_backtest.csv")]

    th_map = {f.replace("_theory_backtest.csv", ""): os.path.join(theory_dir, f) for f in th_files}
    tft_map = {f.replace("_tft_backtest.csv", ""): os.path.join(tft_dir, f) for f in tft_files}

    keys = sorted(set(th_map.keys()).intersection(set(tft_map.keys())))
    return [(th_map[k], tft_map[k]) for k in keys]


def discover_triples(theory_dir: str, tft_dir: str, llm_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find matching (*_theory_backtest.csv, *_tft_backtest.csv, *_llm_backtest.csv)
    by shared basename prefix.
    Example:
      AAPL_theory_backtest.csv <-> AAPL_tft_backtest.csv <-> AAPL_llm_backtest.csv
    """
    theory_dir = os.path.abspath(theory_dir)
    tft_dir = os.path.abspath(tft_dir)
    llm_dir = os.path.abspath(llm_dir)

    th_files = [f for f in os.listdir(theory_dir) if f.endswith("_theory_backtest.csv")]
    tft_files = [f for f in os.listdir(tft_dir) if f.endswith("_tft_backtest.csv")]
    llm_files = [f for f in os.listdir(llm_dir) if f.endswith("_llm_backtest.csv")]

    th_map = {f.replace("_theory_backtest.csv", ""): os.path.join(theory_dir, f) for f in th_files}
    tft_map = {f.replace("_tft_backtest.csv", ""): os.path.join(tft_dir, f) for f in tft_files}
    llm_map = {f.replace("_llm_backtest.csv", ""): os.path.join(llm_dir, f) for f in llm_files}

    keys = sorted(set(th_map.keys()).intersection(set(tft_map.keys())).intersection(set(llm_map.keys())))
    return [(th_map[k], tft_map[k], llm_map[k]) for k in keys]


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def yq_formatter(x, pos=None) -> str:
    dt = mdates.num2date(x)
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}\nQ{q}"


def _pick_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _truth_cols_for(df: pd.DataFrame, v: VarSpec) -> Tuple[Optional[str], Optional[str]]:
    """Return (truth_col, mask_col) names; prefer TFT columns, fall back to theory columns."""
    if v.kind == "logS":
        truth = _pick_existing(df, ["tft_logS_true", "th_logS_true"])
        mask = _pick_existing(df, ["tft_mask_logS", "th_mask_logS"])
        return truth, mask
    if v.kind == "theta" and v.key:
        truth = _pick_existing(df, [f"tft_theta_true_{v.key}", f"th_theta_true_{v.key}"])
        mask = _pick_existing(df, [f"tft_mask_theta_{v.key}", f"th_mask_theta_{v.key}"])
        return truth, mask
    if v.kind == "state" and v.key:
        truth = _pick_existing(df, [f"tft_state_true_{v.key}", f"th_state_true_{v.key}"])
        mask = _pick_existing(df, [f"tft_mask_state_{v.key}", f"th_mask_state_{v.key}"])
        return truth, mask
    if v.kind == "flow" and v.key:
        truth = _pick_existing(df, [f"tft_{v.key}_true", f"th_{v.key}_true"])
        mask = _pick_existing(df, [f"tft_mask_{v.key}", f"th_mask_{v.key}"])
        return truth, mask
    return (None, None)


def _pred_cols_for(v: VarSpec) -> Dict[str, Optional[str]]:
    """
    Return a dict describing prediction columns.
    Keys may include:
      - th: theory point forecast
      - tft_q10/tft_q50/tft_q90: TFT quantiles
      - tft: TFT point/median fallback
      - llm_q10/llm_q50/llm_q90: LLM quantiles when available
      - llm: LLM point/median fallback
    """
    out: Dict[str, Optional[str]] = {
        "th": None,
        "tft_q10": None,
        "tft_q50": None,
        "tft_q90": None,
        "tft": None,
        "llm_q10": None,
        "llm_q50": None,
        "llm_q90": None,
        "llm": None,
    }

    if v.kind == "logS":
        out["th"] = "th_logS_pred"
        out["tft_q10"] = "tft_logS_pred_q10"
        out["tft_q50"] = "tft_logS_pred_q50"
        out["tft_q90"] = "tft_logS_pred_q90"
        out["tft"] = "tft_logS_pred"
        out["llm_q10"] = "llm_logS_pred_q10"
        out["llm_q50"] = "llm_logS_pred_q50"
        out["llm_q90"] = "llm_logS_pred_q90"
        out["llm"] = "llm_logS_pred"
        return out

    if v.kind == "theta" and v.key:
        out["th"] = f"th_theta_{v.key}"
        out["tft_q10"] = f"tft_theta_{v.key}_q10"
        out["tft_q50"] = f"tft_theta_{v.key}_q50"
        out["tft_q90"] = f"tft_theta_{v.key}_q90"
        out["tft"] = f"tft_theta_{v.key}"
        out["llm_q10"] = f"llm_theta_{v.key}_q10"
        out["llm_q50"] = f"llm_theta_{v.key}_q50"
        out["llm_q90"] = f"llm_theta_{v.key}_q90"
        out["llm"] = f"llm_theta_{v.key}"
        return out

    if v.kind == "state" and v.key:
        out["th"] = f"th_state_pred_{v.key}"
        out["tft_q10"] = f"tft_state_pred_{v.key}_q10"
        out["tft_q50"] = f"tft_state_pred_{v.key}_q50"
        out["tft_q90"] = f"tft_state_pred_{v.key}_q90"
        out["tft"] = f"tft_state_pred_{v.key}"
        out["llm_q10"] = f"llm_state_pred_{v.key}_q10"
        out["llm_q50"] = f"llm_state_pred_{v.key}_q50"
        out["llm_q90"] = f"llm_state_pred_{v.key}_q90"
        out["llm"] = f"llm_state_pred_{v.key}"
        return out

    if v.kind == "flow" and v.key:
        out["th"] = f"th_pred_{v.key}"
        out["tft_q10"] = f"tft_pred_{v.key}_q10"
        out["tft_q50"] = f"tft_pred_{v.key}_q50"
        out["tft_q90"] = f"tft_pred_{v.key}_q90"
        out["tft"] = f"tft_pred_{v.key}"
        out["llm_q10"] = f"llm_pred_{v.key}_q10"
        out["llm_q50"] = f"llm_pred_{v.key}_q50"
        out["llm_q90"] = f"llm_pred_{v.key}_q90"
        out["llm"] = f"llm_pred_{v.key}"
        return out

    return out


def _compute_ticks(dates: pd.Series, max_xticks: int) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    d = pd.to_datetime(dates.dropna().unique())
    d = pd.to_datetime(sorted(d))
    if len(d) == 0:
        return np.asarray([]), []
    if len(d) > max_xticks:
        step = int(np.ceil(len(d) / max_xticks))
        tick_dates = d[::step]
    else:
        tick_dates = d
    tick_locs = mdates.date2num(tick_dates)
    return tick_locs, list(tick_dates)


def _finite_np(x: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float, copy=False)
    return arr[np.isfinite(arr)]


def plot_one_page(
    df: pd.DataFrame,
    vars_page: List[VarSpec],
    tick_locs: np.ndarray,
    figsize: Tuple[float, float],
    title: str,
    mode: str = "double",
    plot_llm_q10_q90: bool = False,
    show_legend: bool = True,
    show_suptitle: bool = True,
    show_var_titles: bool = True,
    tight_layout_rect: Optional[Sequence[float]] = None,
    tight_layout_pad: float = 0.8,
) -> plt.Figure:
    n = len(vars_page)
    if n <= 1:
        nrows, ncols = 1, 1
    elif n >= 9:
        nrows, ncols = 5, 2
    else:
        nrows, ncols = int(np.ceil(n / 2)), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    legend_cache = OrderedDict()
    mode_norm = str(mode or "double").lower()
    use_llm = mode_norm == "triple"

    for i, v in enumerate(vars_page):
        ax = axes_flat[i]

        truth_col, mask_col = _truth_cols_for(df, v)
        pred_cols = _pred_cols_for(v)

        step = pd.to_numeric(df.get("step"), errors="coerce")
        date = df.get("date")

        if truth_col and truth_col in df.columns and date is not None:
            y_true = pd.to_numeric(df[truth_col], errors="coerce")
            if mask_col and mask_col in df.columns:
                m_true = pd.to_numeric(df[mask_col], errors="coerce").fillna(0).astype(int) == 1
            else:
                m_true = y_true.notna()

            m_hist = (step.notna()) & (step <= 0) & m_true & y_true.notna()
            if bool(m_hist.any()):
                h = ax.plot(
                    date.loc[m_hist], y_true.loc[m_hist],
                    marker="o", markersize=10, linewidth=1.0,
                    color="gray", alpha=0.5, label="Historical Obs (Truth)"
                )[0]
                legend_cache.setdefault("Historical Obs (Truth)", h)

            m_future_truth = (step.notna()) & (step >= 1) & m_true & y_true.notna()
            if bool(m_future_truth.any()):
                h = ax.scatter(
                    date.loc[m_future_truth], y_true.loc[m_future_truth],
                    marker="x", s=120, color="black", linewidths=2.0,
                    zorder=5, label="Actual Truth"
                )
                legend_cache.setdefault("Actual Truth", h)

        th_col = pred_cols.get("th")
        if th_col and th_col in df.columns and date is not None:
            y_th = pd.to_numeric(df[th_col], errors="coerce")
            m_th = (step.notna()) & (step >= 1) & y_th.notna()
            if bool(m_th.any()):
                h = ax.scatter(
                    date.loc[m_th], y_th.loc[m_th],
                    marker="D", s=90, color="orange", edgecolor="black",
                    alpha=0.85, label="Theory Pred (Point)"
                )
                legend_cache.setdefault("Theory Pred (Point)", h)

        q10 = pred_cols.get("tft_q10")
        q50 = pred_cols.get("tft_q50")
        q90 = pred_cols.get("tft_q90")
        tft_plotted = False
        if q10 in df.columns and q50 in df.columns and q90 in df.columns and date is not None:
            y10 = pd.to_numeric(df[q10], errors="coerce")
            y50 = pd.to_numeric(df[q50], errors="coerce")
            y90 = pd.to_numeric(df[q90], errors="coerce")
            m_q = (step.notna()) & (step >= 1) & y50.notna()
            if bool(m_q.any()):
                xq = date.loc[m_q]
                h = ax.scatter(xq, y10.loc[m_q], marker="v", color="red", s=80, alpha=0.70, label="TFT Pred P10")
                legend_cache.setdefault("TFT Pred P10", h)
                h = ax.scatter(xq, y50.loc[m_q], marker="s", color="blue", s=100, edgecolor="black", label="TFT Pred P50")
                legend_cache.setdefault("TFT Pred P50", h)
                h = ax.scatter(xq, y90.loc[m_q], marker="^", color="green", s=80, alpha=0.70, label="TFT Pred P90")
                legend_cache.setdefault("TFT Pred P90", h)
                tft_plotted = True
        if not tft_plotted:
            tft_col = pred_cols.get("tft")
            if tft_col and tft_col in df.columns and date is not None:
                y_tft = pd.to_numeric(df[tft_col], errors="coerce")
                m_tft = (step.notna()) & (step >= 1) & y_tft.notna()
                if bool(m_tft.any()):
                    h = ax.scatter(
                        date.loc[m_tft], y_tft.loc[m_tft],
                        marker="s", s=45, color="purple", edgecolor="black",
                        alpha=0.80, label="TFT Pred (Point)"
                    )
                    legend_cache.setdefault("TFT Pred (Point)", h)

        llm_q10 = pred_cols.get("llm_q10")
        llm_q50 = pred_cols.get("llm_q50")
        llm_q90 = pred_cols.get("llm_q90")
        llm_point = pred_cols.get("llm")
        llm_plotted = False
        if use_llm and date is not None:
            llm_q50_candidate = llm_q50 if llm_q50 in df.columns else llm_point
            if llm_q50_candidate and llm_q50_candidate in df.columns:
                y_llm_q50 = pd.to_numeric(df[llm_q50_candidate], errors="coerce")
                m_llm = (step.notna()) & (step >= 1) & y_llm_q50.notna()
                if bool(m_llm.any()):
                    x_llm = date.loc[m_llm]
                    h = ax.scatter(
                        x_llm, y_llm_q50.loc[m_llm],
                        marker="o", s=120, facecolors="none", edgecolors="black",
                        linewidths=1.6, alpha=0.95, label="LLM Pred P50"
                    )
                    legend_cache.setdefault("LLM Pred P50", h)
                    llm_plotted = True

                    if bool(plot_llm_q10_q90):
                        if llm_q10 and llm_q10 in df.columns:
                            y_llm_q10 = pd.to_numeric(df[llm_q10], errors="coerce")
                            m_llm_q10 = (step.notna()) & (step >= 1) & y_llm_q10.notna()
                            if bool(m_llm_q10.any()):
                                h = ax.scatter(
                                    date.loc[m_llm_q10], y_llm_q10.loc[m_llm_q10],
                                    marker="1", s=70, color="black", alpha=0.75, label="LLM Pred P10"
                                )
                                legend_cache.setdefault("LLM Pred P10", h)
                        if llm_q90 and llm_q90 in df.columns:
                            y_llm_q90 = pd.to_numeric(df[llm_q90], errors="coerce")
                            m_llm_q90 = (step.notna()) & (step >= 1) & y_llm_q90.notna()
                            if bool(m_llm_q90.any()):
                                h = ax.scatter(
                                    date.loc[m_llm_q90], y_llm_q90.loc[m_llm_q90],
                                    marker="2", s=70, color="black", alpha=0.75, label="LLM Pred P90"
                                )
                                legend_cache.setdefault("LLM Pred P90", h)

        if th_col and th_col in df.columns and date is not None:
            y_th_all = pd.to_numeric(df[th_col], errors="coerce")
            m_forecast = (step.notna()) & (step >= 1) & y_th_all.notna()

            ref_list: List[pd.Series] = []
            if truth_col and truth_col in df.columns:
                y_ref = pd.to_numeric(df[truth_col], errors="coerce")
                if mask_col and mask_col in df.columns:
                    m_ref = pd.to_numeric(df[mask_col], errors="coerce").fillna(0).astype(int) == 1
                else:
                    m_ref = y_ref.notna()
                ref_list.append(y_ref.loc[m_ref & y_ref.notna()])

            for pred_c in [q10, q50, q90, pred_cols.get("tft")]:
                if pred_c and pred_c in df.columns:
                    y_pred = pd.to_numeric(df[pred_c], errors="coerce")
                    ref_list.append(y_pred.loc[(step.notna()) & (step >= 1) & y_pred.notna()])

            if use_llm:
                for pred_c in [llm_q50 if llm_q50 in df.columns else llm_point]:
                    if pred_c and pred_c in df.columns:
                        y_pred = pd.to_numeric(df[pred_c], errors="coerce")
                        ref_list.append(y_pred.loc[(step.notna()) & (step >= 1) & y_pred.notna()])
                if bool(plot_llm_q10_q90):
                    for pred_c in [llm_q10, llm_q90]:
                        if pred_c and pred_c in df.columns:
                            y_pred = pd.to_numeric(df[pred_c], errors="coerce")
                            ref_list.append(y_pred.loc[(step.notna()) & (step >= 1) & y_pred.notna()])

            if ref_list:
                ref_vals = np.concatenate([_finite_np(s) for s in ref_list if s is not None], axis=0)
            else:
                ref_vals = np.asarray([], dtype=float)

            if ref_vals.size > 0:
                y_min, y_max = float(np.min(ref_vals)), float(np.max(ref_vals))
                if y_max == y_min:
                    pad = max(abs(y_min) * 0.1, 1.0)
                else:
                    pad = 0.05 * (y_max - y_min)

                ymin_padded = y_min - pad
                ymax_padded = y_max + pad
                ax.set_ylim(ymin_padded, ymax_padded)

                y_th_forecast = y_th_all.loc[m_forecast]
                out_s = (y_th_forecast > ymax_padded) | (y_th_forecast < ymin_padded)
                out_s_full = pd.Series(False, index=df.index)
                out_s_full.loc[m_forecast] = out_s

                if bool(out_s_full.any()):
                    x_out = date.loc[out_s_full]
                    y_out = pd.to_numeric(y_th_all.loc[out_s_full], errors="coerce").to_numpy(dtype=float, copy=False)
                    y_clip = np.where(y_out > ymax_padded, ymax_padded, np.where(y_out < ymin_padded, ymin_padded, y_out))
                    h = ax.scatter(
                        x_out, y_clip,
                        marker="*", s=90, color="orange", edgecolor="black",
                        zorder=6, label="Theory Outlier (clipped)"
                    )
                    legend_cache.setdefault("Theory Outlier (clipped)", h)
                    ax.text(0.02, 0.98, "★ theory outlier clipped", transform=ax.transAxes, va="top", fontsize=7)

        if show_var_titles:
            ax.set_title(v.name, fontsize=22, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        if len(tick_locs) > 0:
            ax.xaxis.set_major_locator(FixedLocator(tick_locs))
            ax.xaxis.set_major_formatter(FuncFormatter(yq_formatter))
            # ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="both", which="major", labelsize=17)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    if show_legend and legend_cache:
        fig.legend(
            list(legend_cache.values()), list(legend_cache.keys()),
            loc="upper center", bbox_to_anchor=(0.5, 0.995),
            ncol=5, frameon=True, fontsize=9
        )

    if show_suptitle and title:
        fig.suptitle(title, fontsize=22, fontweight="bold", y=0.999)

    if tight_layout_rect is None:
        tight_layout_rect = [0.03, 0.03, 0.97, 0.965] if (show_legend or show_suptitle) else [0.01, 0.01, 0.99, 0.99]
    fig.tight_layout(rect=tight_layout_rect, pad=tight_layout_pad)
    return fig


def plot_ticker_to_pdf(
    df: pd.DataFrame,
    vars_all: List[VarSpec],
    out_pdf: str,
    max_vars_per_page: int = 10,
    figsize: Tuple[float, float] = (12, 18),
    max_xticks: int = 12,
    mode: str = "double",
    plot_llm_q10_q90: bool = False,
) -> None:
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)

    tick_locs, _ = _compute_ticks(df.get("date", pd.Series(dtype="datetime64[ns]")), max_xticks=max_xticks)
    pages = [vars_all[i:i + max_vars_per_page] for i in range(0, len(vars_all), max_vars_per_page)]
    ticker = str(df.get("ticker", pd.Series([""])).dropna().iloc[0]) if len(df) else ""
    sector = str(df.get("sector", pd.Series([""])).dropna().iloc[0]) if len(df) else ""
    mode_label = "Theory vs TFT vs LLM" if str(mode).lower() == "triple" else "Theory vs TFT"

    with PdfPages(out_pdf) as pdf:
        for pidx, vars_page in enumerate(pages, start=1):
            title = f"{ticker} | {sector} | {mode_label} | Page {pidx}/{len(pages)}"
            fig = plot_one_page(
                df=df,
                vars_page=vars_page,
                tick_locs=tick_locs,
                figsize=figsize,
                title=title,
                mode=mode,
                plot_llm_q10_q90=plot_llm_q10_q90,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def save_individual_variable_pngs(
    df: pd.DataFrame,
    vars_all: List[VarSpec],
    out_dir: str,
    figsize: Tuple[float, float] = (12, 4.8),
    max_xticks: int = 12,
    dpi: int = 300,
    mode: str = "double",
    plot_llm_q10_q90: bool = False,
) -> List[str]:
    """Save each variable panel as a standalone SVG without legend or large outer title."""
    _ = dpi  # kept for backward-compatible function signature
    os.makedirs(out_dir, exist_ok=True)
    tick_locs, _ = _compute_ticks(df.get("date", pd.Series(dtype="datetime64[ns]")), max_xticks=max_xticks)

    saved: List[str] = []
    for vidx, var in enumerate(vars_all, start=1):
        fig = plot_one_page(
            df=df,
            vars_page=[var],
            tick_locs=tick_locs,
            figsize=figsize,
            title="",
            mode=mode,
            plot_llm_q10_q90=plot_llm_q10_q90,
            show_legend=False,
            show_suptitle=False,
            show_var_titles=True,
            tight_layout_rect=[0.01, 0.01, 0.99, 0.99],
            tight_layout_pad=0.25,
        )
        filename = f"{vidx:02d}_{_safe_fs_name(var.name)}.svg"
        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        saved.append(out_path)
    return saved


def plot_ticker_outputs(
    df: pd.DataFrame,
    vars_all: List[VarSpec],
    out_pdf: str,
    out_png_dir: str,
    max_vars_per_page: int = 10,
    figsize: Tuple[float, float] = (12, 18),
    max_xticks: int = 12,
    png_dpi: int = 300,
    mode: str = "double",
    plot_llm_q10_q90: bool = False,
) -> List[str]:
    """Save the paginated PDF and standalone per-variable SVGs for one ticker."""
    plot_ticker_to_pdf(
        df=df,
        vars_all=vars_all,
        out_pdf=out_pdf,
        max_vars_per_page=max_vars_per_page,
        figsize=figsize,
        max_xticks=max_xticks,
        mode=mode,
        plot_llm_q10_q90=plot_llm_q10_q90,
    )
    return save_individual_variable_pngs(
        df=df,
        vars_all=vars_all,
        out_dir=out_png_dir,
        figsize=(10, 7.33),
        max_xticks=max_xticks,
        dpi=png_dpi,
        mode=mode,
        plot_llm_q10_q90=plot_llm_q10_q90,
    )


# ---------------------------------------------------------------------
# Package configuration and entry points
# ---------------------------------------------------------------------

@dataclass
class PlotConfig:
    """Configuration for theory/TFT/(optional) LLM comparison plotting."""

    theory_dir: str = "results_theory"
    tft_dir: str = "results_tft"
    llm_dir: str = ""
    out_dir: str = "figures_compare"
    data_dir: str = ""
    tickers: Optional[list[str] | str] = None
    group: str = "all"
    max_vars_per_page: int = 10
    max_xticks: int = 12
    fig_w: float = 12.0
    fig_h: float = 18.0
    png_dpi: int = 300
    plot_llm_q10_q90: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for comparison plotting."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory_dir", type=str, default="results_theory")
    ap.add_argument("--tft_dir", type=str, default="results_tft")
    ap.add_argument("--llm_dir", type=str, default="", help="Optional: directory containing *_llm_backtest.csv files.")
    ap.add_argument("--out_dir", type=str, default="figures_compare")
    ap.add_argument(
        "--data_dir",
        type=str,
        default="",
        help=(
            "Optional: data_uncond directory containing panels/; used to fill warmup truth "
            "for theta/flows if missing in backtest CSVs."
        ),
    )
    ap.add_argument("--tickers", type=str, default="", help="Optional: comma-separated tickers to plot.")
    ap.add_argument("--group", type=str, default="all", help="all | logs_theta | state | flows")
    ap.add_argument("--mode", type=str, default="double", help="double | triple")
    ap.add_argument("--max_vars_per_page", type=int, default=10)
    ap.add_argument("--max_xticks", type=int, default=12)
    ap.add_argument("--fig_w", type=float, default=12.0)
    ap.add_argument("--fig_h", type=float, default=18.0)
    ap.add_argument("--png_dpi", type=int, default=300, help="Legacy option retained for compatibility; SVG exports ignore DPI.")
    ap.add_argument("--plot_llm_q10_q90", action="store_true", help="In triple mode, also show LLM q10/q90 when available.")
    return ap


class ComparisonPlotter:
    """Package-style comparison plotter.

    Parameters
    ----------
    config:
        Optional :class:`PlotConfig` instance.
    **kwargs:
        Keyword arguments used to build :class:`PlotConfig` directly.
    """

    def __init__(self, config: Optional[PlotConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either 'config' or keyword arguments, not both.")
        self.config = config if config is not None else PlotConfig(**kwargs)

    def run(self, mode: str = "double") -> None:
        """Generate comparison PDFs using the stored configuration."""
        cfg = self.config
        mode_norm = str(mode or "double").lower()
        if mode_norm not in {"double", "triple"}:
            raise ValueError(f"Unsupported plotting mode: {mode!r}. Use 'double' or 'triple'.")

        if mode_norm == "triple":
            if not cfg.llm_dir:
                raise ValueError("llm_dir must be provided when mode='triple'.")
            triples = discover_triples(cfg.theory_dir, cfg.tft_dir, cfg.llm_dir)
            if not triples:
                raise FileNotFoundError(
                    "No matching backtest CSV triples found under:\n"
                    f"  theory_dir={cfg.theory_dir}\n"
                    f"  tft_dir={cfg.tft_dir}\n"
                    f"  llm_dir={cfg.llm_dir}\n"
                    "Expected files named *_theory_backtest.csv, *_tft_backtest.csv, and *_llm_backtest.csv."
                )
            iterable = triples
        else:
            pairs = discover_pairs(cfg.theory_dir, cfg.tft_dir)
            if not pairs:
                raise FileNotFoundError(
                    "No matching backtest CSV pairs found under:\n"
                    f"  theory_dir={cfg.theory_dir}\n"
                    f"  tft_dir={cfg.tft_dir}\n"
                    "Expected files named *_theory_backtest.csv and *_tft_backtest.csv."
                )
            iterable = pairs

        vars_all = build_vars(cfg.group)
        ticker_filter = _normalize_ticker_selection(cfg.tickers)
        os.makedirs(cfg.out_dir, exist_ok=True)

        for item in iterable:
            if mode_norm == "triple":
                th_csv, tft_csv, llm_csv = item
                df = load_triple(th_csv, tft_csv, llm_csv)
            else:
                th_csv, tft_csv = item
                df = load_pair(th_csv, tft_csv)

            if cfg.data_dir:
                df = fill_warmup_truth_from_panels(df, cfg.data_dir)
            if ticker_filter is not None:
                ticker_value = str(df.get("ticker", pd.Series([""])).dropna().iloc[0]) if len(df) else ""
                if ticker_value not in ticker_filter:
                    continue

            ticker = str(df.get("ticker", pd.Series([""])).dropna().iloc[0]) if len(df) else "UNKNOWN"
            safe_ticker = _safe_fs_name(ticker)
            suffix = f"compare_{cfg.group}" if mode_norm == "double" else f"compare_triple_{cfg.group}"
            out_pdf = os.path.join(cfg.out_dir, f"{safe_ticker}_{suffix}.pdf")
            png_dir = os.path.join(cfg.out_dir, safe_ticker)
            saved_pngs = plot_ticker_outputs(
                df=df,
                vars_all=vars_all,
                out_pdf=out_pdf,
                out_png_dir=png_dir,
                max_vars_per_page=int(cfg.max_vars_per_page),
                figsize=(float(cfg.fig_w), float(cfg.fig_h)),
                max_xticks=int(cfg.max_xticks),
                png_dpi=int(cfg.png_dpi),
                mode=mode_norm,
                plot_llm_q10_q90=bool(cfg.plot_llm_q10_q90),
            )
            print(f"[OK] saved PDF: {out_pdf}")
            print(f"[OK] saved {len(saved_pngs)} PNG panels under: {png_dir}")

    def run_cli(self) -> None:
        """Compatibility alias retained for existing wrappers."""
        self.run()


def main() -> None:
    """Command-line entry point preserving the original arguments."""
    args = build_arg_parser().parse_args()
    mode = str(getattr(args, "mode", "double"))
    plot_kwargs = {k: v for k, v in vars(args).items() if k != "mode"}
    plotter = ComparisonPlotter(PlotConfig(**plot_kwargs))
    plotter.run(mode=mode)


__all__ = [
    "VarSpec",
    "PlotConfig",
    "ComparisonPlotter",
    "discover_pairs",
    "discover_triples",
    "load_pair",
    "load_triple",
    "fill_warmup_truth_from_panels",
    "plot_ticker_to_pdf",
    "save_individual_variable_pngs",
    "plot_ticker_outputs",
    "main",
]


if __name__ == "__main__":
    main()
