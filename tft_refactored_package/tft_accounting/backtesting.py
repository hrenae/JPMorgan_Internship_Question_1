"""Rolling TFT backtesting utilities.

This module refactors the original inference script into reusable classes while
preserving the same CSV schema and one-step rolling logic.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from .model import TargetSpec, UncondTFT
from .theory import (
    DT_MAX,
    DT_MIN,
    STATE_COLS,
    _get_first_finite,
    _get_logS,
    _has_full_truth_state,
    _is_financial_sector,
    _nan_to_zero_with_mask,
    _safe_name,
    _truth_state_from_row,
    load_panel,
    simulate_step,
)


THETA_KEYS_EVAL = [
    "m_gross",
    "m_opex",
    "DSO",
    "DIO",
    "DPO",
    "alpha_OCA",
    "alpha_ONCA",
    "alpha_OCL",
    "alpha_ONCL",
    "kappa",
    "delta",
    "payout",
    "neteq_to_sales",
    "phi",
    "r_ST",
    "r_LT",
    "tau",
]
FLOW_KEYS_EVAL = ["COGS", "OPEX", "Tax", "NI", "Div", "Int", "TA", "TL", "NetEq"]


def _normalize_ticker_selection(tickers: Optional[Sequence[str] | str]) -> Optional[List[str]]:
    """Normalize ticker selection from either a string or a sequence."""
    if tickers is None:
        return None
    if isinstance(tickers, str):
        normalized = [tok.strip() for tok in tickers.split(",") if tok.strip()]
    else:
        normalized = [str(tok).strip() for tok in tickers if str(tok).strip()]
    return normalized or None


@dataclass
class BacktestConfig:
    """Configuration for TFT rolling backtests."""

    data_dir: str
    ckpt_dir: str
    out_dir: str
    weights: str = ""
    mode: str = "backtest"
    warmup: int = 3
    save_one_file: bool = False
    disable_interest_for_banks: bool = False
    tickers: Optional[list[str] | str] = None


class JsonRepository:
    """Simple JSON IO helper."""

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)


class TargetSpecHelper:
    """Shared helper methods for inference-time target metadata."""

    @staticmethod
    def build_target_specs(meta: Dict[str, Any]) -> List[TargetSpec]:
        return [TargetSpec(**spec_dict) for spec_dict in meta["target_specs"]]

    @staticmethod
    def inverse_constraint_to_z(y: float, name: str, kind: str, lo: float, hi: float, eps: float = 1e-6) -> float:
        if kind == "real":
            return float(y)
        rng = float(hi - lo)
        if rng <= 0:
            return 0.0
        if kind == "bounded":
            p = (float(y) - float(lo)) / rng
            p = min(1.0 - eps, max(eps, p))
            return float(np.log(p / (1.0 - p)))
        if kind == "signed":
            mid = 0.5 * (hi + lo)
            sc = 0.5 * (hi - lo)
            if sc <= 0:
                return 0.0
            u = (float(y) - mid) / sc
            u = min(1.0 - eps, max(-1.0 + eps, u))
            return float(0.5 * np.log((1.0 + u) / (1.0 - u)))
        return float(y)

    @classmethod
    def build_base_z_by_sector(cls, meta: Dict[str, Any], target_specs: List[TargetSpec]) -> tf.Tensor:
        base_dir = meta.get("data_dir") or meta.get("_data_dir") or ""
        if not base_dir:
            base_dir = os.getcwd()
        med_path = os.path.join(base_dir, "sector_theta_medians.json")
        if not os.path.exists(med_path):
            return tf.zeros([len(meta["sectors"]), len(target_specs)], dtype=tf.float32)

        with open(med_path, "r", encoding="utf-8") as file:
            sector_medians = json.load(file)
        global_medians = meta.get("global_theta_medians", {})

        target_dim = len(target_specs)
        n_sectors = len(meta["sectors"])
        base_z = np.zeros((n_sectors, target_dim), dtype=np.float32)
        sector_to_id = meta.get("sector_to_id", {})
        for sec_name, sec_dict in sector_medians.items():
            sector_id = int(sector_to_id.get(sec_name, -1))
            if sector_id < 0 or sector_id >= n_sectors:
                continue
            for j, spec in enumerate(target_specs):
                name = spec.name
                if str(name).lower() in ("logs", "log_s"):
                    continue
                yb = sec_dict.get(name, None)
                if yb is None or (isinstance(yb, float) and not np.isfinite(yb)):
                    yb = global_medians.get(name, None)
                if yb is None:
                    if spec.kind == "bounded":
                        lo = 0.0 if spec.lo is None else float(spec.lo)
                        hi = 0.0 if spec.hi is None else float(spec.hi)
                        yb = 0.5 * (lo + hi)
                    else:
                        yb = 0.0
                lo = 0.0 if spec.lo is None else float(spec.lo)
                hi = 0.0 if spec.hi is None else float(spec.hi)
                base_z[sector_id, j] = cls.inverse_constraint_to_z(float(yb), name, spec.kind, lo, hi)
        return tf.constant(base_z, dtype=tf.float32)


class InferenceInputBuilder:
    """Build standardized single-sample TFT inputs."""

    def __init__(self, meta: Dict[str, Any]) -> None:
        self.meta = meta
        scalers = meta.get("scalers", {})
        hist_std = np.asarray(scalers.get("hist_std", []), dtype=np.float32)
        fut_std = np.asarray(scalers.get("fut_std", []), dtype=np.float32)
        self.hist_mean = tf.constant(np.asarray(scalers.get("hist_mean", []), dtype=np.float32), dtype=tf.float32)
        self.hist_std = tf.constant(np.where(hist_std == 0, 1.0, hist_std), dtype=tf.float32)
        self.fut_mean = tf.constant(np.asarray(scalers.get("fut_mean", []), dtype=np.float32), dtype=tf.float32)
        self.fut_std = tf.constant(np.where(fut_std == 0, 1.0, fut_std), dtype=tf.float32)

    def build(self, df: pd.DataFrame, idx: int, ticker: str, sector: str) -> Dict[str, tf.Tensor]:
        """Build one standardized input sample."""
        lookback = int(self.meta["lookback"])
        horizon = int(self.meta["horizon"])
        hist_cols = list(self.meta["hist_feat_cols"])
        fut_cols = list(self.meta["fut_feat_cols"])
        hist_dim = len(hist_cols)
        fut_dim = len(fut_cols)

        hist = np.zeros((lookback, hist_dim), dtype=np.float32)
        hist_mask = np.zeros((lookback,), dtype=np.float32)
        start = idx - (lookback - 1)
        for j in range(lookback):
            src = start + j
            if 0 <= src < len(df):
                feat = df.iloc[src][hist_cols].to_numpy(dtype=np.float32, copy=True)
                hist[j] = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                hist_mask[j] = 1.0

        future = np.zeros((horizon, fut_dim), dtype=np.float32)
        try:
            hnorm_idx = fut_cols.index("horizon_norm")
        except ValueError:
            hnorm_idx = -1
        for k in range(1, horizon + 1):
            src = idx + k
            if src >= len(df):
                src = len(df) - 1
            feat = df.iloc[src][fut_cols].to_numpy(dtype=np.float32, copy=True)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            if hnorm_idx >= 0:
                feat[hnorm_idx] = float(k) / float(max(1, horizon))
            future[k - 1] = feat

        hist_tensor = tf.convert_to_tensor(hist, dtype=tf.float32)
        future_tensor = tf.convert_to_tensor(future, dtype=tf.float32)
        if int(self.hist_mean.shape[0]) == hist_dim:
            hist_tensor = (hist_tensor - self.hist_mean) / self.hist_std
        if int(self.fut_mean.shape[0]) == fut_dim:
            future_tensor = (future_tensor - self.fut_mean) / self.fut_std

        ticker_to_id = self.meta.get("ticker_to_id", {})
        sector_to_id = self.meta.get("sector_to_id", {})
        ticker_id = int(ticker_to_id.get(ticker, 0))
        sector_id = int(sector_to_id.get(sector, 0))
        ta = df.iloc[idx].get("TA", np.nan)
        size_log_ta = float(np.log(max(1.0, float(ta)))) if pd.notna(ta) else 0.0

        return {
            "hist_feats": hist_tensor[None, ...],
            "hist_mask": tf.convert_to_tensor(hist_mask[None, ...], dtype=tf.float32),
            "future_feats": future_tensor[None, ...],
            "ticker_id": tf.constant([ticker_id], dtype=tf.int32),
            "sector_id": tf.constant([sector_id], dtype=tf.int32),
            "size_log_ta": tf.constant([[size_log_ta]], dtype=tf.float32),
        }


class RollingTFTBacktester:
    """Run rolling one-step TFT backtests over all test tickers.

    Parameters
    ----------
    config:
        Optional :class:`BacktestConfig` instance.
    **kwargs:
        Keyword arguments used to build :class:`BacktestConfig` directly.
    """

    def __init__(self, config: Optional[BacktestConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either 'config' or keyword arguments, not both.")
        self.config = config if config is not None else BacktestConfig(**kwargs)
        self.meta = JsonRepository.load(os.path.join(self.config.data_dir, "meta.json"))
        self.meta["data_dir"] = self.config.data_dir
        self.input_builder = InferenceInputBuilder(self.meta)
        self.model = self._load_model()

    @staticmethod
    def load_test_tickers(data_dir: str) -> List[str]:
        """Load the canonical test ticker list from ``meta.json``."""
        meta = JsonRepository.load(os.path.join(data_dir, "meta.json"))
        return [str(t) for t in meta.get("test_tickers", [])]

    def selected_tickers(self) -> List[str]:
        """Return the explicit ticker selection if provided, else all test tickers."""
        explicit = _normalize_ticker_selection(self.config.tickers)
        return explicit if explicit is not None else [str(t) for t in self.meta.get("test_tickers", [])]

    def _load_model(self) -> UncondTFT:
        cfg_path = os.path.join(self.config.ckpt_dir, "train_config.json")
        train_cfg = JsonRepository.load(cfg_path) if os.path.exists(cfg_path) else {}
        target_specs = TargetSpecHelper.build_target_specs(self.meta)
        residual_theta = bool(train_cfg.get("residual_theta", False))
        residual_scale = float(train_cfg.get("residual_scale", 1.0))
        base_z_by_sector = TargetSpecHelper.build_base_z_by_sector(self.meta, target_specs) if residual_theta else None

        model = UncondTFT(
            hist_dim=len(self.meta["hist_feat_cols"]),
            fut_dim=len(self.meta["fut_feat_cols"]),
            n_tickers=len(self.meta["tickers"]),
            n_sectors=len(self.meta["sectors"]),
            target_specs=target_specs,
            d_model=int(train_cfg.get("d_model", 64)),
            dropout=float(train_cfg.get("dropout", 0.10)),
            num_heads=int(train_cfg.get("num_heads", 4)),
            ticker_emb_dim=int(train_cfg.get("ticker_emb_dim", 16)),
            sector_emb_dim=int(train_cfg.get("sector_emb_dim", 8)),
            residual_theta=residual_theta,
            residual_scale=residual_scale,
            base_z_by_sector=base_z_by_sector,
        )
        dummy_x = {
            "hist_feats": tf.zeros([2, int(self.meta["lookback"]), len(self.meta["hist_feat_cols"])], tf.float32),
            "hist_mask": tf.ones([2, int(self.meta["lookback"])], tf.float32),
            "future_feats": tf.zeros([2, int(self.meta["horizon"]), len(self.meta["fut_feat_cols"])], tf.float32),
            "ticker_id": tf.zeros([2], tf.int32),
            "sector_id": tf.zeros([2], tf.int32),
            "size_log_ta": tf.zeros([2, 1], tf.float32),
        }
        _ = model(dummy_x, training=False)
        weights_path = self.config.weights.strip() or os.path.join(self.config.ckpt_dir, "best.weights.h5")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights not found: {weights_path}")
        model.load_weights(weights_path)
        return model

    def predict_step1_quantiles(self, df: pd.DataFrame, idx: int, ticker: str, sector: str) -> Dict[str, Tuple[float, float, float]]:
        """Return horizon-step-1 quantiles for all targets."""
        inputs = self.input_builder.build(df, idx, ticker, sector)
        yq = self.model(inputs, training=False)
        step1 = yq[0, 0, :, :]
        names = [spec["name"] for spec in self.meta["target_specs"]]
        target_dim = min(len(names), int(step1.shape[0]))

        output: Dict[str, Tuple[float, float, float]] = {}
        tensor_array = tf.TensorArray(dtype=tf.float32, size=target_dim, clear_after_read=False)
        for i in tf.range(target_dim):
            tensor_array = tensor_array.write(i, step1[i])
        stacked = tensor_array.stack().numpy()
        for i in range(target_dim):
            q10, q50, q90 = stacked[i, 0], stacked[i, 1], stacked[i, 2]
            output[names[i]] = (float(q10), float(q50), float(q90))
        return output

    def run_backtest_one_ticker(
        self,
        df: pd.DataFrame,
        ticker: str,
        sector: str,
        warmup: int = 3,
        disable_interest: bool = False,
    ) -> pd.DataFrame:
        """Run the original one-step rolling backtest for one ticker."""
        df = df.sort_values("date").reset_index(drop=True).copy()
        n = len(df)
        if n <= warmup:
            raise ValueError(f"{ticker}: need > warmup rows; got n={n}, warmup={warmup}")

        idx0 = warmup - 1
        row0 = df.iloc[idx0]
        st = _truth_state_from_row(row0)
        st = np.nan_to_num(st, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        logS_t = _get_logS(row0)
        if not np.isfinite(logS_t):
            hist_logS = np.asarray([_get_logS(df.iloc[i]) for i in range(idx0 + 1)], dtype=float)
            hist_logS = hist_logS[np.isfinite(hist_logS)]
            logS_t = float(hist_logS[-1]) if hist_logS.size > 0 else 0.0

        rows: List[Dict[str, Any]] = []

        for i in range(0, idx0 + 1):
            r = df.iloc[i]
            logS_true_i = _get_logS(r)
            logS_true_val_i, mask_logS_i = _nan_to_zero_with_mask(logS_true_i)
            s_true_i = _get_first_finite(r, ["S"])
            s_true_val_i, mask_s_i = _nan_to_zero_with_mask(s_true_i)

            st_true_i = _truth_state_from_row(r)
            st_true_vals_i: List[float] = []
            st_true_masks_i: List[int] = []
            for value in st_true_i.tolist():
                vv, mm = _nan_to_zero_with_mask(float(value) if np.isfinite(value) else float("nan"))
                st_true_vals_i.append(vv)
                st_true_masks_i.append(mm)

            period_days_i = _get_first_finite(r, ["period_days"])
            if not np.isfinite(period_days_i):
                period_days_i = 365.0

            theta_q10 = {k: float("nan") for k in THETA_KEYS_EVAL}
            theta_q50 = {k: float("nan") for k in THETA_KEYS_EVAL}
            theta_q90 = {k: float("nan") for k in THETA_KEYS_EVAL}
            try:
                pred_i_q = self.predict_step1_quantiles(df, i, ticker, sector)
                for k in THETA_KEYS_EVAL:
                    q10, q50, q90 = pred_i_q.get(k, (float("nan"), float("nan"), float("nan")))
                    theta_q10[k] = float(q10)
                    theta_q50[k] = float(q50)
                    theta_q90[k] = float(q90)
                if _is_financial_sector(sector):
                    for k in ["m_gross", "DSO", "DIO", "DPO"]:
                        theta_q10[k] = theta_q50[k] = theta_q90[k] = 0.0
            except Exception:
                pass

            theta_true = {}
            theta_mask = {}
            for k in THETA_KEYS_EVAL:
                v = _get_first_finite(r, [k])
                vv, mm = _nan_to_zero_with_mask(v)
                if _is_financial_sector(sector) and (k in ["m_gross", "DSO", "DIO", "DPO"]):
                    vv, mm = 0.0, 0
                theta_true[k] = vv
                theta_mask[k] = mm

            def abs_if_finite(x: float) -> float:
                return float(abs(x)) if np.isfinite(x) else float("nan")

            flow_specs = {
                "COGS": (["COGS"], None),
                "OPEX": (["OPEX"], None),
                "Tax": (["Tax"], None),
                "NI": (["NI"], None),
                "Div": (["Div"], abs_if_finite),
                "Int": (["I"], abs_if_finite),
                "TA": (["TA"], None),
                "TL": (["TL"], None),
            }
            flow_true = {}
            flow_mask = {}
            for name, (cols, fn) in flow_specs.items():
                value = _get_first_finite(r, cols)
                if fn is not None:
                    value = fn(value)
                vv, mm = _nan_to_zero_with_mask(value)
                flow_true[name] = vv
                flow_mask[name] = mm

            eq = _get_first_finite(r, ["EquityIssues"])
            bb = _get_first_finite(r, ["Buyback"])
            mm = int(np.isfinite(eq) or np.isfinite(bb))
            eqv = float(eq) if np.isfinite(eq) else 0.0
            bbv = float(bb) if np.isfinite(bb) else 0.0
            neteq_true = float(eqv - bbv) if mm else 0.0
            neteq_mask = int(mm)

            rec_warm: Dict[str, Any] = dict(
                ticker=str(ticker),
                sector=str(sector),
                date=str(pd.to_datetime(r["date"]).date()),
                idx=int(i),
                step=int(i - idx0),
                period_days=float(period_days_i),
                logS_pred=float("nan"),
                logS_pred_q10=float("nan"),
                logS_pred_q50=float("nan"),
                logS_pred_q90=float("nan"),
                logS_true=float(logS_true_val_i),
                mask_logS=int(mask_logS_i),
                S_true=float(s_true_val_i),
                mask_S=int(mask_s_i),
                **{f"theta_{k}": float(theta_q50.get(k, np.nan)) for k in THETA_KEYS_EVAL},
                **{f"theta_{k}_q10": float(theta_q10.get(k, np.nan)) for k in THETA_KEYS_EVAL},
                **{f"theta_{k}_q50": float(theta_q50.get(k, np.nan)) for k in THETA_KEYS_EVAL},
                **{f"theta_{k}_q90": float(theta_q90.get(k, np.nan)) for k in THETA_KEYS_EVAL},
            )
            rec_warm.update({f"theta_true_{k}": float(v) for k, v in theta_true.items()})
            rec_warm.update({f"mask_theta_{k}": int(m) for k, m in theta_mask.items()})
            for name in flow_specs.keys():
                rec_warm[f"{name}_true"] = float(flow_true[name])
                rec_warm[f"mask_{name}"] = int(flow_mask[name])
            rec_warm["NetEq_true"] = float(neteq_true)
            rec_warm["mask_NetEq"] = int(neteq_mask)
            rec_warm.update({f"state_true_{STATE_COLS[j]}": float(st_true_vals_i[j]) for j in range(len(STATE_COLS))})
            rec_warm.update({f"mask_state_{STATE_COLS[j]}": int(st_true_masks_i[j]) for j in range(len(STATE_COLS))})
            rows.append(rec_warm)

        for idx in range(idx0, n - 1):
            row_cur = df.iloc[idx]
            row_nxt = df.iloc[idx + 1]

            logS_cur_truth = _get_logS(row_cur)
            if np.isfinite(logS_cur_truth):
                logS_t = float(logS_cur_truth)
            if _has_full_truth_state(row_cur):
                st = _truth_state_from_row(row_cur).astype(float)

            pred_q = self.predict_step1_quantiles(df, idx, ticker, sector)
            logS_q10, logS_q50, logS_q90 = pred_q.get("logS", (logS_t, logS_t, logS_t))
            logS_pred = float(logS_q50)

            theta_pred_q10 = {k: float(pred_q.get(k, (0.0, 0.0, 0.0))[0]) for k in THETA_KEYS_EVAL}
            theta_pred_q50 = {k: float(pred_q.get(k, (0.0, 0.0, 0.0))[1]) for k in THETA_KEYS_EVAL}
            theta_pred_q90 = {k: float(pred_q.get(k, (0.0, 0.0, 0.0))[2]) for k in THETA_KEYS_EVAL}
            theta_pred = theta_pred_q50
            if _is_financial_sector(sector):
                for k in ["m_gross", "DSO", "DIO", "DPO"]:
                    theta_pred_q10[k] = theta_pred_q50[k] = theta_pred_q90[k] = 0.0

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
                st,
                float(logS_pred),
                theta_pred,
                float(period_days),
                is_financial=_is_financial_sector(sector),
                disable_interest=disable_interest,
            )
            try:
                st_pred_q10, diag_q10 = simulate_step(
                    st,
                    float(logS_q10),
                    theta_pred_q10,
                    float(period_days),
                    is_financial=_is_financial_sector(sector),
                    disable_interest=disable_interest,
                )
            except Exception:
                st_pred_q10, diag_q10 = st_pred_next, dict(diag)
            st_pred_q50, diag_q50 = st_pred_next, diag
            try:
                st_pred_q90, diag_q90 = simulate_step(
                    st,
                    float(logS_q90),
                    theta_pred_q90,
                    float(period_days),
                    is_financial=_is_financial_sector(sector),
                    disable_interest=disable_interest,
                )
            except Exception:
                st_pred_q90, diag_q90 = st_pred_next, dict(diag)

            logS_true = _get_logS(row_nxt)
            logS_true_val, mask_logS = _nan_to_zero_with_mask(logS_true)
            s_true = _get_first_finite(row_nxt, ["S"])
            s_true_val, mask_s = _nan_to_zero_with_mask(s_true)

            st_true = _truth_state_from_row(row_nxt)
            st_true_vals = []
            st_true_masks = []
            for value in st_true.tolist():
                vv, mm = _nan_to_zero_with_mask(float(value) if np.isfinite(value) else float("nan"))
                st_true_vals.append(vv)
                st_true_masks.append(mm)

            rec: Dict[str, Any] = dict(
                ticker=str(ticker),
                sector=str(sector),
                date=str(pd.to_datetime(row_nxt["date"]).date()),
                idx=int(idx + 1),
                step=int((idx + 1) - idx0),
                period_days=float(period_days),
                period_days_panel=float(period_days_panel),
                period_days_date=float(period_days_date) if np.isfinite(period_days_date) else float("nan"),
                logS_pred=float(logS_pred),
                logS_pred_q10=float(logS_q10),
                logS_pred_q50=float(logS_q50),
                logS_pred_q90=float(logS_q90),
                logS_true=float(logS_true_val),
                mask_logS=int(mask_logS),
                S_true=float(s_true_val),
                mask_S=int(mask_s),
                **{f"theta_{k}": float(theta_pred.get(k, 0.0)) for k in THETA_KEYS_EVAL},
                **{f"theta_{k}_q10": float(theta_pred_q10.get(k, 0.0)) for k in THETA_KEYS_EVAL},
                **{f"theta_{k}_q50": float(theta_pred_q50.get(k, 0.0)) for k in THETA_KEYS_EVAL},
                **{f"theta_{k}_q90": float(theta_pred_q90.get(k, 0.0)) for k in THETA_KEYS_EVAL},
            )

            theta_true = {}
            theta_mask = {}
            for k in THETA_KEYS_EVAL:
                value = _get_first_finite(row_nxt, [k])
                vv, mm = _nan_to_zero_with_mask(value)
                if _is_financial_sector(sector) and (k in ["m_gross", "DSO", "DIO", "DPO"]):
                    vv, mm = 0.0, 0
                theta_true[k] = vv
                theta_mask[k] = mm
            rec.update({f"theta_true_{k}": float(v) for k, v in theta_true.items()})
            rec.update({f"mask_theta_{k}": int(m) for k, m in theta_mask.items()})

            def abs_if_finite(x: float) -> float:
                return float(abs(x)) if np.isfinite(x) else float("nan")

            flow_specs = {
                "COGS": (["COGS"], None),
                "OPEX": (["OPEX"], None),
                "Tax": (["Tax"], None),
                "NI": (["NI"], None),
                "Div": (["Div"], abs_if_finite),
                "Int": (["I"], abs_if_finite),
                "TA": (["TA"], None),
                "TL": (["TL"], None),
            }
            for name, (cols, fn) in flow_specs.items():
                value = _get_first_finite(row_nxt, cols)
                if fn is not None:
                    value = fn(value)
                vv, mm = _nan_to_zero_with_mask(value)
                rec[f"{name}_true"] = float(vv)
                rec[f"mask_{name}"] = int(mm)

            eq = _get_first_finite(row_nxt, ["EquityIssues"])
            bb = _get_first_finite(row_nxt, ["Buyback"])
            mm = int(np.isfinite(eq) or np.isfinite(bb))
            eqv = float(eq) if np.isfinite(eq) else 0.0
            bbv = float(bb) if np.isfinite(bb) else 0.0
            rec["NetEq_true"] = float(eqv - bbv) if mm else 0.0
            rec["mask_NetEq"] = int(mm)

            rec.update({f"pred_{k}": float(v) for k, v in diag.items()})
            rec.update({f"state_pred_{STATE_COLS[i]}": float(st_pred_next[i]) for i in range(len(STATE_COLS))})
            rec.update({f"state_pred_{STATE_COLS[i]}_q10": float(st_pred_q10[i]) for i in range(len(STATE_COLS))})
            rec.update({f"state_pred_{STATE_COLS[i]}_q50": float(st_pred_q50[i]) for i in range(len(STATE_COLS))})
            rec.update({f"state_pred_{STATE_COLS[i]}_q90": float(st_pred_q90[i]) for i in range(len(STATE_COLS))})
            for k in FLOW_KEYS_EVAL:
                rec[f"pred_{k}_q10"] = float(diag_q10.get(k, np.nan))
                rec[f"pred_{k}_q50"] = float(diag_q50.get(k, np.nan))
                rec[f"pred_{k}_q90"] = float(diag_q90.get(k, np.nan))
            rec.update({f"state_true_{STATE_COLS[i]}": float(st_true_vals[i]) for i in range(len(STATE_COLS))})
            rec.update({f"mask_state_{STATE_COLS[i]}": int(st_true_masks[i]) for i in range(len(STATE_COLS))})
            rows.append(rec)

            st = st_pred_next
            logS_t = float(logS_pred)

        return pd.DataFrame(rows)

    def run_all(self) -> None:
        """Run backtests for all test tickers listed in ``meta.json``."""
        os.makedirs(self.config.out_dir, exist_ok=True)
        all_frames: List[pd.DataFrame] = []
        test_tickers = self.selected_tickers()
        for ticker in test_tickers:
            df = load_panel(self.config.data_dir, ticker)
            sector = str(df["sector"].iloc[-1]) if ("sector" in df.columns and len(df) > 0) else ""
            disable_interest = bool(self.config.disable_interest_for_banks and _is_financial_sector(sector))
            try:
                out = self.run_backtest_one_ticker(
                    df=df,
                    ticker=ticker,
                    sector=sector,
                    warmup=int(self.config.warmup),
                    disable_interest=disable_interest,
                )
            except Exception as exc:
                print(f"[WARN] {ticker}: backtest skipped ({exc})")
                continue

            out_path = os.path.join(self.config.out_dir, f"{_safe_name(ticker)}_tft_backtest.csv")
            out.to_csv(out_path, index=False)
            print(f"[OK] {ticker}: {out_path}")
            if "mask_logS" in out.columns and out["mask_logS"].sum() > 0:
                valid = (out["mask_logS"] == 1) & (out["step"] >= 1) & np.isfinite(out["logS_pred"].to_numpy())
                if bool(valid.any()):
                    mae = float(np.mean(np.abs(out.loc[valid, "logS_pred"] - out.loc[valid, "logS_true"])))
                    print(f"      logS MAE (masked) = {mae:.4f}")
            all_frames.append(out)

        if self.config.save_one_file and all_frames:
            big = pd.concat(all_frames, ignore_index=True)
            out_path = os.path.join(self.config.out_dir, "backtest_all.csv")
            big.to_csv(out_path, index=False)
            print(f"[OK] saved concatenated backtest: {out_path}")

    def run(self) -> None:
        """Alias for :meth:`run_all` to support package-style orchestration."""
        self.run_all()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser while preserving the original arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True, help="Directory containing best.weights.h5 or final.weights.h5.")
    parser.add_argument("--weights", type=str, default="", help="Optional path to weights file.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", type=str, default="backtest", choices=["backtest"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--save_one_file", action="store_true")
    parser.add_argument("--disable_interest_for_banks", action="store_true")
    parser.add_argument("--tickers", type=str, default="", help="Optional comma-separated tickers to backtest.")
    return parser


def main() -> None:
    """Command-line entry point."""
    args = build_arg_parser().parse_args()
    backtester = RollingTFTBacktester(BacktestConfig(**vars(args)))
    backtester.run_all()


# Backward-compatible aliases.
_load_json = JsonRepository.load
_build_target_specs = TargetSpecHelper.build_target_specs
_inv_constraint_to_z = TargetSpecHelper.inverse_constraint_to_z
_build_base_z_by_sector = TargetSpecHelper.build_base_z_by_sector
_get_scalers = lambda meta: (
    np.asarray(meta.get("scalers", {}).get("hist_mean", []), dtype=np.float32),
    np.where(np.asarray(meta.get("scalers", {}).get("hist_std", []), dtype=np.float32) == 0, 1.0, np.asarray(meta.get("scalers", {}).get("hist_std", []), dtype=np.float32)),
    np.asarray(meta.get("scalers", {}).get("fut_mean", []), dtype=np.float32),
    np.where(np.asarray(meta.get("scalers", {}).get("fut_std", []), dtype=np.float32) == 0, 1.0, np.asarray(meta.get("scalers", {}).get("fut_std", []), dtype=np.float32)),
)
_build_one_input = lambda df, idx, ticker, sector, meta: InferenceInputBuilder(meta).build(df, idx, ticker, sector)
_predict_step1_quantiles = lambda model, df, idx, ticker, sector, meta: RollingTFTBacktester.__new__(RollingTFTBacktester)  # placeholder


def run_backtest_one_ticker_tft(
    model: UncondTFT,
    df: pd.DataFrame,
    ticker: str,
    sector: str,
    meta: Dict[str, Any],
    warmup: int = 3,
    disable_interest: bool = False,
) -> pd.DataFrame:
    """Backward-compatible function wrapper for one ticker backtest."""
    obj = RollingTFTBacktester.__new__(RollingTFTBacktester)
    obj.config = BacktestConfig(data_dir=str(meta.get("data_dir", "")), ckpt_dir="", out_dir="", warmup=warmup)
    obj.meta = meta
    obj.input_builder = InferenceInputBuilder(meta)
    obj.model = model
    return obj.run_backtest_one_ticker(df, ticker, sector, warmup=warmup, disable_interest=disable_interest)


def _predict_step1_quantiles(
    model: UncondTFT,
    df: pd.DataFrame,
    idx: int,
    ticker: str,
    sector: str,
    meta: Dict[str, Any],
) -> Dict[str, Tuple[float, float, float]]:
    """Backward-compatible function wrapper for one-step quantile prediction."""
    obj = RollingTFTBacktester.__new__(RollingTFTBacktester)
    obj.meta = meta
    obj.input_builder = InferenceInputBuilder(meta)
    obj.model = model
    return obj.predict_step1_quantiles(df, idx, ticker, sector)


__all__ = [
    "BacktestConfig",
    "RollingTFTBacktester",
    "run_backtest_one_ticker_tft",
    "main",
]
