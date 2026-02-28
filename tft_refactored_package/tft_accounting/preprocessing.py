"""Dataset preprocessing pipeline for unconditional TFT forecasting.

NumPy and pandas remain in use here because the reviewer explicitly allows them
for data access and preprocessing. The refactor focuses on packaging the logic
into classes, adding documentation, and preserving the original outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .prepare_data import (
    add_time_features,
    compute_residuals_and_audits,
    fetch_with_retries,
    merge_and_align_statements,
)


TEST_TICKERS_DEFAULT = ["0700.HK", "9988.HK", "GOOG", "JPM", "MSFT", "VWAGY", "XOM"]
PERIOD_DAYS = {"A": 365.0, "Q": 365.0 / 4.0}


@dataclass(frozen=True)
class TargetSpec:
    """Specification for one forecasting target."""

    name: str
    kind: str
    lo: Optional[float] = None
    hi: Optional[float] = None


TARGET_SPECS: List[TargetSpec] = [
    TargetSpec("logS", "real"),
    TargetSpec("m_gross", "bounded", 0.0, 1.0),
    TargetSpec("m_opex", "bounded", 0.0, 1.0),
    TargetSpec("DSO", "bounded", 0.0, 720.0),
    TargetSpec("DIO", "bounded", 0.0, 720.0),
    TargetSpec("DPO", "bounded", 0.0, 720.0),
    TargetSpec("alpha_OCA", "bounded", 0.0, 0.50),
    TargetSpec("alpha_ONCA", "bounded", 0.0, 1.00),
    TargetSpec("alpha_OCL", "bounded", 0.0, 0.50),
    TargetSpec("alpha_ONCL", "bounded", 0.0, 1.00),
    TargetSpec("kappa", "bounded", 0.0, 0.80),
    TargetSpec("delta", "bounded", 0.0, 0.50),
    TargetSpec("payout", "bounded", 0.0, 1.00),
    TargetSpec("neteq_to_sales", "signed", -1.0, 1.0),
    TargetSpec("phi", "bounded", 0.0, 0.50),
    TargetSpec("r_ST", "bounded", 0.0, 0.50),
    TargetSpec("r_LT", "bounded", 0.0, 0.50),
    TargetSpec("tau", "bounded", 0.0, 0.50),
]

STATE_COLS = [
    "C",
    "AR",
    "Inv",
    "OCA_implied",
    "K",
    "ONCA_implied",
    "AP",
    "OCL_implied",
    "STD",
    "LTD",
    "ONCL_implied",
    "E0",
]

HIST_FEAT_COLS = [
    "logS",
    "logS_growth",
    "m_gross",
    "m_opex",
    "dso_impl",
    "dio_impl",
    "dpo_impl",
    "capex_to_sales",
    "dep_to_ppe",
    "cash_to_sales",
    "debt_to_assets",
    "year_norm",
    "q_sin",
    "q_cos",
    "dt_norm",
]
FUT_FEAT_COLS = ["year_norm", "q_sin", "q_cos", "dt_norm", "horizon_norm"]


@dataclass
class PreprocessConfig:
    """Configuration for dataset preprocessing."""

    universe_csv: str = "DataPrepare.csv"
    freq: str = "MIX"
    lookback: int = 5
    horizon: int = 2
    out_dir: str = "data_uncond"
    test_tickers: str = ",".join(TEST_TICKERS_DEFAULT)
    val_frac: float = 0.0
    seed: int = 42
    avail_threshold: float = 0.20


class FeatureEngineering:
    """Collection of deterministic feature-engineering helpers."""

    @staticmethod
    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        den2 = den.where(den.abs() > 1e-12, np.nan)
        return num / den2

    @staticmethod
    def is_financial(sector: str) -> bool:
        sector_lower = (sector or "").lower()
        return ("financial" in sector_lower) or ("bank" in sector_lower)

    @staticmethod
    def quarter_sincos(dates: pd.Series) -> Tuple[pd.Series, pd.Series]:
        quarter = pd.to_datetime(dates).dt.quarter.astype(float)
        angle = 2.0 * np.pi * (quarter - 1.0) / 4.0
        return np.sin(angle), np.cos(angle)

    @classmethod
    def add_labels_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Add label columns and hand-crafted features to one panel."""
        df = df.sort_values("date").reset_index(drop=True).copy()
        dt = df["period_days"].astype(float)
        df["S"] = df["S"].astype(float)
        df["logS"] = np.log(df["S"].where(df["S"] > 0, np.nan))
        df["logS_growth"] = df["logS"].diff()

        df["m_gross"] = (1.0 - cls.safe_div(df["COGS"].astype(float), df["S"].astype(float))).clip(0.0, 1.0)
        df["m_opex"] = cls.safe_div(df["OPEX"].astype(float), df["S"].astype(float)).clip(0.0, 1.0)
        df["dso_impl"] = cls.safe_div(dt * df["AR"].astype(float), df["S"].astype(float)).clip(0.0, 720.0)
        df["dio_impl"] = cls.safe_div(dt * df["Inv"].astype(float), df["COGS"].astype(float)).clip(0.0, 720.0)
        df["dpo_impl"] = cls.safe_div(dt * df["AP"].astype(float), df["COGS"].astype(float)).clip(0.0, 720.0)

        df["capex_to_sales"] = cls.safe_div(df["CapEx"].astype(float).abs(), df["S"].astype(float)).clip(0.0, 0.80)
        ppe_lag = df["K"].astype(float).shift(1)
        df["dep_to_ppe"] = cls.safe_div(df["DA"].astype(float), ppe_lag).clip(0.0, 0.50)

        debt = df["STD"].astype(float).fillna(0.0) + df["LTD"].astype(float).fillna(0.0)
        df["cash_to_sales"] = cls.safe_div(df["C"].astype(float), df["S"].astype(float))
        df["debt_to_assets"] = cls.safe_div(debt, df["TA"].astype(float))
        df["alpha_OCA"] = cls.safe_div(df["OCA_implied"].astype(float), df["S"].astype(float)).clip(0.0, 0.50)
        df["alpha_ONCA"] = cls.safe_div(df["ONCA_implied"].astype(float), df["S"].astype(float)).clip(0.0, 1.00)
        df["alpha_OCL"] = cls.safe_div(df["OCL_implied"].astype(float), df["S"].astype(float)).clip(0.0, 0.50)
        df["alpha_ONCL"] = cls.safe_div(df["ONCL_implied"].astype(float), df["S"].astype(float)).clip(0.0, 1.00)

        df["DSO"] = df["dso_impl"]
        df["DIO"] = df["dio_impl"]
        df["DPO"] = df["dpo_impl"]
        df["kappa"] = df["capex_to_sales"]
        df["delta"] = df["dep_to_ppe"]

        ni_pos = df["NI"].astype(float).clip(lower=0.0)
        df["payout"] = cls.safe_div(df["Div"].astype(float).abs(), ni_pos).clip(0.0, 1.0)
        df.loc[ni_pos <= 1e-9, "payout"] = 0.0

        neteq = df["EquityIssues"].astype(float).fillna(0.0) - df.get("Buyback", 0.0)
        df["neteq_to_sales"] = cls.safe_div(neteq, df["S"].astype(float)).clip(-1.0, 1.0)
        df["phi"] = cls.safe_div(df["C"].astype(float), df["S"].astype(float)).clip(0.0, 0.50)

        debt_lag = debt.shift(1)
        r = cls.safe_div(df["I"].astype(float).abs(), debt_lag).clip(0.0, 0.50)
        df["r_ST"] = r
        df["r_LT"] = r

        ebt = (df["NI"].astype(float) + df["Tax"].astype(float)).clip(lower=0.0)
        df["tau"] = cls.safe_div(df["Tax"].astype(float).abs(), ebt).clip(0.0, 0.50)
        df.loc[ebt <= 1e-9, "tau"] = 0.0
        return df

    @classmethod
    def add_calendar(cls, df: pd.DataFrame, y_min: int, y_max: int) -> pd.DataFrame:
        dt = pd.to_datetime(df["date"])
        year = dt.dt.year.astype(float)
        year_norm = (year - float(y_min)) / float(max(1, y_max - y_min))
        q_sin, q_cos = cls.quarter_sincos(dt)
        out = df.copy()
        out["year_norm"] = year_norm
        out["q_sin"] = q_sin
        out["q_cos"] = q_cos
        out["dt_norm"] = out["period_days"].astype(float) / 365.0
        out["horizon_norm"] = 0.0
        return out


class PanelFetcher:
    """Download and assemble one company panel from Yahoo Finance."""

    @staticmethod
    def fetch_panel(ticker: str, sector: str, freq_mode: str) -> pd.DataFrame:
        """Build one aligned annual/quarterly panel."""
        ticker_obj = fetch_with_retries(ticker)
        bs_a = getattr(ticker_obj, "balance_sheet", None)
        is_a = getattr(ticker_obj, "financials", None)
        cf_a = getattr(ticker_obj, "cashflow", None)
        bs_q = getattr(ticker_obj, "quarterly_balance_sheet", None)
        is_q = getattr(ticker_obj, "quarterly_financials", None)
        cf_q = getattr(ticker_obj, "quarterly_cashflow", None)

        def build_one(bs, inc, cf, tag: str) -> pd.DataFrame:
            df = merge_and_align_statements(bs, inc, cf)
            if df is None or df.empty:
                return pd.DataFrame()
            out = df.copy()
            out["freq"] = tag
            out["period_days"] = PERIOD_DAYS.get(tag, 365.0)
            out["ticker"] = ticker
            out["sector"] = sector
            out = compute_residuals_and_audits(out)
            out = add_time_features(out)
            return out

        parts: List[pd.DataFrame] = []
        if freq_mode in ("A", "MIX"):
            annual = build_one(bs_a, is_a, cf_a, "A")
            if not annual.empty:
                parts.append(annual)
        if freq_mode in ("Q", "MIX"):
            quarterly = build_one(bs_q, is_q, cf_q, "Q")
            if not quarterly.empty:
                parts.append(quarterly)
        if not parts:
            return pd.DataFrame()

        df = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
        if freq_mode == "MIX":
            df["_prio"] = (df["freq"] == "A").astype(int)
            df = (
                df.sort_values(["date", "_prio"], ascending=[True, False])
                .drop_duplicates("date", keep="first")
                .drop(columns=["_prio"])
            )
        return df.reset_index(drop=True)


class SampleBuilder:
    """Construct sliding-window samples from one preprocessed panel."""

    @staticmethod
    def build_samples(
        df: pd.DataFrame,
        ticker_id: int,
        sector_id: int,
        sector_name: str,
        lookback: int,
        horizon: int,
    ) -> Dict[str, np.ndarray]:
        """Create one sample dictionary for one ticker panel."""
        if df is None or df.empty or len(df) < 2:
            return {}
        df = df.sort_values("date").reset_index(drop=True).copy()
        target_cols = [spec.name for spec in TARGET_SPECS]
        is_fin = FeatureEngineering.is_financial(sector_name)

        hist_list = []
        hmask_list = []
        fut_list = []
        y_list = []
        my_list = []
        y0_list = []
        tid_list = []
        sid_list = []
        size_list = []
        adate_list = []
        tdates_list = []

        def get_y0(i: int) -> np.ndarray:
            row = df.iloc[i]
            cols = [
                "C",
                "AR",
                "Inv",
                "OCA_implied",
                "K",
                "ONCA_implied",
                "AP",
                "OCL_implied",
                "STD",
                "LTD",
                "ONCL_implied",
            ]
            vals = [row.get(col, np.nan) for col in cols]
            vals = [0.0 if pd.isna(v) else float(v) for v in vals]
            ta0 = float(sum(vals[0:6]))
            tl0 = float(sum(vals[6:11]))
            e0 = float(ta0 - tl0)
            vals.append(e0)
            return np.asarray(vals, dtype=np.float32)

        n_rows = len(df)
        for i in range(0, n_rows - horizon):
            y0 = get_y0(i)
            hist = np.zeros((lookback, len(HIST_FEAT_COLS)), dtype=np.float32)
            hist_mask = np.zeros((lookback,), dtype=np.float32)
            start = i - lookback + 1
            for j in range(lookback):
                idx = start + j
                if 0 <= idx <= i:
                    feat = df.iloc[idx][HIST_FEAT_COLS].to_numpy(dtype=np.float32, copy=True)
                    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                    hist[j] = feat
                    hist_mask[j] = 1.0

            future = np.zeros((horizon, len(FUT_FEAT_COLS)), dtype=np.float32)
            target_dates: List[str] = []
            for k in range(1, horizon + 1):
                row = df.iloc[i + k]
                feat = row[FUT_FEAT_COLS].to_numpy(dtype=np.float32, copy=True)
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                feat[-1] = float(k) / float(max(1, horizon))
                future[k - 1] = feat
                target_dates.append(str(pd.to_datetime(row["date"]).date()))

            y = np.zeros((horizon, len(target_cols)), dtype=np.float32)
            my = np.zeros_like(y)
            for k in range(1, horizon + 1):
                row = df.iloc[i + k]
                for j, col in enumerate(target_cols):
                    value = row.get(col, np.nan)
                    if pd.isna(value):
                        y[k - 1, j] = 0.0
                        my[k - 1, j] = 0.0
                    else:
                        y[k - 1, j] = float(value)
                        my[k - 1, j] = 1.0
            if is_fin:
                for name in ["m_gross", "DSO", "DIO", "DPO"]:
                    j = target_cols.index(name)
                    y[:, j] = 0.0
                    my[:, j] = 0.0

            ta = df.iloc[i].get("TA", np.nan)
            size = float(np.log(max(1.0, float(ta)))) if pd.notna(ta) else 0.0

            hist_list.append(hist)
            hmask_list.append(hist_mask)
            fut_list.append(future)
            y_list.append(y)
            my_list.append(my)
            y0_list.append(y0)
            tid_list.append(ticker_id)
            sid_list.append(sector_id)
            size_list.append(size)
            adate_list.append(str(pd.to_datetime(df.iloc[i]["date"]).date()))
            tdates_list.append(target_dates)

        if not hist_list:
            return {}
        return {
            "hist_feats": np.stack(hist_list),
            "hist_mask": np.stack(hmask_list),
            "future_feats": np.stack(fut_list),
            "y_true": np.stack(y_list),
            "mask_y": np.stack(my_list),
            "y0": np.stack(y0_list),
            "ticker_id": np.asarray(tid_list, np.int32),
            "sector_id": np.asarray(sid_list, np.int32),
            "size_log_ta": np.asarray(size_list, np.float32).reshape(-1, 1),
            "anchor_date": np.asarray(adate_list, dtype=object),
            "target_dates": np.asarray(tdates_list, dtype=object),
        }


class DatasetSerializer:
    """Serialization and normalization utilities."""

    @staticmethod
    def concat_dicts(dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        if not dicts:
            return {}
        keys = dicts[0].keys()
        return {key: np.concatenate([d[key] for d in dicts if key in d and d[key].size > 0], axis=0) for key in keys}

    @staticmethod
    def standardize(train: Dict[str, np.ndarray], others: List[Dict[str, np.ndarray]], eps: float = 1e-8):
        hist = train["hist_feats"]
        hist_mask = train["hist_mask"][..., None]
        denom = hist_mask.sum(axis=(0, 1)) + eps
        hist_mean = (hist * hist_mask).sum(axis=(0, 1)) / denom
        hist_std = np.sqrt(((hist - hist_mean) ** 2 * hist_mask).sum(axis=(0, 1)) / denom + eps)

        future = train["future_feats"]
        future_mean = future.mean(axis=(0, 1))
        future_std = future.std(axis=(0, 1)) + eps

        def apply(d: Dict[str, np.ndarray]) -> None:
            d["hist_feats"] = (d["hist_feats"] - hist_mean) / hist_std
            d["future_feats"] = (d["future_feats"] - future_mean) / future_std

        apply(train)
        for other in others:
            apply(other)
        return hist_mean, hist_std, future_mean, future_std

    @staticmethod
    def save_npz(path: str, data: Dict[str, np.ndarray]) -> None:
        np.savez_compressed(path, **data)


class AvailabilityAnalyzer:
    """Compute target-availability statistics and apply sparse-target masking."""

    @staticmethod
    def finite_rate(arr: pd.Series) -> float:
        values = pd.to_numeric(arr, errors="coerce").to_numpy(dtype=float)
        if values.size == 0:
            return 0.0
        return float(np.isfinite(values).mean())

    @staticmethod
    def sector_theta_medians(
        panels: Dict[str, pd.DataFrame],
        universe: pd.DataFrame,
        train_tickers: List[str],
    ) -> Dict[str, Dict[str, float]]:
        theta_cols = [spec.name for spec in TARGET_SPECS if spec.name != "logS"]
        out: Dict[str, Dict[str, float]] = {}
        subset = universe[universe["ticker"].isin(train_tickers)]
        for sector, group in subset.groupby("sector"):
            rows = []
            for ticker in group["ticker"].tolist():
                df = panels.get(ticker)
                if df is None or df.empty:
                    continue
                rows.append(df[theta_cols])
            if not rows:
                continue
            big = pd.concat(rows, ignore_index=True)
            med = big.median(numeric_only=True)
            out[str(sector)] = {col: float(med.get(col, np.nan)) for col in theta_cols}
        return out


class UncondDatasetPreprocessor:
    """End-to-end preprocessing pipeline for the unconditional TFT dataset.

    Parameters
    ----------
    config:
        Optional :class:`PreprocessConfig` instance.
    **kwargs:
        Keyword arguments used to build :class:`PreprocessConfig` directly. This
        allows package-style usage such as
        ``UncondDatasetPreprocessor(universe_csv=..., out_dir=...).run()``.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either 'config' or keyword arguments, not both.")
        self.config = config if config is not None else PreprocessConfig(**kwargs)

    def _load_universe(self) -> pd.DataFrame:
        universe = pd.read_csv(self.config.universe_csv)
        universe = universe.rename(columns={col: col.strip().lower() for col in universe.columns})
        if "lable" in universe.columns and "label" not in universe.columns:
            universe = universe.rename(columns={"lable": "label"})
        if "label" not in universe.columns:
            universe["label"] = ""
        if "sector" not in universe.columns:
            universe["sector"] = "Unknown"
        universe["ticker"] = universe["ticker"].astype(str).str.strip()
        universe["sector"] = universe["sector"].astype(str).str.strip()
        universe["label"] = universe["label"].astype(str).str.strip().str.lower()
        return universe

    def _collect_panels(self, universe: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], List[int]]:
        panels: Dict[str, pd.DataFrame] = {}
        years: List[int] = []
        for _, row in universe.iterrows():
            ticker = row["ticker"]
            sector = row["sector"]
            print(f"[download] {ticker} ({sector})")
            try:
                df = PanelFetcher.fetch_panel(ticker, sector, self.config.freq)
            except Exception as exc:
                print(f"  !! fail {ticker}: {exc}")
                continue
            if df is None or df.empty:
                print(f"  !! empty {ticker}")
                continue
            for col in [
                "S",
                "COGS",
                "OPEX",
                "AR",
                "Inv",
                "AP",
                "C",
                "K",
                "TA",
                "TL",
                "I",
                "NI",
                "Tax",
                "DA",
                "CapEx",
                "Div",
                "EquityIssues",
                "Buyback",
            ]:
                if col not in df.columns:
                    df[col] = np.nan
            df = FeatureEngineering.add_labels_features(df)
            panels[ticker] = df
            years.extend(pd.to_datetime(df["date"]).dt.year.astype(int).tolist())
        return panels, years

    def run(self) -> None:
        """Execute preprocessing and write the same files as the original script."""
        os.makedirs(self.config.out_dir, exist_ok=True)
        universe = self._load_universe()

        test_tickers = [ticker.strip() for ticker in self.config.test_tickers.split(",") if ticker.strip()]
        labeled = universe.loc[universe["label"].isin(["test", "holdout"]), "ticker"].tolist()
        for ticker in labeled:
            if ticker not in test_tickers:
                test_tickers.append(ticker)

        panels, years = self._collect_panels(universe)
        if not panels:
            raise RuntimeError("No panels built. Check network and tickers.")

        y_min = int(np.min(years)) if years else 2000
        y_max = int(np.max(years)) if years else 2025
        for ticker, df in list(panels.items()):
            panels[ticker] = FeatureEngineering.add_calendar(df, y_min, y_max)

        sectors = sorted(set(universe["sector"].tolist()))
        sector_to_id = {sector: i for i, sector in enumerate(sectors)}
        tickers = sorted(set(panels.keys()))
        ticker_to_id = {ticker: i for i, ticker in enumerate(tickers)}

        all_tickers = universe["ticker"].tolist()
        train_tickers = [ticker for ticker in all_tickers if ticker in panels and ticker not in test_tickers]
        test_tickers = [ticker for ticker in test_tickers if ticker in panels]
        val_tickers = set()

        train_samples = []
        val_samples = []
        test_samples = []
        for _, row in universe.iterrows():
            ticker = row["ticker"]
            sector = row["sector"]
            if ticker not in panels:
                continue
            sample = SampleBuilder.build_samples(
                panels[ticker],
                ticker_to_id[ticker],
                sector_to_id[sector],
                sector,
                self.config.lookback,
                self.config.horizon,
            )
            if not sample:
                continue
            if ticker in test_tickers:
                test_samples.append(sample)
            elif ticker in val_tickers:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

        train = DatasetSerializer.concat_dicts(train_samples)
        val = DatasetSerializer.concat_dicts(val_samples) if val_samples else {k: v[:0] for k, v in train.items()}
        test = DatasetSerializer.concat_dicts(test_samples) if test_samples else {}
        if train.get("hist_feats", np.empty((0,))).shape[0] == 0:
            raise RuntimeError("Training set has 0 samples.")

        theta_cols = [spec.name for spec in TARGET_SPECS if spec.name != "logS"]
        target_cols = [spec.name for spec in TARGET_SPECS]

        global_rows = []
        for ticker in train_tickers:
            df = panels.get(ticker)
            if df is None or df.empty:
                continue
            cols = [col for col in theta_cols if col in df.columns]
            if cols:
                global_rows.append(df[cols])
        global_big = pd.concat(global_rows, ignore_index=True) if global_rows else pd.DataFrame(columns=theta_cols)

        global_theta_medians = {
            col: float(pd.to_numeric(global_big.get(col, pd.Series(dtype=float)), errors="coerce").median())
            if col in global_big.columns
            else float("nan")
            for col in theta_cols
        }
        global_avail_panel = {
            col: AvailabilityAnalyzer.finite_rate(global_big[col]) if col in global_big.columns else 0.0 for col in theta_cols
        }

        by_sector_avail_panel: Dict[str, Dict[str, float]] = {}
        subset_train = universe[universe["ticker"].isin(train_tickers)]
        for sector, group in subset_train.groupby("sector"):
            rows = []
            for ticker in group["ticker"].tolist():
                df = panels.get(ticker)
                if df is None or df.empty:
                    continue
                cols = [col for col in theta_cols if col in df.columns]
                if cols:
                    rows.append(df[cols])
            if rows:
                big = pd.concat(rows, ignore_index=True)
                by_sector_avail_panel[str(sector)] = {
                    col: AvailabilityAnalyzer.finite_rate(big[col]) if col in big.columns else 0.0 for col in theta_cols
                }

        disable = np.zeros((len(sectors), len(target_cols)), dtype=np.float32)
        sparse_targets_by_sector: Dict[str, List[str]] = {}
        thr = float(self.config.avail_threshold)
        if thr > 0.0:
            for sector_name, rates in by_sector_avail_panel.items():
                sid = sector_to_id.get(sector_name, None)
                if sid is None:
                    continue
                sparse_list = []
                for j, name in enumerate(target_cols):
                    if name == "logS":
                        continue
                    if float(rates.get(name, 0.0)) < thr:
                        disable[sid, j] = 1.0
                        sparse_list.append(name)
                if sparse_list:
                    sparse_targets_by_sector[str(sector_name)] = sparse_list

        def apply_disable(d: Dict[str, np.ndarray]) -> None:
            if not d or d.get("mask_y", np.empty((0,))).size == 0:
                return
            sid = d["sector_id"].astype(int)
            keep = 1.0 - disable[sid][:, None, :]
            d["mask_y"] = d["mask_y"] * keep
            d["y_true"] = d["y_true"] * keep

        apply_disable(train)
        apply_disable(val)
        apply_disable(test)

        by_sector_avail_trainmask: Dict[str, Dict[str, float]] = {}
        global_avail_trainmask: Dict[str, float] = {}
        mask_y = train["mask_y"]
        for j, name in enumerate(target_cols):
            if name != "logS":
                global_avail_trainmask[name] = float(mask_y[:, :, j].mean())
        for sector_name, sid in sector_to_id.items():
            idx = np.where(train["sector_id"].astype(int) == int(sid))[0]
            if idx.size == 0:
                continue
            mm = train["mask_y"][idx]
            by_sector_avail_trainmask[str(sector_name)] = {
                name: float(mm[:, :, j].mean()) for j, name in enumerate(target_cols) if name != "logS"
            }

        h_mean, h_std, f_mean, f_std = DatasetSerializer.standardize(train, [val, test] if test else [val])
        DatasetSerializer.save_npz(os.path.join(self.config.out_dir, "train.npz"), train)
        DatasetSerializer.save_npz(os.path.join(self.config.out_dir, "val.npz"), val)
        if test:
            DatasetSerializer.save_npz(os.path.join(self.config.out_dir, "test.npz"), test)

        meta = {
            "lookback": self.config.lookback,
            "horizon": self.config.horizon,
            "freq_mode": self.config.freq,
            "tickers": tickers,
            "sectors": sectors,
            "ticker_to_id": ticker_to_id,
            "sector_to_id": sector_to_id,
            "hist_feat_cols": HIST_FEAT_COLS,
            "fut_feat_cols": FUT_FEAT_COLS,
            "state_cols": STATE_COLS,
            "target_specs": [spec.__dict__ for spec in TARGET_SPECS],
            "scalers": {
                "hist_mean": h_mean.tolist(),
                "hist_std": h_std.tolist(),
                "fut_mean": f_mean.tolist(),
                "fut_std": f_std.tolist(),
            },
            "global_year_min": y_min,
            "global_year_max": y_max,
            "test_tickers": test_tickers,
            "val_tickers": sorted(list(val_tickers)),
            "universe_csv": os.path.abspath(self.config.universe_csv),
            "avail_threshold": float(self.config.avail_threshold),
            "target_availability": {"global": global_avail_trainmask, "by_sector": by_sector_avail_trainmask},
            "target_availability_panel": {"global": global_avail_panel, "by_sector": by_sector_avail_panel},
            "sparse_targets_by_sector": sparse_targets_by_sector,
            "global_theta_medians": global_theta_medians,
        }
        with open(os.path.join(self.config.out_dir, "meta.json"), "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2, ensure_ascii=False)

        sector_medians = AvailabilityAnalyzer.sector_theta_medians(panels, universe, train_tickers=train_tickers)
        with open(os.path.join(self.config.out_dir, "sector_theta_medians.json"), "w", encoding="utf-8") as file:
            json.dump(sector_medians, file, indent=2, ensure_ascii=False)

        panels_dir = os.path.join(self.config.out_dir, "panels")
        os.makedirs(panels_dir, exist_ok=True)
        for ticker, df in panels.items():
            safe_name = ticker.replace("/", "_").replace(":", "_")
            df.to_csv(os.path.join(panels_dir, f"{safe_name}.csv"), index=False)
        print(f"[OK] Saved {len(panels)} panel CSVs to {panels_dir}")
        print(f"[done] out_dir={self.config.out_dir}")
        print(
            f"  train={train['hist_feats'].shape[0]} val={val['hist_feats'].shape[0]} "
            f"test={(test['hist_feats'].shape[0] if test else 0)}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser while preserving the original arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe_csv", type=str, default="DataPrepare.csv")
    parser.add_argument("--freq", type=str, default="MIX", choices=["A", "Q", "MIX"])
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="data_uncond")
    parser.add_argument("--test_tickers", type=str, default=",".join(TEST_TICKERS_DEFAULT))
    parser.add_argument("--val_frac", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--avail_threshold",
        type=float,
        default=0.20,
        help="Availability threshold used for sparse-target masking.",
    )
    return parser


def main() -> None:
    """Command-line entry point."""
    args = build_arg_parser().parse_args()
    preprocessor = UncondDatasetPreprocessor(PreprocessConfig(**vars(args)))
    preprocessor.run()


# Backward-compatible function aliases.
safe_div = FeatureEngineering.safe_div
is_financial = FeatureEngineering.is_financial
quarter_sincos = FeatureEngineering.quarter_sincos
fetch_panel = PanelFetcher.fetch_panel
add_labels_features = FeatureEngineering.add_labels_features
add_calendar = FeatureEngineering.add_calendar
build_samples = SampleBuilder.build_samples
concat_dicts = DatasetSerializer.concat_dicts
standardize = DatasetSerializer.standardize
save_npz = DatasetSerializer.save_npz
sector_theta_medians = AvailabilityAnalyzer.sector_theta_medians


__all__ = [
    "TargetSpec",
    "TARGET_SPECS",
    "STATE_COLS",
    "HIST_FEAT_COLS",
    "FUT_FEAT_COLS",
    "PreprocessConfig",
    "FeatureEngineering",
    "PanelFetcher",
    "SampleBuilder",
    "DatasetSerializer",
    "AvailabilityAnalyzer",
    "UncondDatasetPreprocessor",
    "build_arg_parser",
    "main",
]
