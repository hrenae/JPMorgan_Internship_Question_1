from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tft_accounting.preprocessing import FeatureEngineering, UncondDatasetPreprocessor


def _make_raw_panel(ticker: str, sector: str) -> pd.DataFrame:
    dates = pd.date_range("2020-03-31", periods=4, freq="Q")
    base_sales = {"AAA": 100.0, "BBB": 120.0}[ticker]
    rows = []
    for i, date in enumerate(dates):
        sales = base_sales + 8.0 * i
        cogs = sales * 0.55
        row = {
            "date": date,
            "ticker": ticker,
            "sector": sector,
            "freq": "Q",
            "period_days": 91.25,
            "S": sales,
            "COGS": cogs,
            "OPEX": sales * 0.18,
            "AR": sales * 0.20,
            "Inv": cogs * 0.25,
            "AP": cogs * 0.18,
            "C": sales * 0.10,
            "K": 150.0 + 5.0 * i,
            "TA": 260.0 + 10.0 * i,
            "TL": 140.0 + 6.0 * i,
            "I": 3.0 + 0.2 * i,
            "NI": sales * 0.08,
            "Tax": sales * 0.015,
            "DA": 4.0 + 0.1 * i,
            "CapEx": 5.0 + 0.2 * i,
            "Div": 1.0 + 0.1 * i,
            "EquityIssues": 0.5,
            "Buyback": 0.2,
            "STD": 25.0 + i,
            "LTD": 50.0 + 2.0 * i,
            "OCA_implied": sales * 0.03,
            "ONCA_implied": sales * 0.05,
            "OCL_implied": sales * 0.025,
            "ONCL_implied": sales * 0.04,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df = FeatureEngineering.add_labels_features(df)
    return df


def test_preprocessor_writes_expected_artifacts(tmp_path, monkeypatch):
    universe_csv = tmp_path / "DataPrepare.csv"
    pd.DataFrame(
        [
            {"ticker": "AAA", "sector": "Tech", "label": "train"},
            {"ticker": "BBB", "sector": "Energy", "label": "test"},
        ]
    ).to_csv(universe_csv, index=False)

    panels = {
        "AAA": _make_raw_panel("AAA", "Tech"),
        "BBB": _make_raw_panel("BBB", "Energy"),
    }
    years = [2020] * 8

    def _collect_panels(self, universe):
        return panels, years

    monkeypatch.setattr(UncondDatasetPreprocessor, "_collect_panels", _collect_panels)

    out_dir = tmp_path / "data_uncond"
    preprocessor = UncondDatasetPreprocessor(
        universe_csv=str(universe_csv),
        freq="MIX",
        lookback=2,
        horizon=1,
        out_dir=str(out_dir),
    )
    preprocessor.run()

    assert (out_dir / "meta.json").exists()
    assert (out_dir / "train.npz").exists()
    assert (out_dir / "val.npz").exists()
    assert (out_dir / "test.npz").exists()
    assert (out_dir / "sector_theta_medians.json").exists()
    assert (out_dir / "panels" / "AAA.csv").exists()
    assert (out_dir / "panels" / "BBB.csv").exists()

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["lookback"] == 2
    assert meta["horizon"] == 1
    assert "BBB" in meta["test_tickers"]
    assert "hist_feat_cols" in meta and len(meta["hist_feat_cols"]) > 0
    assert "fut_feat_cols" in meta and len(meta["fut_feat_cols"]) > 0

    train = np.load(out_dir / "train.npz", allow_pickle=True)
    test = np.load(out_dir / "test.npz", allow_pickle=True)
    assert train["hist_feats"].shape[1] == 2
    assert train["future_feats"].shape[1] == 1
    assert train["y_true"].shape[-1] == len(meta["target_specs"])
    assert test["hist_feats"].shape[0] > 0
