from __future__ import annotations

import json

import pandas as pd
import pytest


def test_tft_backtester_writes_expected_csv_columns(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    import tft_accounting.backtesting as backtesting_module
    from tft_accounting.backtesting import BacktestConfig, RollingTFTBacktester

    out_dir = tmp_path / "results_tft"
    obj = RollingTFTBacktester.__new__(RollingTFTBacktester)
    obj.config = BacktestConfig(
        data_dir=str(tmp_path),
        ckpt_dir=str(tmp_path),
        out_dir=str(out_dir),
        tickers=["AAA"],
    )
    obj.meta = {"test_tickers": ["AAA"]}

    monkeypatch.setattr(backtesting_module, "load_panel", lambda data_dir, ticker: pd.DataFrame({"sector": ["Tech"]}))
    obj.run_backtest_one_ticker = lambda df, ticker, sector, warmup, disable_interest: pd.DataFrame({
        "ticker": [ticker],
        "sector": [sector],
        "idx": [1],
        "step": [1],
        "date": ["2022-03-31"],
        "logS_pred": [1.0],
        "logS_true": [1.1],
        "mask_logS": [1],
    })

    obj.run()

    out = pd.read_csv(out_dir / "AAA_tft_backtest.csv")
    assert {"ticker", "sector", "idx", "step", "date", "logS_pred", "logS_true", "mask_logS"}.issubset(out.columns)
    assert out.loc[0, "ticker"] == "AAA"


def test_theory_runner_writes_expected_csv_columns(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    import tft_accounting.theory as theory_module
    from tft_accounting.theory import TheoryBacktestRunner, TheoryConfig

    data_dir = tmp_path / "data_uncond"
    out_dir = tmp_path / "results_theory"
    data_dir.mkdir()
    out_dir.mkdir()

    (data_dir / "meta.json").write_text(json.dumps({"test_tickers": ["AAA"]}), encoding="utf-8")
    (data_dir / "sector_theta_medians.json").write_text(json.dumps({"Tech": {"m_gross": 0.4}}), encoding="utf-8")
    monkeypatch.setattr(theory_module, "load_panel", lambda data_dir, ticker: pd.DataFrame({"sector": ["Tech"]}))
    monkeypatch.setattr(
        theory_module,
        "run_backtest_one_ticker",
        lambda df, ticker, sector, theta, warmup, min_ar1_points, disable_interest: pd.DataFrame({
            "ticker": [ticker],
            "sector": [sector],
            "idx": [1],
            "step": [1],
            "date": ["2022-03-31"],
            "logS_pred": [1.0],
            "logS_true": [1.1],
            "mask_logS": [1],
        }),
    )

    runner = TheoryBacktestRunner(
        TheoryConfig(data_dir=str(data_dir), out_dir=str(out_dir), tickers=["AAA"])
    )
    runner.run()

    out = pd.read_csv(out_dir / "AAA_theory_backtest.csv")
    assert {"ticker", "sector", "idx", "step", "date", "logS_pred", "logS_true", "mask_logS"}.issubset(out.columns)
    assert out.loc[0, "ticker"] == "AAA"


def test_llm_runner_writes_expected_csv_columns(tmp_path, monkeypatch):
    import tft_accounting.LLM_based as llm_module
    from tft_accounting.LLM_based import LLMApiConfig, LLMBacktestConfig, LLMBacktestRunner

    data_dir = tmp_path / "data_uncond"
    out_dir = tmp_path / "result_llm"
    data_dir.mkdir()
    out_dir.mkdir()

    (data_dir / "meta.json").write_text(json.dumps({"test_tickers": ["AAA"]}), encoding="utf-8")
    (data_dir / "sector_theta_medians.json").write_text(json.dumps({
        "theta_cols": ["m_gross"],
        "sector_medians": {"Tech": {"m_gross": 0.1}},
    }), encoding="utf-8")

    monkeypatch.setattr(llm_module, "load_panel", lambda data_dir, ticker: pd.DataFrame({"sector": ["Tech"]}))

    def _run_one(self, df, ticker, sector, warmup, disable_interest):
        return pd.DataFrame({
            "ticker": [ticker],
            "sector": [sector],
            "idx": [1],
            "step": [1],
            "date": ["2022-03-31"],
            "logS_pred": [1.0],
            "logS_true": [1.1],
            "mask_logS": [1],
        }), [{"phase": "forecast", "ticker": ticker}]

    monkeypatch.setattr(LLMBacktestRunner, "run_backtest_one_ticker", _run_one)

    runner = LLMBacktestRunner(
        LLMBacktestConfig(
            data_dir=str(data_dir),
            out_dir=str(out_dir),
            api=LLMApiConfig(enabled=False),
            tickers=["AAA"],
            duplicate_tft_filename=False,
        )
    )
    runner.run()

    out = pd.read_csv(out_dir / "AAA_llm_backtest.csv")
    assert {"ticker", "sector", "idx", "step", "date", "logS_pred", "logS_true", "mask_logS"}.issubset(out.columns)
    assert out.loc[0, "ticker"] == "AAA"
    assert (out_dir / "raw_json" / "AAA_llm_outputs.json").exists()
    assert (out_dir / "run_summary.json").exists()
