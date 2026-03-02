from __future__ import annotations

import json

import pandas as pd
import pytest


def test_backtester_respects_explicit_ticker_selection(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    import tft_accounting.backtesting as backtesting_module
    from tft_accounting.backtesting import BacktestConfig, RollingTFTBacktester

    out_dir = tmp_path / "results_tft"
    obj = RollingTFTBacktester.__new__(RollingTFTBacktester)
    obj.config = BacktestConfig(
        data_dir=str(tmp_path),
        ckpt_dir=str(tmp_path),
        out_dir=str(out_dir),
        tickers=["BBB", "AAA"],
    )
    obj.meta = {"test_tickers": ["AAA", "BBB", "CCC"]}

    requested = []

    monkeypatch.setattr(backtesting_module, "load_panel", lambda data_dir, ticker: pd.DataFrame({"sector": ["Tech"]}))

    def _run_one(df, ticker, sector, warmup, disable_interest):
        requested.append(ticker)
        return pd.DataFrame({
            "ticker": [ticker],
            "step": [1],
            "mask_logS": [1],
            "logS_pred": [0.1],
            "logS_true": [0.2],
        })

    obj.run_backtest_one_ticker = _run_one
    obj.run_all()

    assert requested == ["BBB", "AAA"]
    assert (out_dir / "BBB_tft_backtest.csv").exists()
    assert (out_dir / "AAA_tft_backtest.csv").exists()
    assert not (out_dir / "CCC_tft_backtest.csv").exists()


def test_theory_runner_respects_explicit_ticker_selection(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    import tft_accounting.theory as theory_module
    from tft_accounting.theory import TheoryBacktestRunner, TheoryConfig

    data_dir = tmp_path / "data_uncond"
    out_dir = tmp_path / "results_theory"
    data_dir.mkdir()
    out_dir.mkdir()

    (data_dir / "meta.json").write_text(json.dumps({"test_tickers": ["AAA", "BBB", "CCC"]}))
    (data_dir / "sector_theta_medians.json").write_text(json.dumps({
        "theta_cols": ["m_gross"],
        "sector_medians": {"Tech": {"m_gross": 0.1}},
    }))

    requested = []
    monkeypatch.setattr(theory_module, "load_panel", lambda data_dir, ticker: pd.DataFrame({"sector": ["Tech"]}))

    def _run_one(df, ticker, sector, theta, warmup, min_ar1_points, disable_interest):
        requested.append(ticker)
        return pd.DataFrame({
            "ticker": [ticker],
            "step": [1],
            "mask_logS": [1],
            "logS_pred": [0.1],
            "logS_true": [0.2],
        })

    monkeypatch.setattr(theory_module, "run_backtest_one_ticker", _run_one)

    runner = TheoryBacktestRunner(
        TheoryConfig(
            data_dir=str(data_dir),
            out_dir=str(out_dir),
            tickers=["CCC", "AAA"],
        )
    )
    runner.run()

    assert requested == ["CCC", "AAA"]
    assert (out_dir / "CCC_theory_backtest.csv").exists()
    assert (out_dir / "AAA_theory_backtest.csv").exists()
    assert not (out_dir / "BBB_theory_backtest.csv").exists()


def test_llm_runner_respects_explicit_ticker_selection(tmp_path, monkeypatch):
    import tft_accounting.LLM_based as llm_module
    from tft_accounting.LLM_based import LLMApiConfig, LLMBacktestConfig, LLMBacktestRunner

    data_dir = tmp_path / "data_uncond"
    out_dir = tmp_path / "result_llm"
    data_dir.mkdir()
    out_dir.mkdir()

    (data_dir / "meta.json").write_text(json.dumps({"test_tickers": ["AAA", "BBB", "CCC"]}), encoding="utf-8")
    (data_dir / "sector_theta_medians.json").write_text(json.dumps({
        "theta_cols": ["m_gross"],
        "sector_medians": {"Tech": {"m_gross": 0.1}},
    }), encoding="utf-8")

    requested = []
    monkeypatch.setattr(llm_module, "load_panel", lambda data_dir, ticker: pd.DataFrame({"sector": ["Tech"]}))

    def _run_one(self, df, ticker, sector, warmup, disable_interest):
        requested.append(ticker)
        return pd.DataFrame({
            "ticker": [ticker],
            "sector": [sector],
            "idx": [1],
            "step": [1],
            "date": ["2022-03-31"],
            "logS_pred": [0.1],
            "logS_true": [0.2],
            "mask_logS": [1],
        }), []

    monkeypatch.setattr(LLMBacktestRunner, "run_backtest_one_ticker", _run_one)

    runner = LLMBacktestRunner(
        LLMBacktestConfig(
            data_dir=str(data_dir),
            out_dir=str(out_dir),
            api=LLMApiConfig(enabled=False),
            tickers=["CCC", "AAA"],
            duplicate_tft_filename=False,
        )
    )
    runner.run()

    assert requested == ["CCC", "AAA"]
    assert (out_dir / "CCC_llm_backtest.csv").exists()
    assert (out_dir / "AAA_llm_backtest.csv").exists()
    assert not (out_dir / "BBB_llm_backtest.csv").exists()
