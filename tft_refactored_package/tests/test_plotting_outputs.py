from __future__ import annotations

from pathlib import Path

import pandas as pd

from tft_accounting.plotting import ComparisonPlotter, PlotConfig


def test_plotter_saves_pdf_and_high_res_pngs(tmp_path):
    theory_dir = tmp_path / "results_theory"
    tft_dir = tmp_path / "results_tft"
    out_dir = tmp_path / "figures_compare"
    theory_dir.mkdir()
    tft_dir.mkdir()

    dates = pd.date_range("2022-01-01", periods=4, freq="Q")
    theory_df = pd.DataFrame({
        "idx": [0, 1, 2, 3],
        "date": dates,
        "ticker": ["AAA"] * 4,
        "sector": ["Tech"] * 4,
        "step": [-1, 0, 1, 2],
        "logS_pred": [0.0, 0.0, 1.0, 1.1],
        "logS_true": [0.8, 0.9, 1.05, 1.15],
        "mask_logS": [1, 1, 1, 1],
    })
    tft_df = pd.DataFrame({
        "idx": [0, 1, 2, 3],
        "date": dates,
        "ticker": ["AAA"] * 4,
        "sector": ["Tech"] * 4,
        "step": [-1, 0, 1, 2],
        "logS_pred": [0.0, 0.0, 1.02, 1.12],
        "logS_pred_q10": [0.0, 0.0, 0.95, 1.05],
        "logS_pred_q50": [0.0, 0.0, 1.02, 1.12],
        "logS_pred_q90": [0.0, 0.0, 1.08, 1.18],
        "logS_true": [0.8, 0.9, 1.05, 1.15],
        "mask_logS": [1, 1, 1, 1],
    })

    theory_df.to_csv(theory_dir / "AAA_theory_backtest.csv", index=False)
    tft_df.to_csv(tft_dir / "AAA_tft_backtest.csv", index=False)

    plotter = ComparisonPlotter(
        PlotConfig(
            theory_dir=str(theory_dir),
            tft_dir=str(tft_dir),
            out_dir=str(out_dir),
            tickers=["AAA"],
            group="logs_theta",
            png_dpi=300,
        )
    )
    plotter.run()

    assert (out_dir / "AAA_compare_logs_theta.pdf").exists()
    png_dir = out_dir / "AAA"
    pngs = sorted(png_dir.glob("*.png"))
    assert png_dir.exists()
    assert len(pngs) > 0
    assert any("logS" in png.name for png in pngs)


def test_plotter_saves_triple_comparison_outputs(tmp_path):
    theory_dir = tmp_path / "results_theory"
    tft_dir = tmp_path / "results_tft"
    llm_dir = tmp_path / "result_llm"
    out_dir = tmp_path / "figures_compare_all"
    theory_dir.mkdir()
    tft_dir.mkdir()
    llm_dir.mkdir()

    dates = pd.date_range("2022-01-01", periods=4, freq="Q")
    theory_df = pd.DataFrame({
        "idx": [0, 1, 2, 3],
        "date": dates,
        "ticker": ["AAA"] * 4,
        "sector": ["Tech"] * 4,
        "step": [-1, 0, 1, 2],
        "logS_pred": [0.0, 0.0, 1.00, 1.10],
        "logS_true": [0.8, 0.9, 1.05, 1.15],
        "mask_logS": [1, 1, 1, 1],
    })
    tft_df = pd.DataFrame({
        "idx": [0, 1, 2, 3],
        "date": dates,
        "ticker": ["AAA"] * 4,
        "sector": ["Tech"] * 4,
        "step": [-1, 0, 1, 2],
        "logS_pred_q10": [0.0, 0.0, 0.95, 1.05],
        "logS_pred_q50": [0.0, 0.0, 1.02, 1.12],
        "logS_pred_q90": [0.0, 0.0, 1.08, 1.18],
        "logS_true": [0.8, 0.9, 1.05, 1.15],
        "mask_logS": [1, 1, 1, 1],
    })
    llm_df = pd.DataFrame({
        "idx": [0, 1, 2, 3],
        "date": dates,
        "ticker": ["AAA"] * 4,
        "sector": ["Tech"] * 4,
        "step": [-1, 0, 1, 2],
        "logS_pred_q10": [0.0, 0.0, 0.93, 1.03],
        "logS_pred_q50": [0.0, 0.0, 1.01, 1.11],
        "logS_pred_q90": [0.0, 0.0, 1.07, 1.17],
        "logS_true": [0.8, 0.9, 1.05, 1.15],
        "mask_logS": [1, 1, 1, 1],
    })

    theory_df.to_csv(theory_dir / "AAA_theory_backtest.csv", index=False)
    tft_df.to_csv(tft_dir / "AAA_tft_backtest.csv", index=False)
    llm_df.to_csv(llm_dir / "AAA_llm_backtest.csv", index=False)

    plotter = ComparisonPlotter(
        PlotConfig(
            theory_dir=str(theory_dir),
            tft_dir=str(tft_dir),
            llm_dir=str(llm_dir),
            out_dir=str(out_dir),
            tickers=["AAA"],
            group="logs_theta",
            png_dpi=300,
        )
    )
    plotter.run(mode="triple")

    assert (out_dir / "AAA_compare_triple_logs_theta.pdf").exists()
    png_dir = out_dir / "AAA"
    pngs = sorted(png_dir.glob("*.png"))
    assert png_dir.exists()
    assert len(pngs) > 0
    assert any("logS" in png.name for png in pngs)