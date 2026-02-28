"""Example package-style workflow for the refactored TFT / theory / LLM project.

Run after installing the project in editable mode:

    pip install -e .
    python examples/package_workflow.py

Adjust paths, company list, model hyperparameters, and API information as needed.
"""

from __future__ import annotations

import os

from tft_accounting import (
    ComparisonPlotter,
    LLMApiConfig,
    LLMBacktestRunner,
    RollingTFTBacktester,
    TFTTrainer,
    TheoryBacktestRunner,
    UncondDatasetPreprocessor,
)


# ---------------------------------------------------------------------
# Global switches
# ---------------------------------------------------------------------
RUN_PREPROCESS = True # Set to True to run data preprocessing. Make sure to set the correct path to "DataPrepare.csv" in the UncondDatasetPreprocessor below.
RUN_TFT_TRAINING = True # Set to True to run TFT training. Adjust hyperparameters in the TFTTrainer initialization below as needed.
RUN_TFT_BACKTEST = True # Set to True to run TFT backtesting. Make sure to run TFT training first and set the correct checkpoint directory in the RollingTFTBacktester initialization below.
RUN_THEORY_BACKTEST = True # Set to True to run theory backtesting. Adjust settings in the TheoryBacktestRunner initialization below as needed.
RUN_LLM_BACKTEST = True # Set to True to run LLM backtesting. Make sure to fill in the LLM API configuration area below and adjust settings in the LLMBacktestRunner initialization as needed.

RUN_PLOT_THEORY_TFT = False # Set to True to plot TFT vs Theory comparison. Make sure to run both backtests first and set the correct results directories in the ComparisonPlotter initialization below.
RUN_PLOT_THEORY_TFT_LLM = True # Set to True to plot TFT vs Theory vs LLM comparison. Make sure to run all backtests first and set the correct results directories in the ComparisonPlotter initialization below.

PLOT_LLM_Q10_Q90_IN_TRIPLE = False # If False, only LLM Q50 is plotted in Theory/TFT/LLM comparison mode. It's easy to check the result. Set to True to include LLM Q10 and Q90 in the triple comparison plot.

# ---------------------------------------------------------------------
# Shared experiment settings
# ---------------------------------------------------------------------
DATA_DIR = "data_uncond"
TFT_CKPT_DIR = "tft_uncond_ckpt"
RESULTS_TFT_DIR = "results_tft"
RESULTS_THEORY_DIR = "results_theory"
RESULTS_LLM_DIR = "result_llm_gpt"

FIGURES_TFT_DIR = "figures_compare"
FIGURES_ALL_DIR = "figures_compare_all"

TEST_COMPANIES = [
    "GOOG",
    "JPM",
    "MSFT",
    "VWAGY",
    "XOM",
]


# ---------------------------------------------------------------------
# LLM API configuration area
# ---------------------------------------------------------------------
# Keep these settings here so a client can modify them directly in one place.
# This module expects an OpenAI-compatible chat-completions style endpoint.
#
# Recommended practice:
#   1) Prefer environment variables for the API key.
#   2) Set RUN_LLM_BACKTEST = True.
#   3) Fill base URL / endpoint / model according to the provider.
#
# Example:
#   Linux/macOS:
#       export OPENAI_API_KEY="your_real_key"
#   Windows PowerShell:
#       $env:OPENAI_API_KEY="your_real_key"

# ---------------------------------------------------------------------
# For using in hong kong, set proxy to avoid connection issues. Remove these lines if not needed.
# ---------------------------------------------------------------------

import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY']  = 'http://127.0.0.1:7890'

# ---------------------------------------------------------------------
# Example for using gemini-2.5-flash via Google Gemini API (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------

# LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" # Gemini OpenAI 兼容端点
# LLM_ENDPOINT = "/chat/completions"
# LLM_MODEL = "gemini-2.5-flash"
# LLM_API_KEY = ""
# LLM_API_KEY_ENV = "GOOGLE_API_KEY"
# LLM_TIMEOUT_S = 120
# LLM_TEMPERATURE = 0.0
# LLM_MAX_TOKENS = 8000
# LLM_EXTRA_HEADERS = {}

# ---------------------------------------------------------------------
# For conveinience, I used api from a website called MetaChat 
# ===== MetaChat / OpenAI-compatible / OpenAI strongest general model =====
# ---------------------------------------------------------------------

LLM_BASE_URL = "https://llm-api.mmchat.xyz/v1"
LLM_ENDPOINT = "/chat/completions"
LLM_MODEL = "gpt-5.2"
# LLM_MODEL = "claude-opus-4-6"
# LLM_MODEL = "gemini-3.1-pro-preview"
LLM_API_KEY = "sk-live-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJNZXRhQ2hhdCIsInN1YiI6IjY5NTQ4YTA5N2UyMmU5YWVmMzNlM2YwZiIsImNsaWVudF9pZCI6ImE5NGZiODU0N2ZmNmM3YWZiZTUwYjNmZWU2ZDE5MmJhIiwiaWF0IjoxNzcwODg0NjIwfQ.wENa7n7zr3jBgUvBR9D1_U-0XNPY_mS4-ILEx72f-uM"
LLM_API_KEY_ENV = "METACHAT_API_KEY"
LLM_TIMEOUT_S = 120
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 8000
LLM_EXTRA_HEADERS = {}



LLM_API_CONFIG = LLMApiConfig(
    enabled=RUN_LLM_BACKTEST,
    base_url=LLM_BASE_URL,
    endpoint=LLM_ENDPOINT,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    api_key_env=LLM_API_KEY_ENV,
    timeout_s=LLM_TIMEOUT_S,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
    extra_headers=LLM_EXTRA_HEADERS,
)


def main() -> None:
    if RUN_PREPROCESS:
        preprocessor = UncondDatasetPreprocessor(
            universe_csv="DataPrepare.csv",
            freq="MIX",
            lookback=6,
            horizon=2,
            out_dir=DATA_DIR,
        )
        preprocessor.run()

    if RUN_TFT_TRAINING:
        trainer = TFTTrainer(
            data_dir=DATA_DIR,
            out_dir=TFT_CKPT_DIR,
            epochs=100,
            batch_size=32,
            lr=1e-4,
            residual_theta=True,
            residual_scale=0.5,
            scale_loss=True,
        )
        trainer.run()

    if RUN_TFT_BACKTEST:
        backtester = RollingTFTBacktester(
            data_dir=DATA_DIR,
            ckpt_dir=TFT_CKPT_DIR,
            out_dir=RESULTS_TFT_DIR,
            mode="backtest",
            warmup=4,
            disable_interest_for_banks=True,
            tickers=TEST_COMPANIES,
        )
        backtester.run()

    if RUN_THEORY_BACKTEST:
        theory_runner = TheoryBacktestRunner(
            data_dir=DATA_DIR,
            out_dir=RESULTS_THEORY_DIR,
            mode="backtest",
            warmup=4,
            disable_interest_for_banks=True,
            tickers=TEST_COMPANIES,
        )
        theory_runner.run()

    if RUN_LLM_BACKTEST:
        llm_runner = LLMBacktestRunner(
            data_dir=DATA_DIR,
            out_dir=RESULTS_LLM_DIR,
            api=LLM_API_CONFIG,
            mode="backtest",
            warmup=4,
            min_ar1_points=3,
            save_one_file=True,
            disable_interest_for_banks=True,
            tickers=TEST_COMPANIES,
            prompt_history_window=6,
            save_raw_prompts=True,
            retry_on_invalid_json=1,
            duplicate_tft_filename=False,
        )
        llm_runner.run()

    if RUN_PLOT_THEORY_TFT:
        ComparisonPlotter(
            theory_dir=RESULTS_THEORY_DIR,
            tft_dir=RESULTS_TFT_DIR,
            out_dir=FIGURES_TFT_DIR,
            data_dir=DATA_DIR,
            tickers=TEST_COMPANIES,
            group="all",
            max_vars_per_page=10,
            max_xticks=12,
            png_dpi=300,
        ).run()

    if RUN_PLOT_THEORY_TFT_LLM:
        if not os.path.isdir(RESULTS_LLM_DIR):
            raise FileNotFoundError(
                f"{RESULTS_LLM_DIR} does not exist. Run the LLM backtest first or set RUN_LLM_BACKTEST=True."
            )
        ComparisonPlotter(
            theory_dir=RESULTS_THEORY_DIR,
            tft_dir=RESULTS_TFT_DIR,
            llm_dir=RESULTS_LLM_DIR,
            out_dir=FIGURES_ALL_DIR,
            data_dir=DATA_DIR,
            tickers=TEST_COMPANIES,
            group="all",
            max_vars_per_page=10,
            max_xticks=12,
            png_dpi=300,
            plot_llm_q10_q90=PLOT_LLM_Q10_Q90_IN_TRIPLE,
        ).run(mode="triple")

if __name__ == "__main__":
    main()
