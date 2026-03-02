# TFT Accounting Forecasting Package

A refactored Python package for accounting-aware financial statement forecasting.

This repository reorganizes the original internship codebase into a package-oriented, object-oriented structure and uses TensorFlow for the core model, training, and inference components. The project supports three complementary forecasting pipelines:

1. **Theory baseline**: AR(1)-style sales forecasting plus accounting-aware simulation.
2. **TensorFlow TFT model**: an unconditional Temporal Fusion Transformer style model for target forecasting and rolling backtesting.
3. **LLM-based forecasting**: an optional OpenAI-compatible pipeline that generates target forecasts from prompts and compares them against the theory and TFT pipelines.

The code is designed for:
- reusable package imports;
- command-line execution for each stage;
- rolling backtesting and multi-company comparison;
- automated testing with `pytest`.

---

## Repository layout

```text
tft_refactored_package/
├── pyproject.toml
├── requirements.txt
├── README.md
├── examples/
│   └── package_workflow.py
├── tft_accounting/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── training.py
│   ├── backtesting.py
│   ├── theory.py
│   ├── LLM_based.py
│   └── plotting.py
└── tests/
```

---

## What each module does

### `tft_accounting/__init__.py`
Package entry point. Exposes the public API and uses lazy imports so heavy TensorFlow modules are loaded only when needed.

### `tft_accounting/prepare_data.py`
Lower-level data preparation utilities based on `yfinance`.
It extracts annual balance sheet, income statement, and cash flow information and builds aligned period and transition tables.

### `tft_accounting/preprocessing.py`
Main preprocessing pipeline for the unconditional forecasting setup.
It:
- downloads and assembles raw panels,
- engineers features and labels,
- builds train/validation/test arrays,
- writes `meta.json`, `*.npz`, panel CSV files, and sector-level baseline statistics.

### `tft_accounting/model.py`
TensorFlow model definition.
It contains:
- custom TensorFlow layers and helper modules,
- target constraint handling,
- the `UncondTFT` model,
- dataset utilities and masked quantile loss.

### `tft_accounting/training.py`
Training utilities for the TensorFlow model.
It loads the serialized dataset, builds the TFT model, runs optimization, and saves checkpoints and training configuration.

### `tft_accounting/backtesting.py`
Rolling one-step backtesting for the trained TFT model.
It loads model weights, performs one-step rolling forecasts, and saves per-ticker and combined backtest CSV files.

### `tft_accounting/theory.py`
Accounting-aware theory baseline.
It fits a simple sales dynamics model, combines it with sector median policy variables, and simulates next-period statements under the accounting constraints used in the report.

### `tft_accounting/LLM_based.py`
Optional LLM-based rolling backtesting.
It queries an OpenAI-compatible endpoint, parses and sanitizes forecast outputs, then produces backtest CSV files in a format comparable to theory and TFT outputs.

### `tft_accounting/plotting.py`
Comparison plotting utilities.
It generates per-ticker PDF and SVG comparison figures for:
- theory vs TFT, or
- theory vs TFT vs LLM.

### `examples/package_workflow.py`
A single example script that runs the end-to-end workflow using package imports instead of ad hoc scripts.
This is the easiest place to see the intended pipeline.

### `tests/`
Automated `pytest` test suite for preprocessing, model behavior, training, backtesting, and plotting.

---

## Environment setup

### 1. Create and activate a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the project in **editable mode**.

That means:
- ordinary source code changes inside `tft_accounting/` usually take effect immediately;
- you normally **do not need** to uninstall and reinstall after every Python code edit.

You should reinstall only when you change package metadata or dependencies, for example:
- `pyproject.toml`
- `requirements.txt`
- console entry points or build configuration

If you do need a clean reinstall, use:

```bash
pip uninstall -y tft-accounting
pip install -r requirements.txt
pip install -e .
```

If the package name in your local environment differs from `tft-accounting`, uninstall using the installed distribution name shown by `pip list`.

---

## Input data requirements

The main workflow expects a universe file such as `DataPrepare.csv` with ticker and sector information.

Typical columns are:
- `ticker`
- `sector`
- a train/test indicator column used by your preprocessing logic

The preprocessing pipeline then downloads statement data from `yfinance` and builds the package-specific dataset files.

---

## Recommended workflow

There are two convenient ways to run the project.

### Option A: run the example workflow

```bash
python examples/package_workflow.py
```

Before running it, edit the switches and paths in `examples/package_workflow.py`, especially:
- whether preprocessing, training, and each backtest stage should run;
- output directories;
- test company list;
- LLM API settings if you want to enable the LLM pipeline.

### Option B: run each stage separately

#### Step 1: preprocess data

```bash
python -m tft_accounting.preprocessing \
  --universe_csv DataPrepare.csv \
  --freq MIX \
  --lookback 6 \
  --horizon 2 \
  --out_dir data_uncond
```

Typical outputs:
- `data_uncond/meta.json`
- `data_uncond/train.npz`
- `data_uncond/val.npz`
- `data_uncond/test.npz`
- `data_uncond/sector_theta_medians.json`
- `data_uncond/panels/*.csv`

#### Step 2: train the TensorFlow TFT model

```bash
python -m tft_accounting.training \
  --data_dir data_uncond \
  --out_dir tft_uncond_ckpt \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --residual_theta \
  --residual_scale 0.5 \
  --scale_loss
```

Typical outputs:
- `tft_uncond_ckpt/best.weights.h5`
- `tft_uncond_ckpt/final.weights.h5`
- `tft_uncond_ckpt/train_config.json`

#### Step 3: run the theory baseline backtest

```bash
python -m tft_accounting.theory \
  --data_dir data_uncond \
  --out_dir results_theory \
  --mode backtest \
  --warmup 4 \
  --disable_interest_for_banks \
  --tickers MSFT,GOOG,JPM,VWAGY,XOM
```

Typical outputs:
- `results_theory/<TICKER>_theory_backtest.csv`
- optionally `results_theory/backtest_all.csv`

#### Step 4: run the TFT rolling backtest

```bash
python -m tft_accounting.backtesting \
  --data_dir data_uncond \
  --ckpt_dir tft_uncond_ckpt \
  --out_dir results_tft \
  --mode backtest \
  --warmup 4 \
  --disable_interest_for_banks \
  --tickers MSFT,GOOG,JPM,VWAGY,XOM
```

Typical outputs:
- `results_tft/<TICKER>_tft_backtest.csv`
- optionally `results_tft/backtest_all.csv`

#### Step 5: optional LLM rolling backtest

Set your API key first, for example:

Linux/macOS:

```bash
export OPENAI_API_KEY="your_key_here"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Then run:

```bash
python -m tft_accounting.LLM_based \
  --data_dir data_uncond \
  --out_dir result_llm \
  --base_url https://api.openai.com/v1 \
  --endpoint /chat/completions \
  --model gpt-5.2 \
  --api_key_env OPENAI_API_KEY \
  --warmup 4 \
  --min_ar1_points 3 \
  --save_one_file \
  --disable_interest_for_banks \
  --tickers MSFT,GOOG,JPM,VWAGY,XOM \
  --prompt_history_window 6 \
  --retry_on_invalid_json 1
```

Typical outputs:
- `result_llm/<TICKER>_llm_backtest.csv`
- optional prompt / response / summary artifacts, depending on configuration

#### Step 6: generate comparison figures

**Theory vs TFT**

```bash
python -m tft_accounting.plotting \
  --theory_dir results_theory \
  --tft_dir results_tft \
  --out_dir figures_compare \
  --data_dir data_uncond \
  --tickers MSFT,GOOG,JPM,VWAGY,XOM \
  --group all \
  --mode double
```

**Theory vs TFT vs LLM**

```bash
python -m tft_accounting.plotting \
  --theory_dir results_theory \
  --tft_dir results_tft \
  --llm_dir result_llm \
  --out_dir figures_compare_all \
  --data_dir data_uncond \
  --tickers MSFT,GOOG,JPM,VWAGY,XOM \
  --group all \
  --mode triple
```

Typical outputs:
- per-ticker comparison PDFs
- per-variable SVG figure files

---

## Testing

Run the test suite with:

```bash
pytest -q
```

The provided tests cover:
- preprocessing artifacts,
- TensorFlow model outputs and constraints,
- one-epoch training and checkpoint saving,
- ticker selection and backtest output generation,
- plotting output generation.

If TensorFlow is not installed in the environment, TensorFlow-specific tests are skipped by design in the current test configuration.

---

## Notes on editable installs and updates

### If you only changed Python source files
For example:
- `tft_accounting/model.py`
- `tft_accounting/training.py`
- `tft_accounting/theory.py`

then editable mode usually means you can just rerun your command:

```bash
python -m tft_accounting.training --data_dir data_uncond
```

### If you changed dependencies or packaging metadata
For example:
- added a new package to `requirements.txt`
- changed `pyproject.toml`
- changed packaging layout

then do a reinstall:

```bash
pip uninstall -y tft-accounting
pip install -r requirements.txt
pip install -e .
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'tft_accounting'`
Make sure you are running commands from the repository root and that you already executed:

```bash
pip install -e .
```

### TensorFlow import or GPU issues
This repository can be used on CPU. If GPU support is not configured in your environment, install the CPU-compatible TensorFlow package specified by your dependency setup.

### No data generated during preprocessing
Check:
- internet access for `yfinance`;
- ticker validity in `DataPrepare.csv`;
- write permissions for the output directory.

### LLM backtest fails
Check:
- API key environment variable;
- base URL and endpoint format;
- model name;
- proxy/network restrictions.

---

## Important project notes

- NumPy and pandas are still used in data downloading, preprocessing, result assembly, and plotting.
- TensorFlow is used for the core model, training, and relevant inference-side components.
- The LLM pipeline is **optional** and intended as an additional comparison baseline rather than a replacement for the main theory and TFT pipelines.

---

## Citation / submission note

If you are using this repository for the internship resubmission, the key deliverables are:
- package-oriented OOP refactor;
- TensorFlow-based core modeling and training;
- automated `pytest` tests;
- accounting-aware theory baseline;
- TFT rolling backtesting;
- optional LLM-based forecasting and comparison plots.

