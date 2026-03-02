# TFT Accounting Forecasting Package

A refactored Python package for accounting-aware financial statement forecasting.

<img width="2816" height="1536" alt="unwatermarked_Gemini_Generated_Image_4usqx24usqx24usq" src="https://github.com/user-attachments/assets/33d12122-0eb0-4c4a-8c9e-3e87649bcc59" />


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


### 1. Install the package

```bash
pip install -e .
```

`pip install -e .` installs the project in **editable mode**.

That means:
- ordinary source code changes inside `tft_accounting/` usually take effect immediately;
- you normally **do not need** to uninstall and reinstall after every Python code edit.

You should reinstall only when you change package metadata or dependencies, for example:
- `pyproject.toml`

If you do need a clean reinstall, use:

```bash
pip uninstall tft-accounting-refactor
pip install -e .
```

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

### run the example workflow

```bash
python examples/package_workflow.py
```

Before running it, edit the switches and paths in `examples/package_workflow.py`, especially:
- whether preprocessing, training, and each backtest stage should run;
- output directories;
- test company list;
- LLM API settings if you want to enable the LLM pipeline.

---

## Testing

Run the test suite with:

```bash
pip install -e .[test]
pytest
```

The provided tests cover:
- preprocessing artifacts,
- TensorFlow model outputs and constraints,
- one-epoch training and checkpoint saving,
- ticker selection and backtest output generation,
- plotting output generation.

If TensorFlow is not installed in the environment, TensorFlow-specific tests are skipped by design in the current test configuration.
