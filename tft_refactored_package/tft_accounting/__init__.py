"""Refactored TensorFlow accounting forecasting package.

The package exposes a package-first API while keeping imports lightweight.
Heavy TensorFlow modules are loaded lazily so plotting utilities and tests that
only need non-TensorFlow functionality can import submodules without pulling in
all training dependencies at package import time.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "TargetSpec",
    "UncondTFT",
    "masked_quantile_loss",
    "make_dataset",
    "TFTTrainer",
    "TrainingConfig",
    "UncondDatasetPreprocessor",
    "PreprocessConfig",
    "RollingTFTBacktester",
    "BacktestConfig",
    "TheoryBacktestRunner",
    "TheoryConfig",
    "ComparisonPlotter",
    "PlotConfig",
    # LLM-based backtesting
    "LLMApiConfig",
    "LLMBacktestConfig",
    "LLMBacktestRunner",
]

_EXPORT_MAP = {
    "TargetSpec": ("tft_accounting.model", "TargetSpec"),
    "UncondTFT": ("tft_accounting.model", "UncondTFT"),
    "masked_quantile_loss": ("tft_accounting.model", "masked_quantile_loss"),
    "make_dataset": ("tft_accounting.model", "make_dataset"),
    "TFTTrainer": ("tft_accounting.training", "TFTTrainer"),
    "TrainingConfig": ("tft_accounting.training", "TrainingConfig"),
    "UncondDatasetPreprocessor": ("tft_accounting.preprocessing", "UncondDatasetPreprocessor"),
    "PreprocessConfig": ("tft_accounting.preprocessing", "PreprocessConfig"),
    "RollingTFTBacktester": ("tft_accounting.backtesting", "RollingTFTBacktester"),
    "BacktestConfig": ("tft_accounting.backtesting", "BacktestConfig"),
    "TheoryBacktestRunner": ("tft_accounting.theory", "TheoryBacktestRunner"),
    "TheoryConfig": ("tft_accounting.theory", "TheoryConfig"),
    "ComparisonPlotter": ("tft_accounting.plotting", "ComparisonPlotter"),
    "PlotConfig": ("tft_accounting.plotting", "PlotConfig"),
    "LLMApiConfig": ("tft_accounting.LLM_based", "LLMApiConfig"),
    "LLMBacktestConfig": ("tft_accounting.LLM_based", "LLMBacktestConfig"),
    "LLMBacktestRunner": ("tft_accounting.LLM_based", "LLMBacktestRunner"),
}


if TYPE_CHECKING:  # pragma: no cover
    from .backtesting import BacktestConfig, RollingTFTBacktester
    from .model import TargetSpec, UncondTFT, make_dataset, masked_quantile_loss
    from .plotting import ComparisonPlotter, PlotConfig
    from .preprocessing import PreprocessConfig, UncondDatasetPreprocessor
    from .theory import TheoryBacktestRunner, TheoryConfig
    from .training import TFTTrainer, TrainingConfig
    from .LLM_based import LLMApiConfig, LLMBacktestConfig, LLMBacktestRunner


def __getattr__(name: str):
    """Lazily import public package symbols on first access."""
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'tft_accounting' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
