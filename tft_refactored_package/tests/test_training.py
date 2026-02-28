from __future__ import annotations

import json

import numpy as np
import pytest


def _write_tiny_dataset(data_dir):
    target_specs = [
        {"name": "logS", "kind": "real", "lo": None, "hi": None},
        {"name": "margin", "kind": "bounded", "lo": 0.0, "hi": 1.0},
    ]
    meta = {
        "lookback": 2,
        "horizon": 1,
        "tickers": ["AAA", "BBB"],
        "sectors": ["Tech", "Energy"],
        "ticker_to_id": {"AAA": 0, "BBB": 1},
        "sector_to_id": {"Tech": 0, "Energy": 1},
        "hist_feat_cols": ["h1", "h2", "h3"],
        "fut_feat_cols": ["f1", "f2"],
        "target_specs": target_specs,
        "global_theta_medians": {"margin": 0.5},
    }
    (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    n_train = 4
    n_test = 2
    hist_feats = np.array([
        [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]],
        [[0.0, 0.1, 0.2], [0.3, 0.2, 0.1]],
        [[0.2, 0.0, 0.1], [0.1, 0.3, 0.2]],
        [[0.3, 0.2, 0.1], [0.0, 0.1, 0.2]],
    ], dtype=np.float32)
    future_feats = np.array([
        [[0.1, 0.2]],
        [[0.2, 0.1]],
        [[0.15, 0.25]],
        [[0.05, 0.10]],
    ], dtype=np.float32)
    y_true = np.array([
        [[4.60, 0.40]],
        [[4.70, 0.42]],
        [[4.65, 0.38]],
        [[4.80, 0.45]],
    ], dtype=np.float32)
    common = {
        "hist_mask": np.ones((n_train, 2), dtype=np.float32),
        "ticker_id": np.array([0, 1, 0, 1], dtype=np.int32),
        "sector_id": np.array([0, 1, 0, 1], dtype=np.int32),
        "size_log_ta": np.ones((n_train, 1), dtype=np.float32),
        "mask_y": np.ones((n_train, 1, 2), dtype=np.float32),
    }
    np.savez_compressed(data_dir / "train.npz", hist_feats=hist_feats, future_feats=future_feats, y_true=y_true, **common)
    np.savez_compressed(
        data_dir / "test.npz",
        hist_feats=hist_feats[:n_test],
        hist_mask=np.ones((n_test, 2), dtype=np.float32),
        future_feats=future_feats[:n_test],
        y_true=y_true[:n_test],
        mask_y=np.ones((n_test, 1, 2), dtype=np.float32),
        ticker_id=np.array([0, 1], dtype=np.int32),
        sector_id=np.array([0, 1], dtype=np.int32),
        size_log_ta=np.ones((n_test, 1), dtype=np.float32),
    )


def test_trainer_runs_one_epoch_and_saves_checkpoints(tmp_path):
    pytest.importorskip("tensorflow")
    from tft_accounting.training import TFTTrainer

    data_dir = tmp_path / "data_uncond"
    out_dir = tmp_path / "ckpt"
    data_dir.mkdir()
    _write_tiny_dataset(data_dir)

    trainer = TFTTrainer(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        epochs=1,
        batch_size=2,
        lr=1e-3,
        d_model=8,
        dropout=0.0,
        num_heads=1,
        ticker_emb_dim=4,
        sector_emb_dim=2,
        seed=7,
    )
    best_eval = trainer.run()

    assert np.isfinite(best_eval)
    assert (out_dir / "best.weights.h5").exists()
    assert (out_dir / "final.weights.h5").exists()
    assert (out_dir / "train_config.json").exists()
