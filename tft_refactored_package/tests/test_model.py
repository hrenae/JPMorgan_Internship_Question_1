from __future__ import annotations

import numpy as np
import pytest


def test_uncond_tft_forward_shape_and_quantile_order():
    tf = pytest.importorskip("tensorflow")
    from tft_accounting.model import TargetSpec, UncondTFT

    target_specs = [
        TargetSpec("logS", "real"),
        TargetSpec("margin", "bounded", 0.0, 1.0),
    ]
    model = UncondTFT(
        hist_dim=3,
        fut_dim=2,
        n_tickers=3,
        n_sectors=2,
        target_specs=target_specs,
        d_model=8,
        dropout=0.0,
        num_heads=1,
        ticker_emb_dim=4,
        sector_emb_dim=2,
    )

    inputs = {
        "hist_feats": tf.zeros([4, 2, 3], dtype=tf.float32),
        "hist_mask": tf.ones([4, 2], dtype=tf.float32),
        "future_feats": tf.zeros([4, 1, 2], dtype=tf.float32),
        "ticker_id": tf.constant([0, 1, 2, 0], dtype=tf.int32),
        "sector_id": tf.constant([0, 1, 0, 1], dtype=tf.int32),
        "size_log_ta": tf.ones([4, 1], dtype=tf.float32),
    }

    yq = model(inputs, training=False).numpy()
    p50 = model.predict_p50(inputs).numpy()

    assert yq.shape == (4, 1, 2, 3)
    assert p50.shape == (4, 1, 2)
    assert np.all(yq[..., 0] <= yq[..., 1])
    assert np.all(yq[..., 1] <= yq[..., 2])
    assert np.all((0.0 <= yq[:, :, 1, :]) & (yq[:, :, 1, :] <= 1.0))
