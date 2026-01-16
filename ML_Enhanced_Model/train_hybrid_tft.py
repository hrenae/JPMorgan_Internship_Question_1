# train_hybrid_tft.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, os
import numpy as np
import tensorflow as tf

from hybrid_tft_accounting_model import masked_quantile_loss, HybridTFTAccounting


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def make_tf_dataset(split, batch_size: int, shuffle: bool):
    # pack targets + mask into one tensor for a custom loss
    y_pack = np.concatenate([split["y_true"], split["mask_y"]], axis=-1).astype(np.float32)

    inputs = {
        "hist_feats": split["hist_feats"].astype(np.float32),
        "hist_mask": split["hist_mask"].astype(np.float32),
        "future_feats": split["future_feats"].astype(np.float32),
        "y0": split["y0"].astype(np.float32),
        "x_future": split["x_future"].astype(np.float32),
        "period_days_future": split["period_days_future"].astype(np.float32),
        "ticker_id": split["ticker_id"].astype(np.int32).reshape([-1]),
        "sector_id": split["sector_id"].astype(np.int32).reshape([-1]),
        "size_log_ta": split["size_log_ta"].astype(np.float32),
    }

    ds = tf.data.Dataset.from_tensor_slices((inputs, y_pack))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(4096, len(y_pack)), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_hybrid")
    ap.add_argument("--model_dir", type=str, default="model_hybrid_tft")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json"), "r", encoding="utf-8"))
    horizon = int(meta["horizon"])

    train = load_npz(os.path.join(args.data_dir, "train.npz"))
    val_path = os.path.join(args.data_dir, "val.npz")
    val = load_npz(val_path) if os.path.exists(val_path) else None

    model = HybridTFTAccounting(
        n_tickers=len(meta["tickers"]),
        n_sectors=len(meta["sectors"]),
        hist_feat_dim=int(meta["hist_feat_dim"]),
        fut_feat_dim=int(meta["fut_feat_dim"]),
        horizon=horizon,
        taus=[0.1, 0.5, 0.9],
        d_model=64,
        n_heads=4,
        lstm_units=64,
        dropout=0.1,
    )

    def loss_fn(y_pack, y_pred_q):
        y_true = y_pack[..., :10]
        mask_y = y_pack[..., 10:]
        return masked_quantile_loss(y_true, y_pred_q, mask_y, taus=[0.1, 0.5, 0.9])

    def masked_mse_median(y_pack, y_pred_q):
        y_true = y_pack[..., :10]
        mask_y = y_pack[..., 10:]
        y_pred = y_pred_q[..., 1]  # p50
        se = tf.square(y_true - y_pred) * mask_y
        return tf.reduce_sum(se) / (tf.reduce_sum(mask_y) + 1e-8)

    ds_train = make_tf_dataset(train, args.batch_size, shuffle=True)
    ds_val = make_tf_dataset(val, args.batch_size, shuffle=False) if val is not None else None

    # --- NEW: build model variables explicitly before compile/fit ---
    sample_inputs, _ = next(iter(ds_train.take(1)))
    _ = model(sample_inputs, training=False)
    print("Built model. trainable_variables =", len(model.trainable_variables))


    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=loss_fn,
        metrics=[masked_mse_median],
    )

    # ds_train = make_tf_dataset(train, args.batch_size, shuffle=True)
    # ds_val = make_tf_dataset(val, args.batch_size, shuffle=False) if val is not None else None

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if ds_val is not None else "loss",
            patience=25,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.model_dir, "weights_best.weights.h5"),
            monitor="val_loss" if ds_val is not None else "loss",
            save_best_only=True,
            save_weights_only=True,
        ),

        tf.keras.callbacks.CSVLogger(os.path.join(args.model_dir, "train_log.csv")),
    ]

    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=callbacks, verbose=1)

    model.save_weights(os.path.join(args.model_dir, "weights_last.weights.h5"))
    with open(os.path.join(args.model_dir, "meta_train.json"), "w", encoding="utf-8") as f:
        json.dump({"data_dir": args.data_dir, "horizon": horizon}, f, indent=2, ensure_ascii=False)

    print(f"Saved model to: {args.model_dir}")


if __name__ == "__main__":
    main()
