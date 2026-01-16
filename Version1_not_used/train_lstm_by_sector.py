"""
train_lstm_by_sector.py
-----------------------

读取 prepare_data_by_sector.py 生成的每个行业的 dataset：
    data_by_sector/<sector>_dataset.npz

对每个行业：
  1. 用训练样本的一个子集作为验证集（例如 train 内部 80/20 划分）。
  2. 计算标准化参数 (X_mean, X_std, y_mean, y_std)。
  3. 训练两个模型：
     - Unconstrained LSTM
     - Constraint-aware LSTM（通过 Lambda 层保证 Assets = Liab + Equity）
  4. 把训练好的模型和标准化参数存到 models_by_sector/<sector>/ 目录。
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATA_DIR = "data_by_sector"
MODEL_DIR = "models_by_sector"

FREQUENCY = "quarterly"   # 只用于记录 / 图标题时保持一致
SEQ_LEN = 2               # 必须和 prepare_data_by_sector.py 保持一致

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

Y_COLS = ["C", "AR", "Inv", "K", "AP", "D", "E"]
X_COLS = [
    "S", "COGS", "OPEX", "Dep", "I",
    "NewDebt", "Repay", "EquityIssues", "Dividends"
]

NUM_INDEPENDENT = 6  # C, AR, Inv, K, AP, D
NUM_TARGET = len(Y_COLS)


def sanitize_sector_name(sector: str) -> str:
    return sector.replace(" ", "_").replace("&", "and").replace("/", "-")


def build_unconstrained_lstm(input_dim, seq_len):
    inp = layers.Input(shape=(seq_len, input_dim), name="input_seq")
    x = layers.LSTM(64, activation="tanh", name="lstm")(inp)
    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    out = layers.Dense(NUM_TARGET, activation="linear", name="output")(x)
    model = models.Model(inputs=inp, outputs=out, name="UnconstrainedLSTM")
    return model


def build_constraint_aware_lstm(input_dim, seq_len):
    inp = layers.Input(shape=(seq_len, input_dim), name="input_seq")
    x = layers.LSTM(64, activation="tanh", name="lstm")(inp)
    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    ind = layers.Dense(NUM_INDEPENDENT, activation="linear", name="independent_raw")(x)

    def constraint_reconstruction(independent):
        C, AR, Inv, K, AP, D = tf.split(independent, 6, axis=-1)
        assets = C + AR + Inv + K
        liab = AP + D
        E = assets - liab
        full = tf.concat([C, AR, Inv, K, AP, D, E], axis=-1)
        return full

    out = layers.Lambda(constraint_reconstruction, name="constraint_layer")(ind)
    model = models.Model(inputs=inp, outputs=out, name="ConstraintAwareLSTM")
    return model


def train_for_sector(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    sector = str(data["sector"])

    sector_safe = sanitize_sector_name(sector)
    print("\n" + "=" * 80)
    print(f"Training models for sector: {sector} (file: {os.path.basename(npz_path)})")
    print("=" * 80)
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)

    input_dim = X_train.shape[2]

    # 1) 在训练集中再拆一个验证集 (80% / 20%)
    N = X_train.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)

    val_size = int(0.2 * N)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    print(f"  Train samples: {X_tr.shape[0]}")
    print(f"  Val   samples: {X_val.shape[0]}")

    # 2) 标准化：只用训练子集计算 mean/std
    X_mean = X_tr.mean(axis=(0, 1), keepdims=True)
    X_std = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8

    y_mean = y_tr.mean(axis=0, keepdims=True)
    y_std = y_tr.std(axis=0, keepdims=True) + 1e-8

    X_tr_n = (X_tr - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std
    y_tr_n = (y_tr - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    # 3) 训练 Unconstrained LSTM
    model_uc = build_unconstrained_lstm(input_dim, SEQ_LEN)
    model_uc.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"],
    )
    es_uc = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    print("\n[Sector:", sector, "] Training Unconstrained LSTM...")
    model_uc.fit(
        X_tr_n,
        y_tr_n,
        validation_data=(X_val_n, y_val_n),
        epochs=100,
        batch_size=32,
        callbacks=[es_uc],
        verbose=1,
    )

    # 4) 训练 Constraint-Aware LSTM
    model_ca = build_constraint_aware_lstm(input_dim, SEQ_LEN)
    model_ca.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"],
    )
    es_ca = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    print("\n[Sector:", sector, "] Training Constraint-Aware LSTM...")
    model_ca.fit(
        X_tr_n,
        y_tr_n,
        validation_data=(X_val_n, y_val_n),
        epochs=100,
        batch_size=32,
        callbacks=[es_ca],
        verbose=1,
    )

    # 5) 保存模型和标准化参数
    sector_dir = os.path.join(MODEL_DIR, sector_safe)
    os.makedirs(sector_dir, exist_ok=True)

    uc_path = os.path.join(sector_dir, "unconstrained_lstm.h5")
    ca_path = os.path.join(sector_dir, "constraint_aware_lstm.h5")
    scalers_path = os.path.join(sector_dir, "scalers.npz")

    model_uc.save(uc_path)
    model_ca.save(ca_path)
    np.savez(
        scalers_path,
        X_mean=X_mean,
        X_std=X_std,
        y_mean=y_mean,
        y_std=y_std,
        sector=np.array(sector),
    )

    print(f"\nSaved Unconstrained LSTM to {uc_path}")
    print(f"Saved Constraint-Aware LSTM to {ca_path}")
    print(f"Saved scalers to {scalers_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_dataset.npz")))

    if not npz_files:
        print("No dataset npz files found in", DATA_DIR)
        return

    for npz_path in npz_files:
        train_for_sector(npz_path)

    print("\nAll sectors trained.")


if __name__ == "__main__":
    main()
