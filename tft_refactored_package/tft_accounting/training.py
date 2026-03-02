"""Training utilities for the unconditional TFT model.

This module keeps the original optimization logic, but exposes it through
classes so the project can be delivered as a structured Python package.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from .model import DatasetBuilder, TargetSpec, UncondTFT, masked_quantile_loss


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    data_dir: str
    out_dir: str = "tft_uncond_ckpt"
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    residual_theta: bool = False
    residual_scale: float = 1.0
    scale_loss: bool = False
    real_scale_mode: str = "std"
    d_model: int = 64
    dropout: float = 0.10
    num_heads: int = 4
    ticker_emb_dim: int = 16
    sector_emb_dim: int = 8
    seed: int = 42


class NpzRepository:
    """Helper for loading serialized training arrays."""

    @staticmethod
    def load_npz(path: str) -> Dict[str, np.ndarray]:
        """Load one NPZ file into an in-memory dictionary."""
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}


class TargetSpecFactory:
    """Factory methods for target specifications and sector baselines."""

    @staticmethod
    def build_target_specs(meta: Dict[str, object]) -> List[TargetSpec]:
        return [TargetSpec(**spec_dict) for spec_dict in meta["target_specs"]]

    @staticmethod
    def inverse_constraint_to_z(y: float, spec: TargetSpec, eps: float = 1e-6) -> float:
        """Invert the target constraint approximately in raw z-space."""
        if spec.kind == "real":
            return float(y)
        lo = 0.0 if spec.lo is None else float(spec.lo)
        hi = 0.0 if spec.hi is None else float(spec.hi)
        rng = hi - lo
        if rng <= 0:
            return 0.0
        if spec.kind == "bounded":
            p = (float(y) - lo) / rng
            p = min(1.0 - eps, max(eps, p))
            return float(np.log(p / (1.0 - p)))
        if spec.kind == "signed":
            mid = 0.5 * (hi + lo)
            scale = 0.5 * (hi - lo)
            if scale <= 0:
                return 0.0
            u = (float(y) - mid) / scale
            u = min(1.0 - eps, max(-1.0 + eps, u))
            return float(0.5 * np.log((1.0 + u) / (1.0 - u)))
        return float(y)

    @classmethod
    def build_base_z_by_sector(cls, meta: Dict[str, object], target_specs: List[TargetSpec]) -> tf.Tensor:
        """Build the sector baseline tensor used by residual-theta training."""
        med_path = os.path.join(str(meta["data_dir"]), "sector_theta_medians.json")
        if not os.path.exists(med_path):
            return tf.zeros([len(meta["sectors"]), len(target_specs)], dtype=tf.float32)
        with open(med_path, "r", encoding="utf-8") as file:
            sector_medians = json.load(file)
        global_medians = meta.get("global_theta_medians", {})

        target_dim = len(target_specs)
        n_sectors = len(meta["sectors"])
        base_z = np.zeros((n_sectors, target_dim), dtype=np.float32)
        sector_to_id = meta.get("sector_to_id", {})

        for sec_name, sec_dict in sector_medians.items():
            sector_id = int(sector_to_id.get(sec_name, -1))
            if sector_id < 0 or sector_id >= n_sectors:
                continue
            for j, spec in enumerate(target_specs):
                name = spec.name
                if str(name).lower() in ("logs", "log_s"):
                    continue
                yb = sec_dict.get(name, None)
                if yb is None or (isinstance(yb, float) and not np.isfinite(yb)):
                    yb = global_medians.get(name, None)
                if yb is None:
                    if spec.kind == "bounded":
                        lo = 0.0 if spec.lo is None else float(spec.lo)
                        hi = 0.0 if spec.hi is None else float(spec.hi)
                        yb = 0.5 * (lo + hi)
                    else:
                        yb = 0.0
                base_z[sector_id, j] = cls.inverse_constraint_to_z(float(yb), spec)
        return tf.constant(base_z, dtype=tf.float32)


class LossScaleBuilder:
    """Compute the optional per-target loss scaling vector."""

    def __init__(self, target_specs: List[TargetSpec], config: TrainingConfig) -> None:
        self.target_specs = target_specs
        self.config = config

    def build(self, train_npz: Dict[str, np.ndarray]) -> tf.Variable:
        """Return the loss scale vector as a non-trainable ``tf.Variable``."""
        kinds = [spec.kind for spec in self.target_specs]
        lo = np.array([0.0 if spec.lo is None else float(spec.lo) for spec in self.target_specs], dtype=np.float32)
        hi = np.array([0.0 if spec.hi is None else float(spec.hi) for spec in self.target_specs], dtype=np.float32)
        ranges = np.maximum(hi - lo, 1e-6).astype(np.float32)

        scales = np.ones(len(self.target_specs), dtype=np.float32)
        if self.config.scale_loss:
            y_train = train_npz["y_true"].astype(np.float32)
            m_train = train_npz["mask_y"].astype(np.float32)
            for j in range(len(self.target_specs)):
                if kinds[j] in ("bounded", "signed"):
                    scales[j] = float(ranges[j])
                elif self.config.real_scale_mode == "std":
                    values = y_train[:, :, j][m_train[:, :, j] > 0.0]
                    std = float(np.std(values)) if values.size > 0 else 1.0
                    scales[j] = max(std, 1e-3)
                else:
                    scales[j] = 1.0
        return tf.Variable(scales, trainable=False, dtype=tf.float32, name="loss_scale_d")


class TFTTrainer:
    """Encapsulates end-to-end training of the unconditional TFT model.

    Parameters
    ----------
    config:
        Optional :class:`TrainingConfig` instance.
    **kwargs:
        Keyword arguments used to build :class:`TrainingConfig` directly.
    """

    def __init__(self, config: Optional[TrainingConfig] = None, **kwargs) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either 'config' or keyword arguments, not both.")
        self.config = config if config is not None else TrainingConfig(**kwargs)
        self.meta: Dict[str, object] = {}
        self.target_specs: List[TargetSpec] = []
        self.model: Optional[UncondTFT] = None
        self.optimizer: Optional[tf.keras.optimizers.Optimizer] = None
        self.scale_d: Optional[tf.Variable] = None
        self.eval_name: str = "val"

    def load_metadata(self) -> None:
        """Read ``meta.json`` and initialize target specifications."""
        with open(os.path.join(self.config.data_dir, "meta.json"), "r", encoding="utf-8") as file:
            self.meta = json.load(file)
        self.meta["data_dir"] = self.config.data_dir
        self.target_specs = TargetSpecFactory.build_target_specs(self.meta)

    def build_model(self) -> UncondTFT:
        """Construct and build the TensorFlow model once."""
        base_z_by_sector = None
        if self.config.residual_theta:
            base_z_by_sector = TargetSpecFactory.build_base_z_by_sector(self.meta, self.target_specs)

        model = UncondTFT(
            hist_dim=self.meta["hist_feat_dim"] if "hist_feat_dim" in self.meta else len(self.meta["hist_feat_cols"]),
            fut_dim=self.meta["fut_feat_dim"] if "fut_feat_dim" in self.meta else len(self.meta["fut_feat_cols"]),
            n_tickers=len(self.meta["tickers"]),
            n_sectors=len(self.meta["sectors"]),
            target_specs=self.target_specs,
            d_model=self.config.d_model,
            dropout=self.config.dropout,
            num_heads=self.config.num_heads,
            ticker_emb_dim=self.config.ticker_emb_dim,
            sector_emb_dim=self.config.sector_emb_dim,
            residual_theta=self.config.residual_theta,
            residual_scale=self.config.residual_scale,
            base_z_by_sector=base_z_by_sector,
        )
        dummy_x = {
            "hist_feats": tf.zeros([2, self.meta["lookback"], len(self.meta["hist_feat_cols"])], tf.float32),
            "hist_mask": tf.ones([2, self.meta["lookback"]], tf.float32),
            "future_feats": tf.zeros([2, self.meta["horizon"], len(self.meta["fut_feat_cols"])], tf.float32),
            "ticker_id": tf.zeros([2], tf.int32),
            "sector_id": tf.zeros([2], tf.int32),
            "size_log_ta": tf.zeros([2, 1], tf.float32),
        }
        _ = model(dummy_x, training=False)
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.lr)
        return model

    def load_datasets(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load train, validation, and evaluation NPZ files."""
        train_npz = NpzRepository.load_npz(os.path.join(self.config.data_dir, "train.npz"))
        val_path = os.path.join(self.config.data_dir, "val.npz")
        if os.path.exists(val_path):
            val_npz = NpzRepository.load_npz(val_path)
            if "y_true" in val_npz and len(val_npz["y_true"]) > 0:
                for key in list(train_npz.keys()):
                    if key in val_npz and getattr(val_npz[key], "size", 0) > 0:
                        train_npz[key] = np.concatenate([train_npz[key], val_npz[key]], axis=0)

        test_path = os.path.join(self.config.data_dir, "test.npz")
        if os.path.exists(test_path):
            self.eval_name = "test"
            eval_npz = NpzRepository.load_npz(test_path)
        else:
            self.eval_name = "val"
            eval_npz = NpzRepository.load_npz(val_path) if os.path.exists(val_path) else train_npz
        return {"train": train_npz, "eval": eval_npz}

    @tf.function
    def train_step(self, x: Dict[str, tf.Tensor], y: tf.Tensor, m: tf.Tensor) -> tf.Tensor:
        """One gradient update step."""
        assert self.model is not None
        assert self.optimizer is not None
        assert self.scale_d is not None
        with tf.GradientTape() as tape:
            yq = self.model(x, training=True)
            loss = masked_quantile_loss(y, yq, m, scale_d=self.scale_d)
            loss += 1e-6 * tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def val_step(self, x: Dict[str, tf.Tensor], y: tf.Tensor, m: tf.Tensor) -> tf.Tensor:
        """One evaluation step without gradient updates."""
        assert self.model is not None
        assert self.scale_d is not None
        yq = self.model(x, training=False)
        return masked_quantile_loss(y, yq, m, scale_d=self.scale_d)

    def train(self) -> float:
        """Run full training and return the best evaluation loss."""
        os.makedirs(self.config.out_dir, exist_ok=True)
        tf.random.set_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.load_metadata()
        datasets = self.load_datasets()
        self.build_model()
        self.scale_d = LossScaleBuilder(self.target_specs, self.config).build(datasets["train"])

        ds_train = DatasetBuilder.make_dataset(
            datasets["train"], batch_size=self.config.batch_size, shuffle=True, seed=self.config.seed
        )
        ds_eval = DatasetBuilder.make_dataset(
            datasets["eval"], batch_size=self.config.batch_size, shuffle=False, seed=self.config.seed
        )

        best_eval = float("inf")
        best_path = os.path.join(self.config.out_dir, "best.weights.h5")

        for epoch in range(1, self.config.epochs + 1):
            tr_losses: List[float] = []
            for x, y, m in ds_train:
                tr_losses.append(float(self.train_step(x, y, m).numpy()))
            tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")

            ev_losses: List[float] = []
            for x, y, m in ds_eval:
                ev_losses.append(float(self.val_step(x, y, m).numpy()))
            ev_loss = float(np.mean(ev_losses)) if ev_losses else float("nan")

            print(f"epoch {epoch:03d} | train {tr_loss:.6f} | {self.eval_name} {ev_loss:.6f}")
            if np.isfinite(ev_loss) and ev_loss < best_eval:
                best_eval = ev_loss
                assert self.model is not None
                self.model.save_weights(best_path)
                with open(os.path.join(self.config.out_dir, f"best_{self.eval_name}.txt"), "w", encoding="utf-8") as file:
                    file.write(f"{best_eval}\n")

        assert self.model is not None
        self.model.save_weights(os.path.join(self.config.out_dir, "final.weights.h5"))

        cfg = asdict(self.config)
        cfg["best_eval"] = best_eval
        cfg["eval_split"] = self.eval_name
        with open(os.path.join(self.config.out_dir, "train_config.json"), "w", encoding="utf-8") as file:
            json.dump(cfg, file, indent=2)
        print(f"[done] best_{self.eval_name}={best_eval:.6f} | saved to {self.config.out_dir}")
        return best_eval

    def run(self) -> float:
        """Alias for :meth:`train` to support package-style orchestration."""
        return self.train()


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="tft_uncond_ckpt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--residual_theta", action="store_true", help="Residual learning for theta in raw z-space.")
    parser.add_argument("--residual_scale", type=float, default=1.0, help="Scale for residual deltas in z-space.")
    parser.add_argument("--scale_loss", action="store_true", help="Scale-normalize pinball loss per target dimension.")
    parser.add_argument("--real_scale_mode", type=str, default="std", choices=["std", "1.0"])
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ticker_emb_dim", type=int, default=16)
    parser.add_argument("--sector_emb_dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Entry point preserving the original CLI."""
    args = build_arg_parser().parse_args()
    trainer = TFTTrainer(TrainingConfig(**vars(args)))
    trainer.train()


__all__ = [
    "TrainingConfig",
    "NpzRepository",
    "TargetSpecFactory",
    "LossScaleBuilder",
    "TFTTrainer",
    "build_arg_parser",
    "main",
]
