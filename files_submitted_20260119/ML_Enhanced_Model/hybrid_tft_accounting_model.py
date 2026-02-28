# hybrid_tft_accounting_model.py
# -*- coding: utf-8 -*-
"""
Hybrid model:
  TFT-style learner -> predicts dynamic params θ_t (quantiles)
  Differentiable accounting layer -> maps to balance sheet outputs (quantiles)

Key engineering upgrades:
- Uses hist_mask to ignore padded history in LSTM and attention.
- Passes period_days_future into accounting layer for annual/quarterly/MIX.
"""

from __future__ import annotations
from typing import Dict, List
import tensorflow as tf

from differentiable_accounting import PARAM_SPECS, constrain_params, step_tf


def pinball_loss(y_true: tf.Tensor, y_pred: tf.Tensor, tau: float) -> tf.Tensor:
    e = y_true - y_pred
    return tf.maximum(tau * e, (tau - 1.0) * e)


def masked_quantile_loss(
    y_true: tf.Tensor,
    y_pred_q: tf.Tensor,
    mask_y: tf.Tensor,
    taus: List[float],
) -> tf.Tensor:
    """
    y_true: (B,H,Y)
    y_pred_q: (B,H,Y,Q)
    mask_y: (B,H,Y)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred_q = tf.cast(y_pred_q, tf.float32)
    mask_y = tf.cast(mask_y, tf.float32)

    losses = []
    for qi, tau in enumerate(taus):
        l = pinball_loss(y_true, y_pred_q[..., qi], tau)  # (B,H,Y)
        l = l * mask_y
        losses.append(tf.reduce_sum(l) / (tf.reduce_sum(mask_y) + 1e-8))
    return tf.add_n(losses) / float(len(taus))


class VariableSelection(tf.keras.layers.Layer):
    """Simplified TFT-style variable selection network."""

    def __init__(self, n_features: int, hidden: int = 64, name=None):
        super().__init__(name=name)
        self.d1 = tf.keras.layers.Dense(hidden, activation="elu")
        self.d2 = tf.keras.layers.Dense(n_features, activation=None)

    def call(self, x: tf.Tensor, context: tf.Tensor) -> tf.Tensor:
        # x: (B,T,F), context: (B,C) -> broadcast to (B,T,C)
        ctx = tf.expand_dims(context, axis=1)
        ctx = tf.repeat(ctx, repeats=tf.shape(x)[1], axis=1)
        h = tf.concat([x, ctx], axis=-1)
        w = tf.nn.softmax(self.d2(self.d1(h)), axis=-1)
        return x * w


class HybridTFTAccounting(tf.keras.Model):
    """
    Inputs (dict):
      hist_feats:        (B,L,Fh)
      hist_mask:         (B,L)  1 for real steps, 0 for padding
      future_feats:      (B,H,Ff) known future (time features, macro optional)
      y0:                (B,8) last observed state
      x_future:          (B,H,6) flows for solver (S,COGS,OPEX,EquityIssues,NI,Div) (NaN ok)
      period_days_future:(B,H,1) e.g. 365 (annual) or 365/4 (quarter)
      ticker_id:         (B,)
      sector_id:         (B,)
      size_log_ta:       (B,1)

    Output:
      y_pred_q: (B,H,10,Q=3) for [C,AR,Inv,K,AP,STD,LTD,TA,TL,E_implied]
    """

    def __init__(
        self,
        n_tickers: int,
        n_sectors: int,
        hist_feat_dim: int,
        fut_feat_dim: int,
        horizon: int = 2,
        taus: List[float] | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        lstm_units: int = 64,
        dropout: float = 0.1,
        name: str = "HybridTFTAccounting",
    ):
        super().__init__(name=name)
        self.horizon = int(horizon)
        self.taus = taus or [0.1, 0.5, 0.9]
        self.n_q = len(self.taus)
        self.n_params = len(PARAM_SPECS)

        # static embeddings
        self.ticker_emb = tf.keras.layers.Embedding(n_tickers, 16)
        self.sector_emb = tf.keras.layers.Embedding(n_sectors, 8)
        self.static_proj = tf.keras.layers.Dense(d_model, activation="elu")

        # variable selection
        self.vsn_hist = VariableSelection(hist_feat_dim, hidden=d_model)
        self.vsn_fut = VariableSelection(fut_feat_dim, hidden=d_model)

        # encoder/decoder
        self.encoder = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)

        # cross attention (decoder queries encoder)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=max(8, d_model // n_heads),
            dropout=dropout,
        )
        self.attn_proj = tf.keras.layers.Dense(lstm_units, activation=None)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)

        # output: raw_q50 + down/up -> build q10/q50/q90
        self.param_head = tf.keras.layers.Dense(self.n_params * 3, activation=None)

    def _context(self, ticker_id, sector_id, size_log_ta):
        t = self.ticker_emb(ticker_id)
        s = self.sector_emb(sector_id)
        ctx = tf.concat([t, s, tf.cast(size_log_ta, tf.float32)], axis=-1)
        return self.static_proj(ctx)

    def _theta_quantiles(self, hist_feats, future_feats, ctx, hist_mask_bool):
        # variable selection
        hsel = self.vsn_hist(hist_feats, ctx)
        fsel = self.vsn_fut(future_feats, ctx)

        # encoder with mask (ignore padded steps)
        enc_seq, h, c = self.encoder(hsel, mask=hist_mask_bool)

        # decoder
        dec_seq, _, _ = self.decoder(fsel, initial_state=[h, c])

        # attention mask: (B, Tq, Tk) bool
        # hist_mask_bool: (B, Tk)
        attn_mask = tf.expand_dims(hist_mask_bool, axis=1)
        attn_mask = tf.repeat(attn_mask, repeats=tf.shape(dec_seq)[1], axis=1)

        attn = self.cross_attn(query=dec_seq, value=enc_seq, key=enc_seq, attention_mask=attn_mask)
        attn = self.attn_proj(attn)

        z = self.norm(dec_seq + attn)
        z = self.drop(z)


        raw = self.param_head(z)  # (B,H,P*3)
        raw = tf.reshape(raw, [-1, self.horizon, self.n_params, 3])

        raw_q50 = raw[..., 0]
        raw_down = tf.nn.softplus(raw[..., 1])
        raw_up = tf.nn.softplus(raw[..., 2])

        raw_q10 = raw_q50 - raw_down
        raw_q90 = raw_q50 + raw_up

        theta_q10 = constrain_params(raw_q10)
        theta_q50 = constrain_params(raw_q50)
        theta_q90 = constrain_params(raw_q90)

        return tf.stack([theta_q10, theta_q50, theta_q90], axis=-1)  # (B,H,P,Q)

    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        hist_feats = inputs["hist_feats"]
        hist_mask = inputs["hist_mask"]  # (B,L)
        future_feats = inputs["future_feats"]
        y0 = inputs["y0"]
        x_future = inputs["x_future"]
        period_days_future = inputs["period_days_future"]  # (B,H,1)
        ticker_id = inputs["ticker_id"]
        sector_id = inputs["sector_id"]
        size_log_ta = inputs["size_log_ta"]

        hist_mask_bool = tf.cast(hist_mask > 0.5, tf.bool)

        ctx = self._context(ticker_id, sector_id, size_log_ta)
        theta_q = self._theta_quantiles(hist_feats, future_feats, ctx, hist_mask_bool)  # (B,H,P,Q)

        y_all = []
        for qi in range(self.n_q):
            y_prev = y0
            ys = []
            for k in range(self.horizon):
                pdays = period_days_future[:, k, 0]
                y_next = step_tf(y_prev, x_future[:, k, :], theta_q[:, k, :, qi], pdays)  # (B,10)
                ys.append(y_next)

                # next state's 8 dims
                C, AR, Inv, K, AP, STD, LTD, TA, TL, E = tf.unstack(y_next, axis=-1)
                y_prev = tf.stack([C, AR, Inv, K, AP, STD, LTD, E], axis=-1)

            y_all.append(tf.stack(ys, axis=1))  # (B,H,10)

        return tf.stack(y_all, axis=-1)  # (B,H,10,Q)
