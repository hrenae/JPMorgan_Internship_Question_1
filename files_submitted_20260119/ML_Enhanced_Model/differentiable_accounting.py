# differentiable_accounting.py
# -*- coding: utf-8 -*-
"""
Differentiable accounting (structural) layer.

Core idea:
  Learner predicts dynamic parameters theta_t (quantiles).
  This layer applies accounting logic to produce balance sheet states.
  Identity TA = TL + E_implied is enforced by construction.

This is a TensorFlow re-implementation of the logic style in your baseline
(working-capital turnover, capex/dep, financing to maintain cash buffer, etc.),
but with TF ops to enable backprop.
"""

from __future__ import annotations
from dataclasses import dataclass
import tensorflow as tf


@dataclass
class ParamSpec:
    name: str
    lo: float
    hi: float


# 14 parameters (can be extended later)
PARAM_SPECS = [
    ParamSpec("dso", 0.0, 720.0),               # days
    ParamSpec("dio", 0.0, 720.0),               # days
    ParamSpec("dpo", 0.0, 720.0),               # days
    ParamSpec("oca_to_sales", 0.0, 0.50),
    ParamSpec("onca_to_sales", 0.0, 1.00),
    ParamSpec("ocl_to_sales", 0.0, 0.50),
    ParamSpec("oncl_to_sales", 0.0, 1.00),
    ParamSpec("capex_to_sales", 0.0, 0.80),
    ParamSpec("dep_to_ppe", 0.0, 0.50),
    ParamSpec("cash_min_to_sales", 0.0, 0.50),
    ParamSpec("r_st", 0.0, 0.50),               # per-period rate
    ParamSpec("r_lt", 0.0, 0.50),               # per-period rate
    ParamSpec("tax_rate", 0.0, 0.50),
    ParamSpec("payout", 0.0, 1.00),
]


def _pos(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.relu(x)


def _finite_or(x: tf.Tensor, default: float) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    return tf.where(tf.math.is_finite(x), x, tf.cast(default, x.dtype))


def constrain_params(raw: tf.Tensor) -> tf.Tensor:
    """
    raw: (..., n_params) unconstrained -> bounded params via sigmoid scaling.
    """
    raw = tf.convert_to_tensor(raw, dtype=tf.float32)
    outs = []
    for j, spec in enumerate(PARAM_SPECS):
        u = tf.sigmoid(raw[..., j])
        p = spec.lo + (spec.hi - spec.lo) * u
        outs.append(p)
    return tf.stack(outs, axis=-1)


def step_tf(y_prev: tf.Tensor, x_t: tf.Tensor, theta: tf.Tensor, period_days: tf.Tensor) -> tf.Tensor:
    """
    One-step transition y_t = f(y_{t-1}, x_t; θ, period_days).

    y_prev: (..., 8)  [C, AR, Inv, K, AP, STD, LTD, E_implied]
    x_t:    (..., 6)  [S, COGS, OPEX, EquityIssues, NI_obs, Div_obs] (can be NaN)
    theta:  (..., 14) order == PARAM_SPECS
    period_days: (...) scalar, e.g. 365 for annual, 365/4 for quarterly

    output: (..., 10) [C,AR,Inv,K,AP,STD,LTD,TA,TL,E_implied]
    """
    y_prev = tf.convert_to_tensor(y_prev, tf.float32)
    x_t = tf.convert_to_tensor(x_t, tf.float32)
    theta = tf.convert_to_tensor(theta, tf.float32)
    period_days = tf.cast(period_days, tf.float32)

    C0, AR0, Inv0, K0, AP0, STD0, LTD0, E0 = tf.unstack(y_prev, axis=-1)
    S, COGS, OPEX, equity_issues, NI_obs, Div_obs = tf.unstack(x_t, axis=-1)

    # sanitize
    C0 = _finite_or(C0, 0.0); AR0 = _finite_or(AR0, 0.0); Inv0 = _finite_or(Inv0, 0.0)
    K0 = _finite_or(K0, 0.0); AP0 = _finite_or(AP0, 0.0); STD0 = _finite_or(STD0, 0.0)
    LTD0 = _finite_or(LTD0, 0.0); E0 = _finite_or(E0, 0.0)

    S = _finite_or(S, 0.0); COGS = _finite_or(COGS, 0.0); OPEX = _finite_or(OPEX, 0.0)
    equity_issues = _finite_or(equity_issues, 0.0)

    (dso, dio, dpo,
     oca_to_sales, onca_to_sales, ocl_to_sales, oncl_to_sales,
     capex_to_sales, dep_to_ppe, cash_min_to_sales,
     r_st, r_lt, tax_rate, payout) = tf.unstack(theta, axis=-1)

    # working capital from turnover days
    denom = tf.maximum(period_days, 1.0)
    AR = _pos(dso / denom * S)
    Inv = _pos(dio / denom * COGS)
    AP = _pos(dpo / denom * COGS)

    # residual categories scaled by sales
    OCA = _pos(oca_to_sales * S)
    ONCA = _pos(onca_to_sales * S)
    OCL = _pos(ocl_to_sales * S)
    ONCL = _pos(oncl_to_sales * S)

    # PPE dynamics
    dep = _pos(dep_to_ppe * K0)
    capex = _pos(capex_to_sales * S)
    K = _pos(K0 + capex - dep)

    # interest & earnings (rates are per-period)
    interest = _pos(r_st * STD0 + r_lt * LTD0)
    ebit = (S - COGS - OPEX) - dep
    taxable = _pos(ebit - interest)
    taxes = _pos(tax_rate * taxable)
    net_income = ebit - interest - taxes

    # CFO
    d_nwc = (AR - AR0) + (Inv - Inv0) - (AP - AP0)
    cfo = net_income + dep - d_nwc
    cfi = -capex

    # dividends policy (use observed if present)
    div_model = _pos(payout * _pos(net_income))
    dividends = tf.where(tf.math.is_finite(Div_obs), _pos(Div_obs), div_model)

    cash_pre = C0 + cfo + cfi + equity_issues - dividends

    # financing policy: maintain minimum cash buffer
    cash_min = _pos(cash_min_to_sales * S)
    borrow = _pos(cash_min - cash_pre)
    cash_after_borrow = cash_pre + borrow

    excess = _pos(cash_after_borrow - cash_min)
    repay_std = tf.minimum(STD0, excess)
    excess2 = excess - repay_std
    repay_ltd = tf.minimum(LTD0, excess2)

    STD = _pos(STD0 + borrow - repay_std)
    LTD = _pos(LTD0 - repay_ltd)
    C = _pos(cash_min + (excess2 - repay_ltd))

    # identities by construction
    TA = C + AR + Inv + K + OCA + ONCA
    TL = AP + OCL + ONCL + STD + LTD
    E_implied = TA - TL

    return tf.stack([C, AR, Inv, K, AP, STD, LTD, TA, TL, E_implied], axis=-1)
