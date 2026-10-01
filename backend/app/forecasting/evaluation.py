"""
Evaluation metrics used to score and compare forecasting models.

These are plain numpy computations -- never estimated or guessed by the LLM.
The agent's "LLM explanation" node is only ever shown the *output* of these
functions and asked to describe them in words; it never computes them itself.
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Mean Absolute Percentage Error. Returns None if y_true contains zeros
    (MAPE is undefined / explodes when the true value is 0)."""
    if np.any(y_true == 0):
        return None
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
