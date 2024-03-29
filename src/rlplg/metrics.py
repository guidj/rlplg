"""
This module defines metrics functions.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.stats import mstats


def rmse(
    pred: np.ndarray, actual: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Root mean squared error.
    sum((y_pred - y_actual) ** 2) / n
    """
    if mask is None:
        mask = np.ones_like(pred)
    mask = mask.astype(bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    rmse_: np.ndarray = np.sqrt(np.mean(np.power(y_pred - y_actual, 2.0)))
    return rmse_


def rmsle(
    pred: np.ndarray,
    actual: np.ndarray,
    mask: Optional[np.ndarray] = None,
    translate: bool = False,
) -> np.ndarray:
    """
    Root mean squared logarithmic error.
    sum((log(y_pred + 1) - log(y_actual + 1)) ** 2) / n
    https://medium.com/analytics-vidhya/root-mean-square-log-error-rmse-vs-rmlse-935c6cc1802a

    `translate` is used to avoid negative logarithms
    """
    if mask is None:
        mask = np.ones_like(pred)
    mask = mask.astype(bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    if translate:
        lower_bound = np.min([y_pred, y_actual])
        y_pred = y_pred - lower_bound
        y_actual = y_actual - lower_bound

    rmsle_: np.ndarray = np.sqrt(
        np.mean(np.power(np.log1p(y_pred) - np.log1p(y_actual), 2.0))
    )
    return rmsle_


def mean_error(
    pred: np.ndarray, actual: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Mean error.
    sum(y_pred - y_actual) / 2
    """
    if mask is None:
        mask = np.ones_like(pred)
    mask = mask.astype(bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    mean_error_: np.ndarray = np.mean(y_pred - y_actual)
    return mean_error_


def pearson_correlation(
    pred: np.ndarray, actual: np.ndarray, mask: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Mean error.
    sum(y_pred - y_actual) / 2
    """
    if mask is None:
        mask = np.ones_like(pred)
    mask = mask.astype(bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    result: Tuple[float, float] = mstats.pearsonr(
        x=y_pred.flatten(), y=y_actual.flatten()
    )
    return result


def spearman_correlation(
    pred: np.ndarray, actual: np.ndarray, mask: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Mean error.
    sum(y_pred - y_actual) / 2
    """
    if mask is None:
        mask = np.ones_like(pred)
    mask = mask.astype(bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    result: Tuple[float, float] = mstats.spearmanr(
        x=y_pred.flatten(), y=y_actual.flatten()
    )
    return result
