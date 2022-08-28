from typing import Tuple

import numpy as np
from scipy.stats import mstats


def rmse(pred: np.ndarray, actual: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Root mean squared error.
    sum((y_pred - y_actual) ** 2) / n
    """
    if mask is None:
        mask = np.ones_like(pred)
    mask = mask.astype(np.bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    return np.math.sqrt(np.mean(np.power(y_pred - y_actual, 2.0)))


def rmsle(
    pred: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray = None,
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
    mask = mask.astype(np.bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    if translate:
        lower_bound = np.min([y_pred, y_actual])
        y_pred = y_pred - lower_bound
        y_actual = y_actual - lower_bound

    return np.math.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_actual), 2.0)))


def mean_error(
    pred: np.ndarray, actual: np.ndarray, mask: np.ndarray = None
) -> np.ndarray:
    """
    Mean error.
    sum(y_pred - y_actual) / 2
    """
    if mask is None:
        mask = np.ones_like(pred)
    mask = mask.astype(np.bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    return np.mean(y_pred - y_actual)


def pearson_correlation(
    pred: np.ndarray, actual: np.ndarray, mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean error.
    sum(y_pred - y_actual) / 2
    """
    if mask is None:
        mask = np.ones_like(pred)
    mask = mask.astype(np.bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    return mstats.pearsonr(x=y_pred.flatten(), y=y_actual.flatten())


def spearman_correlation(
    pred: np.ndarray, actual: np.ndarray, mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean error.
    sum(y_pred - y_actual) / 2
    """
    if mask is None:
        mask = np.ones_like(pred)
    mask = mask.astype(np.bool)
    y_pred = pred[mask].astype(np.float32)
    y_actual = actual[mask].astype(np.float32)
    return mstats.spearmanr(x=y_pred.flatten(), y=y_actual.flatten())
