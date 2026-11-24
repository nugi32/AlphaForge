"""Automatic look-ahead test (spec 6.5): features at rows <= cut must not change when the FUTURE is randomised."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd


def check_no_lookahead(build_fn: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame, cut: int, seed: int = 0, rtol: float = 1e-5) -> List[str]:
    """Return the names of features whose values at rows <= cut changed after perturbing every numeric column after ``cut``."""
    rng = np.random.default_rng(seed)
    base = build_fn(df)
    pert = df.copy()
    numeric = pert.select_dtypes(include=[np.number]).columns
    future = pert.index[cut + 1 :]
    for col in numeric:
        noise = rng.uniform(0.5, 1.5, size=len(future)).astype(pert[col].dtype)
        pert.loc[future, col] = pert.loc[future, col].to_numpy() * noise
    other = build_fn(pert)
    a = base.iloc[: cut + 1].to_numpy(np.float64)
    b = other.iloc[: cut + 1].to_numpy(np.float64)
    same = np.isclose(a, b, rtol=rtol, atol=1e-6, equal_nan=True).all(axis=0)
    return [c for c, ok in zip(base.columns, same) if not ok]
