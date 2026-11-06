"""Exact risk-neutral Black-Scholes transitions at the requested times."""

from collections.abc import Iterator
from math import isfinite
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contract import Market, Snowball


def normal_batches(
    n_paths: int,
    n_steps: int,
    *,
    seed: int,
    batch_size: int,
    antithetic: bool,
) -> Iterator[NDArray[np.float64]]:
    for name, value in (("n_paths", n_paths), ("n_steps", n_steps), ("batch_size", batch_size)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if antithetic and (n_paths % 2 or batch_size < 2):
        raise ValueError("antithetic simulation requires an even path count and batch_size >= 2")
    rng = np.random.default_rng(seed)
    if antithetic:
        pairs_per_batch = batch_size // 2
        for start in range(0, n_paths // 2, pairs_per_batch):
            z = rng.standard_normal((min(pairs_per_batch, n_paths // 2 - start), n_steps))
            batch = np.empty((2 * len(z), n_steps))
            batch[::2] = z
            batch[1::2] = -z
            yield batch
    else:
        for start in range(0, n_paths, batch_size):
            yield rng.standard_normal((min(batch_size, n_paths - start), n_steps))


def gbm_paths(spot: float, times: ArrayLike, market: Market, normals: ArrayLike) -> NDArray[np.float64]:
    """Map an array of independent shocks to prices, including the initial spot column."""
    times = np.asarray(times, dtype=float)
    z = np.asarray(normals, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("times must be a finite one-dimensional grid")
    dt = np.diff(times)
    if (
        not isfinite(spot)
        or spot <= 0
        or np.any(dt <= 0)
        or z.ndim != 2
        or z.shape[1] != len(dt)
        or not np.isfinite(z).all()
    ):
        raise ValueError("require a positive spot, increasing times and finite shocks of matching shape")

    paths = np.empty((len(z), len(times)))
    paths[:, 0] = spot
    log_spots = paths[:, 1:]
    with np.errstate(over="raise", under="raise", invalid="raise"):
        np.multiply(z, market.volatility * np.sqrt(dt), out=log_spots)
        log_spots += market.log_drift * dt
        np.cumsum(log_spots, axis=1, out=log_spots)
        log_spots += np.log(spot)
        np.exp(log_spots, out=log_spots)
    return paths


def simulate_paths(
    contract: Snowball,
    market: Market,
    n_paths: int,
    *,
    seed: int = 42,
    antithetic: bool = False,
) -> NDArray[np.float64]:
    """Generate a small issue-date path panel for inspection or hedge simulation."""
    normals = next(normal_batches(
        n_paths, contract.schedule.n_steps, seed=seed, batch_size=n_paths, antithetic=antithetic
    ))
    return gbm_paths(contract.initial_spot, contract.schedule.times, market, normals)
