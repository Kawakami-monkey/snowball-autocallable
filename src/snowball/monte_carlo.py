"""Batched pricing with uncertainty estimated from independent observations."""

from dataclasses import dataclass, fields
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .contract import Cashflow, Market, Snowball, includes_principal
from .payoff import Payoffs, classify_payoffs, discount_cashflows
from .simulation import gbm_paths, normal_batches

Z_95 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class Estimate:
    mean: float
    standard_error: float
    n_independent: int

    @property
    def confidence_interval(self) -> tuple[float, float]:
        radius = Z_95 * self.standard_error
        return self.mean - radius, self.mean + radius


def estimate_samples(values: ArrayLike, *, antithetic: bool = False) -> Estimate:
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("samples must be a finite vector")
    if antithetic:
        if len(samples) % 2:
            raise ValueError("antithetic samples must be in complete adjacent pairs")
        samples = samples.reshape(-1, 2).mean(axis=1)
    if len(samples) < 2:
        raise ValueError("standard errors require at least two independent observations")
    return Estimate(
        float(samples.mean()),
        float(samples.std(ddof=1) / np.sqrt(len(samples))),
        len(samples),
    )


@dataclass(frozen=True)
class MonteCarloResult:
    estimate: Estimate
    payoffs: Payoffs
    discounted: NDArray[np.float64]
    runtime_seconds: float
    antithetic: bool
    cashflow: Cashflow

    @property
    def price(self) -> float:
        return self.estimate.mean


def price_monte_carlo(
    contract: Snowball,
    market: Market,
    *,
    n_paths: int = 300_000,
    seed: int = 42,
    batch_size: int = 8_192,
    antithetic: bool = False,
    cashflow: Cashflow = "net",
    spot: float | None = None,
    step: int = 0,
    knocked_in: bool = False,
    elapsed: float = 0.0,
) -> MonteCarloResult:
    """Value an alive claim after the events at step. Keep outcomes, not the price paths."""
    includes_principal(cashflow)
    if n_paths < (4 if antithetic else 2):
        raise ValueError("pricing requires at least two independent draws or pairs")
    t0 = perf_counter()
    times = contract.schedule.remaining_times(step, elapsed)
    spot = contract.initial_spot if spot is None else spot
    batches = []
    for normals in normal_batches(
        n_paths, len(times) - 1, seed=seed, batch_size=batch_size, antithetic=antithetic
    ):
        paths = gbm_paths(spot, times, market, normals)
        batches.append(classify_payoffs(contract, paths, step=step, knocked_in=knocked_in))
        del paths
    payoffs = Payoffs(**{
        field.name: np.concatenate([getattr(batch, field.name) for batch in batches])
        for field in fields(Payoffs)
    })
    discounted = discount_cashflows(
        payoffs.cashflows(cashflow), payoffs.payment_time, market.rate, valuation_time=times[0]
    )
    estimate = estimate_samples(discounted, antithetic=antithetic)
    return MonteCarloResult(estimate, payoffs, discounted, perf_counter() - t0, antithetic, cashflow)
