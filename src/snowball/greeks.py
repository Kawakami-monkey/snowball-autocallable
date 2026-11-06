"""PDE finite differences and paired common-random-number estimates."""

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contract import Cashflow, Market, Snowball, includes_principal
from .monte_carlo import Estimate, estimate_samples
from .payoff import classify_payoffs, discount_cashflows
from .pde import PDESolution, solve_pde
from .simulation import gbm_paths, normal_batches

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")


@dataclass(frozen=True)
class GreekBumps:
    spot_fraction: float = 0.005
    volatility: float = 0.005
    rate: float = 0.001

    def __post_init__(self) -> None:
        if not all(isfinite(h) and h > 0 for h in (self.spot_fraction, self.volatility, self.rate)):
            raise ValueError("Greek bumps must be positive and finite")


@dataclass(frozen=True)
class GreekCurve:
    value: NDArray[np.float64]
    delta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    vega: NDArray[np.float64]
    theta: NDArray[np.float64]
    rho: NDArray[np.float64]


def spot_delta(
    solution: PDESolution, spot: ArrayLike, step: int, knocked_in: bool, bump: float
) -> NDArray[np.float64]:
    """Hedge shares per contract using an absolute spot bump and a frozen KI state."""
    spot = np.asarray(spot, dtype=float)
    return (
        solution.value(spot + bump, step=step, knocked_in=knocked_in)
        - solution.value(spot - bump, step=step, knocked_in=knocked_in)
    ) / (2 * bump)


def _bumped_markets(market: Market, bumps: GreekBumps) -> tuple[Market, Market, Market, Market]:
    if bumps.volatility >= market.volatility:
        raise ValueError("volatility bump must be smaller than the base volatility")
    return (
        replace(market, volatility=market.volatility + bumps.volatility),
        replace(market, volatility=market.volatility - bumps.volatility),
        replace(market, rate=market.rate + bumps.rate),
        replace(market, rate=market.rate - bumps.rate),
    )


class PDERisk:
    """Cache the four market bumps and reuse them across spot, date and KI state."""

    def __init__(self, solution: PDESolution, bumps: GreekBumps = GreekBumps()) -> None:
        self.solution = solution
        self.bumps = bumps
        self.vol_up, self.vol_down, self.rate_up, self.rate_down = (
            solve_pde(solution.contract, market, solution.grid, cashflow=solution.cashflow)
            for market in _bumped_markets(solution.market, bumps)
        )

    def evaluate(
        self, spot: ArrayLike, *, step: int = 0, knocked_in: bool = False
    ) -> GreekCurve:
        """Derivatives per unit input. Theta is calendar time, with dates fixed."""
        solution = self.solution
        if step >= solution.contract.schedule.n_steps:
            raise ValueError("Greeks are defined here only while the contract is live")
        spot = np.atleast_1d(np.asarray(spot, dtype=float))
        bump = self.bumps.spot_fraction * solution.contract.initial_spot
        state = {"step": step, "knocked_in": knocked_in}
        price = solution.value(spot, **state)
        up = solution.value(spot + bump, **state)
        down = solution.value(spot - bump, **state)
        delta = (up - down) / (2 * bump)
        gamma = (up - 2 * price + down) / bump**2
        vega = (self.vol_up.value(spot, **state) - self.vol_down.value(spot, **state)) / (
            2 * self.bumps.volatility
        )
        rho = (self.rate_up.value(spot, **state) - self.rate_down.value(spot, **state)) / (
            2 * self.bumps.rate
        )
        market = solution.market
        theta = (
            market.rate * price - (market.rate - market.dividend) * spot * delta
            - 0.5 * market.volatility**2 * spot**2 * gamma
        )
        return GreekCurve(price, delta, gamma, vega, theta, rho)


def monte_carlo_greeks(
    contract: Snowball,
    market: Market,
    *,
    n_paths: int = 120_000,
    seed: int = 123,
    batch_size: int = 8_192,
    antithetic: bool = True,
    spot: float | None = None,
    step: int = 0,
    knocked_in: bool = False,
    cashflow: Cashflow = "net",
    bumps: GreekBumps = GreekBumps(),
) -> dict[str, Estimate]:
    """Use the same shock matrix for all bumps, and estimate errors on pathwise differences."""
    includes_principal(cashflow)
    spot = contract.initial_spot if spot is None else spot
    ds = bumps.spot_fraction * contract.initial_spot
    if not isfinite(spot) or spot <= ds:
        raise ValueError("spot must be larger than the absolute spot bump")
    vol_up, vol_down, rate_up, rate_down = _bumped_markets(market, bumps)
    times = contract.schedule.remaining_times(step)
    roll = (times[1] - times[0]) / 4
    rolled_times = contract.schedule.remaining_times(step, roll)
    batches = []
    for z in normal_batches(
        n_paths, len(times) - 1, seed=seed, batch_size=batch_size, antithetic=antithetic
    ):
        # A positive roll shortens only the first interval. Original accrual and dates stay fixed.
        def value(current_spot: float, inputs: Market, dates: NDArray = times) -> NDArray:
            paths = gbm_paths(current_spot, dates, inputs, z)
            payoffs = classify_payoffs(contract, paths, step=step, knocked_in=knocked_in)
            return discount_cashflows(
                payoffs.cashflows(cashflow), payoffs.payment_time,
                inputs.rate, valuation_time=dates[0],
            )

        base = value(spot, market)
        up = value(spot + ds, market)
        down = value(spot - ds, market)
        batches.append(np.column_stack((
            base,
            (up - down) / (2 * ds),
            (up - 2 * base + down) / ds**2,
            (value(spot, vol_up) - value(spot, vol_down)) / (2 * bumps.volatility),
            (value(spot, market, rolled_times) - base) / roll,
            (value(spot, rate_up) - value(spot, rate_down)) / (2 * bumps.rate),
        )))
    return {
        name: estimate_samples(values, antithetic=antithetic)
        for name, values in zip(("value", *GREEK_NAMES), np.concatenate(batches).T)
    }
