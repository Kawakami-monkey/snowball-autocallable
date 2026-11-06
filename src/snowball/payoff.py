"""Vectorized cash-flow rules, independent of the path generator and pricing model."""

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .contract import Cashflow, Snowball, includes_principal


class Scenario(IntEnum):
    KO_WITHOUT_KI = 1
    NO_KI_NO_KO = 2
    KI_NO_KO_RECOVERED = 3
    KI_NO_KO_LOSS = 4
    KO_AFTER_KI = 5


@dataclass(frozen=True)
class Payoffs:
    scenario: NDArray[np.int8]
    first_ki_step: NDArray[np.int64]
    first_ko_step: NDArray[np.int64]
    payment_step: NDArray[np.int64]
    payment_time: NDArray[np.float64]
    net_profit: NDArray[np.float64]
    gross_redemption: NDArray[np.float64]
    terminal_spot: NDArray[np.float64]

    def cashflows(self, cashflow: Cashflow) -> NDArray[np.float64]:
        return self.gross_redemption if includes_principal(cashflow) else self.net_profit


def _first_hit(hits: NDArray[np.bool_], steps: NDArray[np.int64]) -> NDArray[np.int64]:
    if not steps.size:
        return np.full(hits.shape[0], -1, dtype=np.int64)
    return np.where(hits.any(axis=1), steps[hits.argmax(axis=1)], -1)


def classify_payoffs(
    contract: Snowball, paths: ArrayLike, *, step: int = 0, knocked_in: bool = False
) -> Payoffs:
    """Classify live paths from immediately after the observation at step.

    Column zero is current spot, followed by the remaining schedule points.
    A missing barrier hit uses index -1. For a previously knocked-in contract,
    first_ki_step is set to step to encode the known state.
    """
    schedule = contract.schedule
    if not isinstance(step, Integral) or not 0 <= step < schedule.n_steps:
        raise ValueError("valuation step must precede maturity")
    prices = np.asarray(paths, dtype=float)
    if prices.ndim != 2 or prices.shape[1] != len(schedule.times) - step or not len(prices):
        raise ValueError("paths must have one column for current spot and each remaining date")
    if not np.isfinite(prices).all() or np.any(prices < 0):
        raise ValueError("path prices must be finite and nonnegative")

    ko_steps = np.asarray([i for i in schedule.ko_steps if i > step], dtype=np.int64)
    ki_steps = np.asarray([i for i in schedule.ki_steps if i > step], dtype=np.int64)
    first_ko = _first_hit(prices[:, ko_steps - step] >= contract.ko_barrier, ko_steps)
    if knocked_in:
        first_ki = np.full(len(prices), step, dtype=np.int64)
    else:
        daily = prices[:, 1:] if len(ki_steps) == prices.shape[1] - 1 else prices[:, ki_steps - step]
        first_ki = _first_hit(daily < contract.ki_barrier, ki_steps)
    called = first_ko >= 0
    first_ki = np.where(called & (first_ki >= first_ko), -1, first_ki)
    hit_ki = first_ki >= 0
    terminal = prices[:, -1].copy()
    below_reference = terminal < contract.initial_spot

    scenario = np.select(
        [called & ~hit_ki, called, ~hit_ki, below_reference],
        [Scenario.KO_WITHOUT_KI, Scenario.KO_AFTER_KI, Scenario.NO_KI_NO_KO,
         Scenario.KI_NO_KO_LOSS],
        default=Scenario.KI_NO_KO_RECOVERED,
    ).astype(np.int8)
    payment_step = np.where(called, first_ko, schedule.n_steps)
    accrual = np.asarray(schedule.accruals)[payment_step]
    downside = np.minimum(terminal, contract.initial_spot) / contract.initial_spot - 1
    net = contract.notional * np.where(called | ~hit_ki, contract.coupon * accrual, downside)
    # Compute fractional principal directly for losses deep in the tail.
    gross = np.where(
        scenario == Scenario.KI_NO_KO_LOSS,
        contract.notional * (np.minimum(terminal, contract.initial_spot) / contract.initial_spot),
        contract.notional + net,
    )
    return Payoffs(
        scenario, first_ki, first_ko, payment_step,
        np.asarray(schedule.times)[payment_step], net, gross, terminal,
    )


def discount_cashflows(
    amounts: ArrayLike, payment_times: ArrayLike, rate: float, *, valuation_time: float = 0.0
) -> NDArray[np.float64]:
    """Discount a single settlement per path from its own payment time."""
    amounts = np.asarray(amounts, dtype=float)
    times = np.asarray(payment_times, dtype=float)
    if (
        not isfinite(rate)
        or not isfinite(valuation_time)
        or not np.isfinite(times).all()
        or not np.isfinite(amounts).all()
        or np.any(times < valuation_time)
    ):
        raise ValueError("cash flows and rates must be finite, with no payment before valuation")
    with np.errstate(over="raise", invalid="raise"):
        return amounts * np.exp(-rate * (times - valuation_time))


def decomposition_payoffs(contract: Snowball, payoffs: Payoffs) -> NDArray[np.float64]:
    """Cash-flow magnitudes of V1, V2, V3 and V4 for the same paths."""
    called = payoffs.first_ko_step >= 0
    clean = payoffs.first_ki_step < 0
    put = contract.notional * (
        1 - np.minimum(payoffs.terminal_spot, contract.initial_spot) / contract.initial_spot
    )
    return np.column_stack((
        np.where(called, payoffs.net_profit, 0),
        np.where(~called & clean, payoffs.net_profit, 0),
        np.where(~called, put, 0),
        np.where(~called & clean, put, 0),
    ))
