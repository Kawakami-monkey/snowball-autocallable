"""Immutable terms and explicit model-time and coupon-accrual calendars."""

from dataclasses import dataclass
from datetime import date
from math import ceil, isfinite
from numbers import Integral
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

Cashflow = Literal["net", "gross"]


def includes_principal(cashflow: Cashflow) -> bool:
    if cashflow not in ("net", "gross"):
        raise ValueError("cashflow must be 'net' or 'gross'")
    return cashflow == "gross"


@dataclass(frozen=True)
class ObservationSchedule:
    """Issue at index 0, maturity at the last index. Neither barrier is observed at issue."""

    times: tuple[float, ...]
    ki_steps: tuple[int, ...]
    ko_steps: tuple[int, ...]
    accruals: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        times = tuple(self.times)
        if (
            len(times) < 2
            or times[0] != 0
            or not all(map(isfinite, times))
            or any(t1 <= t0 for t0, t1 in zip(times, times[1:]))
        ):
            raise ValueError("times must start at zero and increase strictly to maturity")
        object.__setattr__(self, "times", times)
        for name in ("ki_steps", "ko_steps"):
            steps = tuple(getattr(self, name))
            if (
                any(
                    not isinstance(i, Integral) or isinstance(i, bool) or not 0 < i < len(times)
                    for i in steps
                )
                or any(i >= j for i, j in zip(steps, steps[1:]))
            ):
                raise ValueError(f"{name} must contain increasing observation indices after issue")
            object.__setattr__(self, name, steps)
        accruals = times if self.accruals is None else tuple(self.accruals)
        if (
            len(accruals) != len(times)
            or accruals[0] != 0
            or not all(map(isfinite, accruals))
            or any(a1 < a0 for a0, a1 in zip(accruals, accruals[1:]))
        ):
            raise ValueError("accruals must start at zero and be nondecreasing on the time grid")
        object.__setattr__(self, "accruals", accruals)

    @property
    def n_steps(self) -> int:
        return len(self.times) - 1

    @property
    def maturity(self) -> float:
        return self.times[-1]

    @classmethod
    def regular(
        cls, maturity: float = 1.0, trading_days: int = 252, calls_per_year: int = 12
    ) -> "ObservationSchedule":
        """Synthetic trading years. Observe both barriers at maturity, including a final stub."""
        if not isfinite(maturity) or maturity <= 0:
            raise ValueError("maturity must be positive and finite")
        if (
            not isinstance(trading_days, Integral)
            or not isinstance(calls_per_year, Integral)
            or isinstance(trading_days, bool)
            or isinstance(calls_per_year, bool)
            or calls_per_year <= 0
            or trading_days <= 0
            or trading_days % calls_per_year
        ):
            raise ValueError("trading_days must be a positive integer multiple of calls_per_year")
        n_steps = max(1, ceil(round(maturity * trading_days, 12)))
        times = tuple(j / trading_days for j in range(n_steps)) + (maturity,)
        call_period = trading_days // calls_per_year
        ko_steps = tuple(range(call_period, n_steps, call_period)) + (n_steps,)
        return cls(times, tuple(range(1, n_steps + 1)), ko_steps)

    @classmethod
    def from_dates(
        cls,
        issue: date,
        maturity: date,
        *,
        ki_dates: Iterable[date],
        ko_dates: Iterable[date],
    ) -> "ObservationSchedule":
        """Use supplied closing dates and ACT/365F for both model time and coupon."""
        ki = set(ki_dates)
        ko = set(ko_dates)
        if maturity <= issue or any(not issue < d <= maturity for d in ki | ko):
            raise ValueError("observations must lie after issue and no later than maturity")
        dates = [issue, *sorted(ki | ko | {maturity})]
        return cls(
            tuple((d - issue).days / 365 for d in dates),
            tuple(i for i, d in enumerate(dates) if d in ki),
            tuple(i for i, d in enumerate(dates) if d in ko),
        )

    def remaining_times(self, step: int, elapsed: float = 0.0) -> NDArray[np.float64]:
        """Return the valuation time and future dates, conditional on the event at step."""
        if not isinstance(step, Integral) or not 0 <= step < self.n_steps:
            raise ValueError("valuation step must precede maturity")
        if not isfinite(elapsed) or not 0 <= elapsed < self.times[step + 1] - self.times[step]:
            raise ValueError("elapsed must lie before the next observation grid point")
        return np.array((self.times[step] + elapsed, *self.times[step + 1:]))


@dataclass(frozen=True)
class Snowball:
    """Terms are fixed at issue. Spot bumps never move the reference level or barriers."""

    initial_spot: float = 1.0
    ki_ratio: float = 0.85
    ko_ratio: float = 1.03
    coupon: float = 0.20
    notional: float = 1.0
    schedule: ObservationSchedule = ObservationSchedule.regular()

    def __post_init__(self) -> None:
        if not all(map(isfinite, (self.initial_spot, self.ki_ratio, self.ko_ratio,
                                  self.coupon, self.notional))):
            raise ValueError("contract parameters must be finite")
        if self.initial_spot <= 0 or self.notional <= 0 or self.coupon < 0:
            raise ValueError("reference spot and notional must be positive, coupon nonnegative")
        if not 0 < self.ki_ratio < 1 < self.ko_ratio:
            raise ValueError("barriers must satisfy 0 < ki_ratio < 1 < ko_ratio")
        if not isinstance(self.schedule, ObservationSchedule):
            raise TypeError("schedule must be an ObservationSchedule")
        if not isfinite(self.ko_barrier) or self.ki_barrier <= 0:
            raise ValueError("absolute barriers must be representable as finite positive prices")
        if not isfinite(self.notional * (1 + self.coupon * self.schedule.accruals[-1])):
            raise ValueError("maximum redemption must be finite")

    @property
    def ki_barrier(self) -> float:
        return self.ki_ratio * self.initial_spot

    @property
    def ko_barrier(self) -> float:
        return self.ko_ratio * self.initial_spot

    @property
    def maturity(self) -> float:
        return self.schedule.maturity


@dataclass(frozen=True)
class Market:
    """Constant annualized Black-Scholes inputs in decimal units."""

    rate: float = 0.03
    volatility: float = 0.13
    dividend: float = 0.0

    def __post_init__(self) -> None:
        if not all(map(isfinite, (self.rate, self.volatility, self.dividend))):
            raise ValueError("market inputs must be finite")
        if self.volatility <= 0:
            raise ValueError("volatility must be strictly positive")

    @property
    def log_drift(self) -> float:
        return self.rate - self.dividend - 0.5 * self.volatility**2
