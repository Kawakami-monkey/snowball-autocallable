"""Backward Crank-Nicolson valuation on a log-spot grid with discrete event maps."""

from dataclasses import dataclass
from math import ceil, isfinite, log
from numbers import Integral
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded

from .contract import Cashflow, Market, Snowball, includes_principal

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PDEGrid:
    """Spatial intervals and a minimum count of time substeps per calendar interval."""

    space_steps: int = 1_000
    time_substeps: int = 4
    log_width: float = 1.0
    rannacher: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.space_steps, Integral)
            or not isinstance(self.time_substeps, Integral)
            or self.space_steps < 20
            or self.time_substeps < 1
            or isinstance(self.time_substeps, bool)
            or not isfinite(self.log_width)
            or self.log_width <= 0
        ):
            raise ValueError("require space_steps >= 20, time_substeps >= 1 and positive log_width")


@dataclass(frozen=True)
class PDEDiagnostics:
    log_spacing: float
    cn_steps: int
    implicit_steps: int
    max_substeps: int
    used_upwind: bool
    runtime_seconds: float


@dataclass(frozen=True)
class PDESolution:
    contract: Snowball
    market: Market
    grid: PDEGrid
    cashflow: Cashflow
    spots: FloatArray
    values: FloatArray  # time index, spot node, KI state (0 or 1)
    diagnostics: PDEDiagnostics

    def value(
        self,
        spot: ArrayLike | None = None,
        *,
        step: int = 0,
        knocked_in: bool = False,
        before_observation: bool = False,
    ) -> FloatArray | np.float64:
        """Value a surviving claim. Apply exact barrier rules if queried before the event."""
        schedule = self.contract.schedule
        if not isinstance(step, Integral) or not 0 <= step <= schedule.n_steps:
            raise ValueError("step must be a valid schedule index")
        spot = np.asarray(self.contract.initial_spot if spot is None else spot, dtype=float)
        if (
            not np.isfinite(spot).all()
            or np.any(spot < self.spots[0])
            or np.any(spot > self.spots[-1])
        ):
            raise ValueError("requested spot is outside the PDE domain")
        state = int(knocked_in)
        if step == schedule.n_steps:
            principal = self.contract.notional if self.cashflow == "gross" else 0.0
            clean = np.full_like(spot, principal + self.contract.notional
                                 * self.contract.coupon * schedule.accruals[step])
            fraction = np.minimum(spot, self.contract.initial_spot) / self.contract.initial_spot
            hit = self.contract.notional * (fraction if self.cashflow == "gross" else fraction - 1)
            value = hit if state else clean
        else:
            value = np.interp(spot, self.spots, self.values[step, :, state])
            hit = np.interp(spot, self.spots, self.values[step, :, 1]) if (
                before_observation and step in schedule.ki_steps and not knocked_in
            ) else value
        if before_observation and step in schedule.ki_steps and not knocked_in:
            value = np.where(spot < self.contract.ki_barrier, hit, value)
        if before_observation and step in schedule.ko_steps:
            redemption = self.contract.notional * (
                (self.cashflow == "gross") + self.contract.coupon * schedule.accruals[step]
            )
            value = np.where(spot >= self.contract.ko_barrier, redemption, value)
        return value[()] if isinstance(value, np.ndarray) and value.ndim == 0 else value

    @property
    def price(self) -> float:
        return float(self.value())


@dataclass(frozen=True)
class OptionComponents:
    """Unsigned leg prices in report order. V3 is short in the investor portfolio."""

    v1: float
    v2: float
    v3: float
    v4: float
    diagnostics: PDEDiagnostics

    @property
    def total(self) -> float:
        return self.v1 + self.v2 - self.v3 + self.v4


class _LogStepper:
    def __init__(self, market: Market, dx: float, grid: PDEGrid) -> None:
        self.market = market
        self.grid = grid
        diffusion = 0.5 * (market.volatility / dx)**2
        advection = market.log_drift / (2 * dx)
        self.lower = diffusion - advection
        self.upper = diffusion + advection
        self.used_upwind = min(self.lower, self.upper) < 0
        if self.used_upwind:
            # Use a positive spatial generator when the centered stencil is drift-dominated.
            self.lower = diffusion + max(-market.log_drift, 0) / dx
            self.upper = diffusion + max(market.log_drift, 0) / dx
        self.exit_rate = self.lower + self.upper
        if not isfinite(self.exit_rate):
            raise ValueError("market inputs and grid produce nonfinite stencil coefficients")
        self.bands: dict[tuple[float, float], FloatArray] = {}
        self.cn_steps = 0
        self.implicit_steps = 0
        self.max_substeps = 0

    def _step(
        self, later: FloatArray, carry: FloatArray, dt: float, theta: float
    ) -> tuple[FloatArray, FloatArray]:
        implicit = theta * dt
        explicit = (1 - theta) * dt
        key = (dt, theta)
        if key not in self.bands:
            matrix = np.zeros((3, self.grid.space_steps - 1))
            matrix[0, 1:] = -implicit * self.upper
            matrix[1] = 1 + implicit * self.exit_rate
            matrix[2, :-1] = -implicit * self.lower
            self.bands[key] = matrix

        # Factor out the risk-free discount. The stock part of the lower boundary
        # discounts at q, so its transformed boundary changes by exp((r-q) dt).
        discount = np.exp(-self.market.rate * dt)
        left = later[0] + carry * np.expm1((self.market.rate - self.market.dividend) * dt)
        right = later[-1]
        rhs = (
            (1 - explicit * self.exit_rate) * later[1:-1]
            + explicit * (self.lower * later[:-2] + self.upper * later[2:])
        )
        rhs[0] += implicit * self.lower * left
        rhs[-1] += implicit * self.upper * right
        earlier = np.empty_like(later)
        earlier[0] = discount * left
        earlier[-1] = discount * right
        earlier[1:-1] = discount * solve_banded((1, 1), self.bands[key], rhs, check_finite=False)
        return earlier, carry * np.exp(-self.market.dividend * dt)

    def propagate(
        self, values: FloatArray, carry: FloatArray, interval: float, *, damp: bool
    ) -> FloatArray:
        # In addition to CN's linear stability, enforce nonnegative explicit weights.
        count = max(self.grid.time_substeps, ceil(0.5 * interval * self.exit_rate))
        self.max_substeps = max(self.max_substeps, count)
        dt = interval / count
        if damp and self.grid.rannacher:
            for _ in range(2):
                values, carry = self._step(values, carry, dt / 2, 1.0)
            self.implicit_steps += 2
            count -= 1
        for _ in range(count):
            values, carry = self._step(values, carry, dt, 0.5)
        self.cn_steps += count
        return values


def _backward(
    contract: Snowball, market: Market, grid: PDEGrid, *, principal: float, components: bool
) -> tuple[FloatArray, FloatArray, PDEDiagnostics]:
    start = perf_counter()
    schedule = contract.schedule
    times = schedule.times
    coupons = contract.notional * contract.coupon * np.asarray(schedule.accruals)
    n_steps = schedule.n_steps
    ki_steps, ko_steps = set(schedule.ki_steps), set(schedule.ko_steps)
    dx = 2 * grid.log_width / grid.space_steps
    x = np.linspace(-grid.log_width, grid.log_width, grid.space_steps + 1)
    log_ki, log_ko = log(contract.ki_ratio), log(contract.ko_ratio)
    if not x[1] < log_ki < log_ko < x[-2]:
        raise ValueError("keep both barriers at least one log-spot interval inside the PDE domain")
    if x[0] + (market.rate - market.dividend) * contract.maturity >= 0:
        raise ValueError("widen the PDE domain so the lower forward stays below the put strike")

    with np.errstate(over="raise", invalid="raise"):
        spots = np.exp(x) * contract.initial_spot
    if spots[0] <= 0 or np.any(np.diff(spots) <= 0):
        raise ValueError("PDE spot grid must be positive and strictly increasing")
    # Project a discontinuous event onto each log cell using the fraction on either side.
    ki_weights = np.clip(0.5 + (log_ki - x) / dx, 0, 1)
    ko_weights = np.clip(0.5 + (x - log_ko) / dx, 0, 1)
    put = contract.notional * np.maximum(1 - np.exp(x), 0)
    coupon = np.full_like(x, coupons[-1])
    if components:
        terminal = np.column_stack((np.zeros_like(x), coupon, put, put))
    else:
        terminal = np.column_stack((principal + coupon, principal - put))
    values = np.empty((n_steps + 1, *terminal.shape))
    values[-1] = terminal
    stepper = _LogStepper(market, dx, grid)
    future_ki = False
    next_call = None

    with np.errstate(over="raise", invalid="raise"):
        for j in range(n_steps, 0, -1):
            ki = j in ki_steps
            ko = j in ko_steps
            future_ki = future_ki or ki
            if ko:
                next_call = j
            maturity_df = np.exp(-market.rate * (times[-1] - times[j]))
            low_stock = (contract.notional * np.exp(x[0] - market.dividend
                                                   * (times[-1] - times[j])))
            low_put = contract.notional * maturity_df - low_stock
            # At very high spot the next call is certain, when one remains.
            call_df = np.exp(-market.rate * (times[next_call] - times[j])) if (
                next_call is not None
            ) else 0.0
            call_weight = ko_weights if ko else 0.0
            knock_in_weight = ki_weights if ki else 0.0
            survive_call = 1 - call_weight
            survive_both = survive_call - knock_in_weight
            continuation = values[j]
            later = np.empty_like(continuation)

            if components:
                later[:, 0] = survive_call * continuation[:, 0] + call_weight * coupons[j]
                later[:, 1] = survive_both * continuation[:, 1]
                later[:, 2] = survive_call * continuation[:, 2]
                later[:, 3] = survive_both * continuation[:, 3]
                later[0] = (0, 0 if future_ki else coupons[-1] * maturity_df,
                            low_put, 0 if future_ki else low_put)
                later[-1] = (
                    0 if next_call is None else coupons[next_call] * call_df,
                    coupons[-1] * maturity_df if next_call is None else 0, 0, 0,
                )
                carry = np.array([0, 0, -low_stock, 0 if future_ki else -low_stock])
            else:
                # KI and KO occupy disjoint fractions, even if they share a coarse cell.
                call_cash = call_weight * (principal + coupons[j])
                later[:, 0] = (survive_both * continuation[:, 0]
                               + knock_in_weight * continuation[:, 1] + call_cash)
                later[:, 1] = survive_call * continuation[:, 1] + call_cash
                low_hit = principal * maturity_df - low_put
                later[0] = (low_hit if future_ki else (principal + coupons[-1]) * maturity_df,
                            low_hit)
                if next_call is None:
                    later[-1] = ((principal + coupons[-1]) * maturity_df, principal * maturity_df)
                else:
                    later[-1] = (principal + coupons[next_call]) * call_df
                carry = np.array([low_stock if future_ki else 0, low_stock])

            values[j - 1] = stepper.propagate(
                later, carry, times[j] - times[j - 1], damp=ki or ko or j == n_steps
            )

    if not np.isfinite(values).all():
        raise FloatingPointError("PDE calculation produced nonfinite values")
    diagnostics = PDEDiagnostics(dx, stepper.cn_steps, stepper.implicit_steps,
                                 stepper.max_substeps, stepper.used_upwind, perf_counter() - start)
    return spots, values, diagnostics


def solve_pde(
    contract: Snowball, market: Market, grid: PDEGrid = PDEGrid(), *, cashflow: Cashflow = "net"
) -> PDESolution:
    """Solve for separate clean and already-knocked-in continuation surfaces."""
    principal = contract.notional if includes_principal(cashflow) else 0.0
    spots, values, diagnostics = _backward(contract, market, grid, principal=principal,
                                           components=False)
    return PDESolution(contract, market, grid, cashflow, spots, values, diagnostics)


def solve_decomposition(
    contract: Snowball, market: Market, grid: PDEGrid = PDEGrid()
) -> OptionComponents:
    """Price the four discrete-barrier legs independently of the two-state event map."""
    spots, values, diagnostics = _backward(contract, market, grid, principal=0.0, components=True)
    legs = [float(np.interp(contract.initial_spot, spots, leg)) for leg in values[0].T]
    return OptionComponents(*legs, diagnostics)
