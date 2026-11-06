"""Self-financing issuer hedge under the pricing model's spot and cash dynamics."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .greeks import spot_delta
from .payoff import classify_payoffs, discount_cashflows
from .pde import PDESolution

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HedgeTrace:
    indices: tuple[int, ...]
    spot: FloatArray
    shares: FloatArray
    cash: FloatArray
    share_trade: FloatArray


@dataclass(frozen=True)
class HedgeResult:
    settlement_step: NDArray[np.int64]
    pnl_at_settlement: FloatArray
    pnl_pv: FloatArray
    pnl_at_maturity: FloatArray
    replication_error_pv: FloatArray
    turnover_pv: FloatArray
    costs_pv: FloatArray
    ki_trade_shares: FloatArray
    ko_liquidation_shares: FloatArray
    mean_routine_abs_trade_shares: float
    trace: HedgeTrace


def delta_hedge(
    solution: PDESolution,
    paths: ArrayLike,
    *,
    rebalance_every: int | None = 1,
    cost_bps: float = 0.0,
    spot_bump_fraction: float = 0.005,
    trace_indices: Iterable[int] = (),
) -> HedgeResult:
    """Short the claim at its model value, then buy the investor Delta.

    Rebalance every k schedule intervals, or hold cash only when k is None.
    Continuous dividends are reinvested in spot without additional trading costs.
    On a call or maturity date, liquidate before settling the investor's cash flow.
    """
    if rebalance_every is not None and (
        not isinstance(rebalance_every, Integral)
        or isinstance(rebalance_every, bool)
        or rebalance_every < 1
    ):
        raise ValueError("rebalance_every must be a positive integer or None")
    if not isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    if not isfinite(spot_bump_fraction) or spot_bump_fraction <= 0:
        raise ValueError("spot_bump_fraction must be positive and finite")

    contract = solution.contract
    times = np.asarray(contract.schedule.times)
    prices = np.asarray(paths, dtype=float)
    payoffs = classify_payoffs(contract, prices)
    if np.any(prices[:, 0] != contract.initial_spot):
        raise ValueError("hedge paths must start at the issue spot")
    trace_indices = tuple(trace_indices)
    if any(not isinstance(i, Integral) or not 0 <= i < len(prices) for i in trace_indices):
        raise ValueError("trace indices must identify valid paths")
    trace_idx = np.asarray(trace_indices, dtype=int)
    trace_shape = (len(trace_idx), len(times))
    trace = HedgeTrace(trace_indices, *(np.full(trace_shape, np.nan) for _ in range(4)))

    bump = contract.initial_spot * spot_bump_fraction
    rate, q = solution.market.rate, solution.market.dividend
    cashflow = payoffs.cashflows(solution.cashflow)
    payment_step = payoffs.payment_step
    ki_step = payoffs.first_ki_step
    cash = np.full(len(prices), solution.price)
    shares = np.zeros(len(prices))
    turnover_pv = np.zeros(len(prices))
    ki_trade_shares = np.full(len(prices), np.nan)
    ko_liquidation_shares = np.full(len(prices), np.nan)
    pnl = np.empty(len(prices))
    trades = np.zeros(len(prices))
    unit_cost = cost_bps / 10_000
    routine_volume, routine_count = 0.0, 0

    with np.errstate(over="raise", invalid="raise"):
        for j, t in enumerate(times):
            live = np.flatnonzero(payment_step >= j)
            if not live.size:
                break
            spot = prices[live, j]
            if j:
                dt = t - times[j - 1]
                cash[live] *= np.exp(rate * dt)
                shares[live] *= np.exp(q * dt)

            settles = payment_step[live] == j
            target = shares[live].copy()
            target[settles] = 0
            rebalances = rebalance_every is not None and j % rebalance_every == 0
            if rebalances:
                survivors = np.flatnonzero(~settles)
                ki_now = (ki_step[live[survivors]] >= 0) & (ki_step[live[survivors]] <= j)
                for state in (False, True):
                    selected = survivors[ki_now == state]
                    if selected.size:
                        target[selected] = spot_delta(solution, spot[selected], j, state, bump)

            share_trade = target - shares[live]
            value_traded = np.abs(share_trade) * spot
            cash[live] -= share_trade * spot + unit_cost * value_traded
            turnover_pv[live] += np.exp(-rate * t) * value_traded
            shares[live] = target

            new_ki = (ki_step[live] == j) & ~settles
            ki_trade_shares[live[new_ki]] = share_trade[new_ki]
            if j > 0 and rebalances:
                routine = share_trade[~(new_ki | settles)]
                routine_volume += float(np.abs(routine).sum())
                routine_count += len(routine)
            calls = payoffs.first_ko_step[live] == j
            ko_liquidation_shares[live[calls]] = share_trade[calls]
            paid = live[settles]
            cash[paid] -= cashflow[paid]
            pnl[paid] = cash[paid]

            showing = payment_step[trace_idx] >= j
            selected_paths = trace_idx[showing]
            trades[live] = share_trade
            trace.spot[showing, j] = prices[selected_paths, j]
            trace.shares[showing, j] = shares[selected_paths]
            trace.cash[showing, j] = cash[selected_paths]
            trace.share_trade[showing, j] = trades[selected_paths]

        pnl_pv = discount_cashflows(pnl, payoffs.payment_time, rate)
        costs = unit_cost * turnover_pv
        terminal_pnl = pnl_pv * np.exp(rate * contract.maturity)
    return HedgeResult(
        payment_step, pnl, pnl_pv, terminal_pnl, pnl_pv + costs, turnover_pv, costs,
        ki_trade_shares, ko_liquidation_shares,
        routine_volume / routine_count if routine_count else 0.0, trace,
    )
