from math import exp

import numpy as np
import pytest

from snowball import Market, ObservationSchedule, Snowball, solve_pde
from snowball.greeks import spot_delta
from snowball.hedging import delta_hedge


def test_weekly_funding_dividend_reinvestment_costs_and_early_settlement():
    terms = Snowball(schedule=ObservationSchedule((0, 0.1, 0.25, 0.45), (1, 2, 3), (2, 3)))
    market = Market(dividend=0.08)
    solution = solve_pde(terms, market)
    paths = [[1.0, 0.82, 1.07, 0.70]]
    cost = 0.002
    hedge = delta_hedge(solution, paths, rebalance_every=5, cost_bps=20, trace_indices=(0,))
    start_shares = float(spot_delta(solution, 1.0, 0, False, 0.005))
    end_shares = start_shares * exp(market.dividend * 0.25)
    start_cash = solution.price - start_shares - cost * abs(start_shares)
    payout = terms.coupon * 0.25
    expected_pnl = start_cash * exp(market.rate * 0.25) + (
        end_shares - cost * abs(end_shares)
    ) * 1.07 - payout
    assert hedge.settlement_step[0] == 2
    assert hedge.pnl_at_settlement[0] == pytest.approx(expected_pnl)
    assert hedge.pnl_pv[0] == pytest.approx(expected_pnl * exp(-market.rate * 0.25))
    assert hedge.pnl_at_maturity[0] == pytest.approx(expected_pnl * exp(market.rate * 0.2))
    assert hedge.ki_trade_shares[0] == 0
    assert hedge.ko_liquidation_shares[0] == pytest.approx(-end_shares)
    assert hedge.trace.cash[0, 0] == pytest.approx(start_cash)
    assert hedge.trace.shares[0, 2] == 0
    assert np.isnan(hedge.trace.cash[0, 3])
    no_cost = delta_hedge(solution, paths, rebalance_every=5)
    np.testing.assert_allclose(hedge.replication_error_pv, no_cost.pnl_pv)


def test_daily_hedge_uses_post_ki_delta_before_next_observation():
    terms = Snowball(schedule=ObservationSchedule((0, 0.1, 0.25, 0.45), (1, 2, 3), (2, 3)))
    solution = solve_pde(terms, Market())
    paths = [[1.0, 0.82, 1.07, 0.70]]
    hedge = delta_hedge(solution, paths, trace_indices=(0,))
    expected = spot_delta(solution, 0.82, 1, True, 0.005)
    assert hedge.trace.shares[0, 1] == pytest.approx(expected)
    unhedged = delta_hedge(solution, paths, rebalance_every=None, trace_indices=(0,))
    assert unhedged.turnover_pv[0] == 0
    assert unhedged.pnl_pv[0] == pytest.approx(solution.price - 0.05 * exp(-0.03 * 0.25))
