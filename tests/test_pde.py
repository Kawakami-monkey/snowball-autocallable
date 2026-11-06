from dataclasses import replace
from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.stats import norm

from snowball import (
    Market, ObservationSchedule, PDEGrid, Snowball, price_monte_carlo, solve_decomposition,
    solve_pde,
)


def test_plain_put_after_ki_matches_black_scholes():
    terms = Snowball(initial_spot=100, notional=250, schedule=ObservationSchedule((0, 1), (), ()))
    market = Market(rate=-0.01, volatility=0.22, dividend=0.07)
    solution = solve_pde(terms, market, PDEGrid(space_steps=400, log_width=1.3))
    for spot in (78, 100, 114):
        d1 = (log(spot / 100) + market.rate - market.dividend + 0.5 * market.volatility**2
              ) / market.volatility
        d2 = d1 - market.volatility
        put = terms.notional * (
            exp(-market.rate) * norm.cdf(-d2) - spot / 100 * exp(-market.dividend) * norm.cdf(-d1)
        )
        assert solution.value(spot, knocked_in=True) == pytest.approx(-put, abs=0.005)
    assert solution.price == pytest.approx(terms.notional * terms.coupon * exp(-market.rate),
                                            abs=1e-9)


def test_grid_convergence_against_truncated_lognormal_at_single_date():
    terms = Snowball(schedule=ObservationSchedule((0, 0.8), (1,), (1,)))
    market = Market(dividend=0.025)
    t = terms.maturity
    d = (log(terms.ki_ratio) - market.log_drift * t) / (market.volatility * sqrt(t))
    p_below = norm.cdf(d)
    s_below = exp((market.rate - market.dividend) * t) * norm.cdf(d - market.volatility * sqrt(t))
    exact = exp(-market.rate * t) * (terms.coupon * t * (1 - p_below) + s_below - p_below)
    errors = []
    for n in (80, 160, 320):
        sol = solve_pde(terms, market, PDEGrid(space_steps=n))
        errors.append(abs(sol.price - exact))
        assert sol.diagnostics.cn_steps > 0
        assert sol.diagnostics.implicit_steps == 2
    assert errors[-1] < 2e-5
    assert errors[-1] < errors[0] / 8


def test_exact_event_queries_and_after_event_surfaces():
    terms = Snowball(schedule=ObservationSchedule((0, 0.2, 0.53), (1, 2), (1, 2)))
    solution = solve_pde(terms, Market(), cashflow="gross")
    assert solution.value(terms.ko_barrier, step=1, before_observation=True) == pytest.approx(1.04)
    assert solution.value(terms.ki_barrier, step=1, before_observation=True) == solution.value(
        terms.ki_barrier, step=1
    )
    assert solution.value(0.84, step=1, before_observation=True) == solution.value(
        0.84, step=1, knocked_in=True
    )
    np.testing.assert_allclose(
        solution.value([0.8, 0.85, 0.97, 1.03], step=2, before_observation=True),
        [0.8, 1.106, 1.106, 1.106],
    )
    np.testing.assert_allclose(
        solution.value([0.8, 1.0, 1.03], step=2, knocked_in=True, before_observation=True),
        [0.8, 1, 1.106],
    )


def test_decomposition_linearity_even_when_barriers_share_a_grid_cell():
    contract = Snowball(ki_ratio=0.998, ko_ratio=1.001)
    grid = PDEGrid(space_steps=100)
    direct = solve_pde(contract, Market(), grid)
    legs = solve_decomposition(contract, Market(), grid)
    assert legs.total == pytest.approx(direct.price, abs=1e-12)
    assert 0 <= legs.v4 <= legs.v3


@pytest.mark.parametrize("carry", [False, True])
def test_cross_method_agreement(carry):
    terms, market = Snowball(), Market()
    if carry:
        terms = replace(terms, coupon=0.1882)
        market = replace(market, volatility=0.19, dividend=0.11)
    pde = solve_pde(terms, market)
    mc = price_monte_carlo(terms, market, n_paths=100_000, seed=42)
    assert abs(pde.price - mc.price) < 3 * mc.estimate.standard_error
    assert pde.price == pytest.approx(solve_decomposition(terms, market).total, abs=1e-12)


def test_domain_and_monotone_stencil_guard():
    with pytest.raises(ValueError):
        solve_pde(Snowball(), Market(), PDEGrid(log_width=0.1))
    terms = Snowball(schedule=ObservationSchedule((0, 0.5, 1), (1, 2), (2,)))
    solution = solve_pde(terms, Market(rate=0.2, volatility=0.01), PDEGrid(space_steps=40))
    assert solution.diagnostics.used_upwind
    assert np.isfinite(solution.values).all()
    with pytest.raises(ValueError):
        solution.value(4.0)
