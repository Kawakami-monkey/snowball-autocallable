from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from snowball import Market, ObservationSchedule, Scenario, Snowball, classify_payoffs
from snowball.payoff import decomposition_payoffs, discount_cashflows


def test_five_cases_and_termination_chronology():
    schedule = ObservationSchedule((0, 0.1, 0.25, 0.5, 1), (1, 2, 3, 4), (2, 4))
    contract = Snowball(schedule=schedule, notional=1_000)
    paths = [
        [1, 1, 1.05, 0.8, 0.75],
        [1, 1, 1, 0.95, 0.95],
        [1, 0.8, 0.9, 1.05, 1.01],
        [1, 0.8, 0.9, 1.05, 0.95],
        [1, 0.8, 1.05, 0.8, 0.75],
    ]
    payoffs = classify_payoffs(contract, paths)
    np.testing.assert_array_equal(payoffs.scenario, list(Scenario))
    np.testing.assert_array_equal(payoffs.first_ki_step, [-1, -1, 1, 1, 1])
    np.testing.assert_array_equal(payoffs.first_ko_step, [2, -1, -1, -1, 2])
    np.testing.assert_allclose(payoffs.payment_time, [0.25, 1, 1, 1, 0.25])
    np.testing.assert_allclose(payoffs.net_profit, [50, 200, 0, -50, 50])
    np.testing.assert_allclose(payoffs.gross_redemption, [1050, 1200, 1000, 950, 1050])
    legs = decomposition_payoffs(contract, payoffs)
    np.testing.assert_allclose(legs @ [1, 1, -1, 1], payoffs.net_profit)
    expected = [50 * np.exp(-0.03 * 0.25), 200 * np.exp(-0.03), 0, -50 * np.exp(-0.03),
                50 * np.exp(-0.03 * 0.25)]
    np.testing.assert_allclose(discount_cashflows(payoffs.net_profit, payoffs.payment_time, 0.03),
                               expected)


def test_inclusive_ko_strict_ki_and_maturity_observations():
    schedule = ObservationSchedule((0, 0.25, 0.5, 1), (1, 2, 3), (2, 3))
    contract = Snowball(schedule=schedule)
    below_ki = np.nextafter(contract.ki_barrier, 0.0)
    below_ko = np.nextafter(contract.ko_barrier, 0.0)
    payoffs = classify_payoffs(contract, [
        [1, 0.85, below_ko, 0.85],
        [1, below_ki, below_ko, 1.0],
        [1, 1.5, 1.03, 0.8],
        [1, 1.5, below_ko, 1.03],
        [1, 1.5, below_ko, below_ki],
    ])
    np.testing.assert_array_equal(payoffs.scenario, [2, 3, 1, 1, 4])
    np.testing.assert_array_equal(payoffs.first_ko_step, [-1, -1, 2, 3, -1])
    np.testing.assert_allclose(payoffs.net_profit, [0.2, 0, 0.1, 0.2, below_ki - 1])


def test_irregular_calendar_and_fixed_coupon_accrual():
    schedule = ObservationSchedule((0, 0.2, 0.36, 1), (2,), (2,), (0, 0.18, 0.34, 1.01))
    contract = Snowball(schedule=schedule)
    payoffs = classify_payoffs(contract, [[1, 0.5, 1.05, 0.1], [1, 1.5, 1, 0.5]])
    np.testing.assert_array_equal(payoffs.scenario, [1, 2])
    np.testing.assert_allclose(payoffs.net_profit, [0.2 * 0.34, 0.2 * 1.01])
    np.testing.assert_allclose(discount_cashflows(
        payoffs.net_profit, payoffs.payment_time, 0.03, valuation_time=0.1
    ), payoffs.net_profit * np.exp(-0.03 * np.array([0.26, 0.9])))


def test_post_observation_state_and_reference_remain_fixed():
    contract = Snowball(schedule=ObservationSchedule((0, 0.25, 0.5, 1), (1, 2, 3), (1, 3)))
    past_ki = classify_payoffs(contract, [[0.9, 1, 0.94]], step=1, knocked_in=True)
    assert past_ki.first_ki_step[0] == 1
    assert past_ki.scenario[0] == Scenario.KI_NO_KO_LOSS
    assert past_ki.net_profit[0] == pytest.approx(-0.06)
    past_ko_date = classify_payoffs(contract, [[1.03, 0.95, 0.9]], step=1)
    assert past_ko_date.scenario[0] == Scenario.NO_KI_NO_KO


def test_baseline_and_actual_date_schedules():
    regular = Snowball().schedule
    assert regular.ki_steps == tuple(range(1, 253))
    assert regular.ko_steps == tuple(range(21, 253, 21))
    assert regular.times[21] == pytest.approx(1 / 12)
    assert ObservationSchedule.regular(0.31).times[-1] == 0.31
    assert ObservationSchedule.regular(0.31).ko_steps[-1] == 79
    dated = ObservationSchedule.from_dates(
        date(2024, 1, 2), date(2025, 1, 2), ki_dates=[date(2024, 1, 3)], ko_dates=[date(2024, 2, 2)]
    )
    assert dated.ki_steps == (1,)
    assert dated.ko_steps == (2,)
    assert dated.times == (0, 1 / 365, 31 / 365, 366 / 365)
    assert dated.accruals == dated.times


@pytest.mark.parametrize("change", [
    {"initial_spot": 0}, {"notional": -1}, {"coupon": -0.1},
    {"ki_ratio": 1.01}, {"ko_ratio": 1}, {"coupon": float("nan")},
])
def test_invalid_terms(change):
    with pytest.raises(ValueError):
        replace(Snowball(), **change)


@pytest.mark.parametrize("times,ki,ko,accrual", [
    ((0, 1, 0.8), (), (), None),
    ((0, 1), (0,), (1,), None),
    ((0, 1), (), (2,), None),
    ((0, 0.5, 1), (1, 1), (), None),
    ((0, 1), (), (), (0, -0.1)),
])
def test_invalid_observations(times, ki, ko, accrual):
    with pytest.raises(ValueError):
        ObservationSchedule(times, ki, ko, accrual)


def test_bad_prices_market_and_discount_times():
    with pytest.raises(ValueError):
        classify_payoffs(Snowball(), [[1, 2]])
    with pytest.raises(ValueError):
        classify_payoffs(Snowball(), np.full((2, 253), np.nan))
    with pytest.raises(ValueError):
        discount_cashflows([1], [0.3], 0.03, valuation_time=0.5)
    with pytest.raises(ValueError):
        Market(volatility=0.0)
