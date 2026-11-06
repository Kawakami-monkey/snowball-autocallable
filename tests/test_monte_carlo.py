from dataclasses import replace

import numpy as np
import pytest

from snowball import Market, ObservationSchedule, Snowball, price_monte_carlo
from snowball.greeks import monte_carlo_greeks
from snowball.monte_carlo import estimate_samples
from snowball.simulation import gbm_paths


def test_exact_log_transitions_include_risk_neutral_carry():
    market = Market(rate=0.04, volatility=0.20, dividend=0.11)
    t = [0, 0.15, 0.55]
    z = np.array([[0.0, 0.0], [1.0, -1.0]])
    dt = np.diff(t)
    log_increment = (0.04 - 0.11 - 0.02) * dt + 0.20 * np.sqrt(dt) * z
    expected = 105 * np.exp(np.cumsum(log_increment, axis=1))
    actual = gbm_paths(105, t, market, z)
    np.testing.assert_array_equal(actual[:, 0], [105, 105])
    np.testing.assert_allclose(actual[:, 1:], expected)


@pytest.mark.parametrize("antithetic", [False, True])
def test_seed_reproducibility_across_batch_sizes(antithetic):
    contract, market = Snowball(), Market()
    one = price_monte_carlo(contract, market, n_paths=512, seed=18, batch_size=127,
                            antithetic=antithetic)
    two = price_monte_carlo(contract, market, n_paths=512, seed=18, batch_size=512,
                            antithetic=antithetic)
    np.testing.assert_array_equal(one.discounted, two.discounted)
    assert one.estimate == two.estimate
    other = price_monte_carlo(contract, market, n_paths=512, seed=19, antithetic=antithetic)
    assert not np.array_equal(one.discounted, other.discounted)


def test_iid_and_paired_standard_errors_and_intervals():
    samples = np.array([1, 9, 2, 4, 1, 7, 3, 9])
    for paired in (False, True):
        estimate = estimate_samples(samples, antithetic=paired)
        observations = samples.reshape(-1, 2).mean(1) if paired else samples
        expected_se = observations.std(ddof=1) / np.sqrt(len(observations))
        assert estimate.n_independent == len(observations)
        assert estimate.standard_error == pytest.approx(expected_se)
        lo, hi = estimate.confidence_interval
        assert lo < estimate.mean < hi
        assert (hi - lo) / 2 == pytest.approx(1.959964 * expected_se)
    for values in ([1], [1, 2, 3], [1, 2]):
        with pytest.raises(ValueError):
            estimate_samples(values, antithetic=True)


def test_fixed_maturity_and_gross_pv():
    terms = Snowball(schedule=ObservationSchedule((0, 0.2, 0.75), (), (), (0, 0.25, 0.8)),
                     notional=100)
    market = Market(rate=-0.02)
    for mode, amount in [("net", 16), ("gross", 116)]:
        mc = price_monte_carlo(terms, market, n_paths=16, cashflow=mode, step=1, elapsed=0.05)
        assert mc.price == pytest.approx(amount * np.exp(0.02 * 0.50))
        assert mc.estimate.standard_error == pytest.approx(0, abs=1e-12)


def test_common_random_number_greeks_on_deterministic_coupon():
    terms = replace(Snowball(), schedule=ObservationSchedule((0, 0.1, 1), (), ()))
    risk = monte_carlo_greeks(terms, Market(), n_paths=32)
    pv = terms.coupon * np.exp(-0.03)
    for name in ("delta", "gamma", "vega"):
        assert risk[name].mean == pytest.approx(0)
        assert risk[name].standard_error == 0
    assert risk["rho"].mean == pytest.approx(-pv, rel=1e-6)
    assert risk["theta"].mean == pytest.approx(0.03 * pv, rel=1e-3)
