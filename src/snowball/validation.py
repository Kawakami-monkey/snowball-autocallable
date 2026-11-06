"""Numerical diagnostics and read-only reference values from Cinda's Tables 3, 4 and 7."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contract import Market, Snowball
from .monte_carlo import Estimate, MonteCarloResult, estimate_samples, price_monte_carlo
from .payoff import Scenario, decomposition_payoffs, discount_cashflows
from .pde import OptionComponents, PDEGrid, PDESolution, solve_pde

REPORT_PRICES = {
    ("baseline", "Monte Carlo"): 0.05086,
    ("baseline", "PDE"): 0.05141,
    ("effective carry", "PDE"): -0.03731,
}
REPORT_EVENTS = {
    "KO probability": 0.7379,
    "No KI and no KO probability": 0.1369,
    "KI and no KO probability": 0.1252,
    "Loss probability": 0.1232,
    "Mean KO month | KO": 2.59,
    "Mean life in months": 5.74,
}
REPORT_LEGS = (0.04213, 0.02633, 0.02501, 0.00796)
REPORT_MEAN_KO_COUPON = 0.0432  # Section 2.1.1, p. 8


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    contract: Snowball
    market: Market


def benchmark_scenarios() -> tuple[BenchmarkScenario, BenchmarkScenario]:
    return (
        BenchmarkScenario("baseline", Snowball(), Market()),
        BenchmarkScenario("effective carry", Snowball(coupon=0.1882),
                          Market(volatility=0.19, dividend=0.11)),
    )


def estimate_columns(estimate: Estimate) -> dict[str, float]:
    low, high = estimate.confidence_interval
    return {"standard_error": estimate.standard_error, "ci_low": low, "ci_high": high}


def errors(value: float, target: float) -> dict[str, float]:
    absolute = abs(value - target)
    return {"signed_error": value - target, "absolute_error": absolute,
            "relative_error": absolute / abs(target) if target else np.nan}


def benchmark_rows(
    scenario: BenchmarkScenario, mc: MonteCarloResult, pde: PDESolution, gross_pde: PDESolution
) -> pd.DataFrame:
    pv_gross_mc = discount_cashflows(
        mc.payoffs.gross_redemption, mc.payoffs.payment_time, scenario.market.rate
    )
    rows = []
    for method, value, gross, seconds in (
        ("Monte Carlo", mc.price, float(pv_gross_mc.mean()), mc.runtime_seconds),
        ("PDE", pde.price, gross_pde.price, pde.diagnostics.runtime_seconds),
    ):
        ref = REPORT_PRICES.get((scenario.name, method), np.nan)
        rows.append({
            "scenario": scenario.name,
            "method": method,
            "net_pv": value,
            "gross_pv": gross,
            "gross_pv_minus_par": gross - scenario.contract.notional,
            "report_net_pv": ref,
            **errors(value, ref),
            **(estimate_columns(mc.estimate) if method == "Monte Carlo" else {}),
            "runtime_seconds": seconds,
        })
    return pd.DataFrame(rows)


def cross_method_row(name: str, mc: MonteCarloResult, pde: PDESolution) -> dict:
    low, high = mc.estimate.confidence_interval
    se = mc.estimate.standard_error
    return {
        "scenario": name,
        "mc_price": mc.price,
        "pde_price": pde.price,
        **errors(mc.price, pde.price),
        **estimate_columns(mc.estimate),
        "mc_minus_pde_in_standard_errors": (mc.price - pde.price) / se if se else np.nan,
        "pde_within_mc_95_ci": low <= pde.price <= high,
    }


def event_statistics(name: str, mc: MonteCarloResult) -> pd.DataFrame:
    payoffs = mc.payoffs
    called = payoffs.first_ko_step >= 0
    months = 12 * payoffs.payment_time
    mean_ko_month = float(months[called].mean()) if called.any() else np.nan
    estimates = {
        "KO probability": called,
        "No KI and no KO probability": payoffs.scenario == Scenario.NO_KI_NO_KO,
        "KI and no KO probability": ~called & (payoffs.first_ki_step >= 0),
        "Loss probability": payoffs.scenario == Scenario.KI_NO_KO_LOSS,
        "Mean life in months": months,
    }
    # Influence function for the conditional mean E[months | KO].
    if called.any():
        estimates["Mean KO month | KO"] = mean_ko_month + (
            called * (months - mean_ko_month) / called.mean()
        )
        estimates["Mean zero-indexed KO month | KO"] = estimates["Mean KO month | KO"] - 1
    rows = []
    for metric, values in estimates.items():
        estimate = estimate_samples(values, antithetic=mc.antithetic)
        ref = REPORT_EVENTS.get(metric, np.nan) if name == "baseline" else np.nan
        rows.append({
            "scenario": name, "metric": metric, "estimate": estimate.mean, "report": ref,
            **errors(estimate.mean, ref), **estimate_columns(estimate),
        })
    return pd.DataFrame(rows)


def decomposition_rows(
    scenario: BenchmarkScenario, mc: MonteCarloResult, pde: OptionComponents
) -> pd.DataFrame:
    components = decomposition_payoffs(scenario.contract, mc.payoffs)
    components *= np.exp(-scenario.market.rate * mc.payoffs.payment_time)[:, None]
    np.testing.assert_allclose(components @ [1, 1, -1, 1], mc.discounted, atol=1e-12)
    rows = []
    for i, (sign, pde_value, ref) in enumerate(zip(
        (1, 1, -1, 1), (pde.v1, pde.v2, pde.v3, pde.v4), REPORT_LEGS
    )):
        estimate = estimate_samples(components[:, i], antithetic=mc.antithetic)
        ref = ref if scenario.name == "baseline" else np.nan
        rows.append({
            "scenario": scenario.name,
            "leg": f"V{i + 1}",
            "investor_sign": sign,
            "mc_price": estimate.mean,
            "pde_price": pde_value,
            "report_price": ref,
            **errors(pde_value, ref), **estimate_columns(estimate),
        })
    return pd.DataFrame(rows)


def pde_convergence(
    scenario: BenchmarkScenario,
    grids: list[PDEGrid],
    *,
    study: str,
    reference: float,
) -> pd.DataFrame:
    rows = []
    for grid in grids:
        solution = solve_pde(scenario.contract, scenario.market, grid)
        diag = solution.diagnostics
        rows.append({
            "scenario": scenario.name,
            "study": study,
            "space_steps": grid.space_steps,
            "log_width": grid.log_width,
            "log_spacing": diag.log_spacing,
            "min_time_substeps": grid.time_substeps,
            "max_effective_substeps": diag.max_substeps,
            "cn_steps": diag.cn_steps,
            "implicit_steps": diag.implicit_steps,
            "used_upwind": diag.used_upwind,
            "price": solution.price,
            "reference_pde": reference,
            **errors(solution.price, reference),
            "runtime_seconds": diag.runtime_seconds,
        })
    return pd.DataFrame(rows)


def mc_convergence(
    scenario: BenchmarkScenario,
    benchmark: MonteCarloResult,
    paths: tuple[int, ...],
    *,
    seed: int,
    batch_size: int,
    pde_reference: float,
) -> pd.DataFrame:
    rows = []
    for antithetic, counts in ((False, paths), (True, (paths[-1],))):
        for n in counts:
            result = benchmark if not antithetic and n == paths[-1] else price_monte_carlo(
                scenario.contract, scenario.market, n_paths=n, seed=seed, batch_size=batch_size,
                antithetic=antithetic,
            )
            rows.append({
                "n_paths": n,
                "n_independent": result.estimate.n_independent,
                "antithetic": antithetic,
                "price": result.price,
                "reference_pde": pde_reference,
                **errors(result.price, pde_reference),
                **estimate_columns(result.estimate),
                "runtime_seconds": result.runtime_seconds,
            })
    return pd.DataFrame(rows)
