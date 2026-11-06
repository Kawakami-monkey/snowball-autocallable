"""Regenerate all tables and figures from synthetic paths and PDE grids."""

import argparse
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
import json
import platform
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from snowball import PDEGrid, Scenario, classify_payoffs, price_monte_carlo, solve_decomposition, solve_pde
from snowball import visualization as figures
from snowball.greeks import GREEK_NAMES, GreekBumps, PDERisk, monte_carlo_greeks
from snowball.hedging import delta_hedge
from snowball.sensitivity import parameter_risk, volatility_carry
from snowball.simulation import simulate_paths
from snowball.validation import (
    REPORT_EVENTS, REPORT_LEGS, REPORT_MEAN_KO_COUPON, benchmark_rows, benchmark_scenarios,
    cross_method_row, decomposition_rows, estimate_columns, event_statistics, mc_convergence,
    pde_convergence,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
GREEK_SEED = 123
HEDGE_SEED = 7701
PLOT_SEED = 4242
BATCH_SIZE = 8192


@dataclass(frozen=True)
class Profile:
    mc_paths: tuple[int, ...] = (5_000, 15_000, 50_000, 100_000, 300_000)
    space_steps: tuple[int, ...] = (160, 320, 640, 1280, 2560)
    greek_paths: int = 160_000
    hedge_paths: int = 10_000
    risk_grid: PDEGrid = PDEGrid(space_steps=1280)
    sweep_grid: PDEGrid = PDEGrid(space_steps=520)


def pricing_experiments(profile):
    tables = {}
    baseline, carry = benchmark_scenarios()
    mc_results, pde_results = {}, {}
    benchmark, cross, events, legs, grids = [], [], [], [], []
    finest = PDEGrid(space_steps=profile.space_steps[-1])
    for scenario in (baseline, carry):
        c, m = scenario.contract, scenario.market
        print(f"Pricing {scenario.name}", flush=True)
        mc = price_monte_carlo(c, m, n_paths=profile.mc_paths[-1], seed=SEED, batch_size=BATCH_SIZE)
        pde = solve_pde(c, m, finest)
        gross = solve_pde(c, m, finest, cashflow="gross")
        components = solve_decomposition(c, m, finest)
        if abs(pde.price - components.total) > 1e-10 * c.notional:
            raise ArithmeticError(f"decomposition failed for {scenario.name}")
        if abs(mc.price - pde.price) > 4 * mc.estimate.standard_error:
            raise ArithmeticError(f"MC and PDE differ by over four standard errors for {scenario.name}")
        mc_results[scenario.name] = mc
        pde_results[scenario.name] = pde
        benchmark.append(benchmark_rows(scenario, mc, pde, gross))
        cross.append(cross_method_row(scenario.name, mc, pde))
        events.append(event_statistics(scenario.name, mc))
        legs.append(decomposition_rows(scenario, mc, components))
        grids.append(pde_convergence(scenario, [
            replace(finest, space_steps=n) for n in profile.space_steps
        ], study="space", reference=pde.price))

    c, m = baseline.contract, baseline.market
    temporal = [PDEGrid(space_steps=400, time_substeps=n) for n in (2, 4, 8, 16, 32)]
    grids.append(pde_convergence(baseline, temporal, study="time",
                                 reference=solve_pde(c, m, temporal[-1]).price))
    domains = [
        PDEGrid(space_steps=n, time_substeps=8, log_width=width)
        for n, width in ((260, 0.65), (400, 1.0), (540, 1.35))
    ]
    grids.append(pde_convergence(baseline, domains, study="domain",
                                 reference=solve_pde(c, m, domains[-1]).price))

    tables["benchmarks"] = pd.concat(benchmark, ignore_index=True)
    tables["cross_method"] = pd.DataFrame(cross)
    tables["event_statistics"] = pd.concat(events, ignore_index=True)
    tables["decomposition"] = pd.concat(legs, ignore_index=True)
    tables["pde_convergence"] = pd.concat(grids, ignore_index=True)
    mc = mc_results[baseline.name]
    tables["mc_convergence"] = mc_convergence(
        baseline, mc, profile.mc_paths, seed=SEED, batch_size=BATCH_SIZE,
        pde_reference=pde_results[baseline.name].price,
    )
    return tables, mc


def convention_experiments(profile, mc):
    baseline, carry = benchmark_scenarios()
    c, m = baseline.contract, baseline.market
    schedule = c.schedule
    calls = schedule.ko_steps
    conventions = {
        "21 trading days per call": calls,
        "no maturity call": calls[:-1],
        "interim calls one trading day earlier": tuple(i - 1 for i in calls[:-1]) + calls[-1:],
        "interim calls one trading day later": tuple(i + 1 for i in calls[:-1]) + calls[-1:],
    }
    rows = []
    for name, ko in conventions.items():
        terms = replace(c, schedule=replace(schedule, ko_steps=ko))
        result = mc if ko == calls else price_monte_carlo(
            terms, m, n_paths=profile.mc_paths[-1], seed=SEED, batch_size=BATCH_SIZE
        )
        pde = solve_pde(terms, m, profile.risk_grid)
        called = result.payoffs.first_ko_step >= 0
        rows.append({
            "calendar": name,
            "mc_price": result.price,
            "pde_price": pde.price,
            "ko_probability": called.mean(),
            "mean_ko_month": 12 * result.payoffs.payment_time[called].mean(),
            "mean_life_months": 12 * result.payoffs.payment_time.mean(),
            **estimate_columns(result.estimate),
        })
    p = REPORT_EVENTS["KO probability"]
    ko_month = REPORT_EVENTS["Mean KO month | KO"]
    life = REPORT_EVENTS["Mean life in months"]
    identities = pd.DataFrame([
        {"diagnostic": "Mean life implied by reported KO probability and KO month",
         "implied": p * ko_month + (1 - p) * 12, "reported": life},
        {"diagnostic": "Mean KO month implied by reported KO probability and mean life",
         "implied": (life - (1 - p) * 12) / p, "reported": ko_month},
        {"diagnostic": "Upper bound on V1 using reported KO probability and mean KO coupon",
         "implied": p * REPORT_MEAN_KO_COUPON, "reported": REPORT_LEGS[0]},
    ])
    carry_rows = []
    for q in (0.09, 0.10, 0.105, 0.11, 0.115, 0.12, 0.13):
        pde = solve_pde(carry.contract, replace(carry.market, dividend=q), profile.risk_grid)
        carry_rows.append({"effective_dividend_yield": q, "pde_net_pv": pde.price})
    return {
        "calendar_diagnostics": pd.DataFrame(rows),
        "report_timing_identities": identities,
        "effective_carry_diagnostics": pd.DataFrame(carry_rows),
    }


def risk_experiments(solution, profile):
    print("Computing Greeks and parameter sweeps", flush=True)
    c, m = solution.contract, solution.market
    bumps = GreekBumps()
    engine = PDERisk(solution, bumps)
    issue_greeks, spot_greeks, bump_rows = [], [], []
    ratios = np.linspace(0.76, 1.16, 241)
    spot = c.initial_spot * ratios
    for hit, state in ((False, "pre-KI"), (True, "post-KI")):
        curve = engine.evaluate(c.initial_spot, knocked_in=hit)
        crn = monte_carlo_greeks(
            c, m, n_paths=profile.greek_paths, seed=GREEK_SEED, batch_size=BATCH_SIZE,
            knocked_in=hit, bumps=bumps,
        )
        for name in ("value", *GREEK_NAMES):
            issue_greeks.append({
                "ki_state": state, "metric": name, "pde": getattr(curve, name)[0],
                "mc_crn": crn[name].mean, **estimate_columns(crn[name]),
            })
        for step in (0, 20, 125, 230):
            curve = engine.evaluate(spot, step=step, knocked_in=hit)
            spot_greeks.append(pd.DataFrame({
                "ki_state": state,
                "step": step,
                "years_remaining": c.maturity - c.schedule.times[step],
                "spot_ratio": ratios,
                **{name: getattr(curve, name) for name in ("value", *GREEK_NAMES)},
            }))
        state_kwargs = {"step": 20, "knocked_in": hit}
        at_spot = solution.value(spot, **state_kwargs)
        for fraction in (0.0025, 0.005, 0.01):
            ds = fraction * c.initial_spot
            gamma = (solution.value(spot + ds, **state_kwargs) - 2 * at_spot
                     + solution.value(spot - ds, **state_kwargs)) / ds**2
            bump_rows.append(pd.DataFrame({
                "ki_state": state, "step": 20, "spot_bump": fraction,
                "spot_ratio": ratios, "gamma": gamma,
            }))
    return {
        "issue_greeks": pd.DataFrame(issue_greeks),
        "spot_risk": pd.concat(spot_greeks, ignore_index=True),
        "gamma_bump": pd.concat(bump_rows, ignore_index=True),
        "parameter_risk": parameter_risk(c, m, profile.sweep_grid, bumps),
        "volatility_carry": volatility_carry(
            c, m, profile.sweep_grid, (0.09, 0.11, 0.13, 0.16, 0.19, 0.22, 0.25),
            (0.0, 0.025, 0.05, 0.075, 0.10, 0.11, 0.13, 0.15),
        ),
    }


def hedge_experiments(solution, n_paths):
    print("Running the self-financing hedge comparison", flush=True)
    c, m = solution.contract, solution.market
    paths = simulate_paths(c, m, n_paths, seed=HEDGE_SEED)
    payoffs = classify_payoffs(c, paths)
    examples = []
    for case in (Scenario.KI_NO_KO_LOSS, Scenario.KO_AFTER_KI):
        indices = np.flatnonzero(payoffs.scenario == case)
        if not len(indices):
            raise RuntimeError(f"no example for {case.name} in the simulated hedge panel")
        order = indices[np.argsort(payoffs.net_profit[indices], kind="stable")]
        examples.append(int(order[len(order) // 2]))
    labels = ("KI and terminal loss", "KO after KI")
    policies = {"Unhedged": None, "Daily": 1, "Weekly": 5}
    pnl, stats, traces = {}, [], []
    all_pnl = {"path": np.arange(n_paths), "scenario": payoffs.scenario,
               "settlement_step": payoffs.payment_step}
    bp = 10_000 / c.notional
    for strategy, period in policies.items():
        for cost in (0, 5, 20) if period else (0,):
            sample_trace = tuple(examples) if cost == 0 and period else ()
            result = delta_hedge(
                solution, paths, rebalance_every=period, cost_bps=cost, trace_indices=sample_trace
            )
            distribution = result.pnl_at_maturity * bp
            all_pnl[f"{strategy.lower()}_{cost}_bps"] = distribution
            p01, p05, median, p95, p99 = np.quantile(distribution, (0.01, 0.05, 0.5, 0.95, 0.99))
            stats.append({
                "strategy": strategy,
                "cost_bps": cost,
                "mean_maturity_bp": distribution.mean(),
                "std_maturity_bp": distribution.std(ddof=1),
                "p01_maturity_bp": p01,
                "p05_maturity_bp": p05,
                "median_maturity_bp": median,
                "p95_maturity_bp": p95,
                "p99_maturity_bp": p99,
                "mean_cost_pv_bp": result.costs_pv.mean() * bp,
                "rmse_replication_pv_bp": np.sqrt(np.mean(result.replication_error_pv**2)) * bp,
                "mean_turnover_pv_x_notional": result.turnover_pv.mean() / c.notional,
                "mean_routine_abs_trade_shares": result.mean_routine_abs_trade_shares,
                "mean_ki_trade_shares": np.nanmean(result.ki_trade_shares),
                "mean_ki_abs_trade_shares": np.nanmean(np.abs(result.ki_trade_shares)),
                "mean_ko_abs_liquidation_shares": np.nanmean(np.abs(result.ko_liquidation_shares)),
            })
            if cost == 0:
                pnl[strategy] = distribution
            for j, i in enumerate(sample_trace):
                step = np.arange(result.settlement_step[i] + 1)
                traces.append(pd.DataFrame({
                    "example": labels[j], "path": i, "strategy": strategy, "step": step,
                    "month": 12 * np.asarray(c.schedule.times)[step],
                    "spot": result.trace.spot[j, step], "shares": result.trace.shares[j, step],
                    "cash": result.trace.cash[j, step], "trade_shares": result.trace.share_trade[j, step],
                    "ki": step == payoffs.first_ki_step[i], "ko": step == payoffs.first_ko_step[i],
                }))
    return {
        "hedge_summary": pd.DataFrame(stats),
        "hedge_traces": pd.concat(traces, ignore_index=True),
        "hedge_pnl_samples": pd.DataFrame(all_pnl),
    }, pnl


def run(output: Path, profile: Profile, quick: bool):
    started = perf_counter()
    results, images = output / "results", output / "figures"
    results.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    tables, mc = pricing_experiments(profile)
    tables.update(convention_experiments(profile, mc))
    baseline, _ = benchmark_scenarios()
    c, m = baseline.contract, baseline.market
    solution = solve_pde(c, m, profile.risk_grid)
    tables.update(risk_experiments(solution, profile))
    hedge, pnl = hedge_experiments(solution, profile.hedge_paths)
    tables.update(hedge)
    cash = mc.payoffs.net_profit / c.notional
    n_tail = int(np.ceil(0.01 * len(cash)))
    tables["payoff_tail"] = pd.DataFrame([{
        "loss_probability": (cash < 0).mean(),
        "minimum": cash.min(), "maximum": cash.max(),
        "net_payoff_01_quantile": np.quantile(cash, 0.01),
        "mean_worst_01_fraction": np.partition(cash, n_tail - 1)[:n_tail].mean(),
    }])
    for name, table in tables.items():
        table.to_csv(results / f"{name}.csv", index=False, float_format="%.12g")

    print("Rendering research figures", flush=True)
    figures.research_style()
    figures.payoff_scenarios(images, c)
    figures.simulated_paths(images, c, simulate_paths(c, m, 35, seed=PLOT_SEED))
    figures.distributions(images, c, mc)
    figures.event_probabilities(images, c, mc)
    figures.convergence(images, tables["mc_convergence"], tables["pde_convergence"])
    figures.greek_profiles(images, c, tables["spot_risk"], tables["gamma_bump"])
    figures.hedge_paths(images, c, tables["hedge_traces"])
    figures.hedge_distribution(images, pnl, tables["hedge_summary"])
    figures.carry_heatmap(images, tables["volatility_carry"])
    figures.price_comparison(images, tables["benchmarks"])
    figures.parameter_sensitivities(images, tables["parameter_risk"])

    metadata = {
        "profile": "quick" if quick else "full",
        "settings": asdict(profile),
        "seeds": {"pricing": SEED, "greeks": GREEK_SEED, "hedging": HEDGE_SEED, "paths_figure": PLOT_SEED},
        "batch_size": BATCH_SIZE,
        "greek_bumps": asdict(GreekBumps()),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "libraries": {name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")},
        "runtime_seconds": perf_counter() - started,
    }
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(tables["benchmarks"][["scenario", "method", "net_pv", "report_net_pv"]].to_string(index=False))
    print(f"Wrote {len(tables)} tables and 15 figures to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT, help="Parent of results/ and figures/")
    parser.add_argument("--quick", action="store_true", help="Smaller simulations and PDE grids")
    args = parser.parse_args()
    profile = Profile(
        mc_paths=(5_000, 15_000, 40_000), space_steps=(160, 320, 640),
        greek_paths=30_000, hedge_paths=2500,
        risk_grid=PDEGrid(space_steps=640), sweep_grid=PDEGrid(space_steps=360),
    ) if args.quick else Profile()
    run(args.output.resolve(), profile, args.quick)


if __name__ == "__main__":
    main()
