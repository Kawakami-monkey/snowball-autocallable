"""Controlled one-factor risk sweeps and a joint volatility/carry experiment."""

from dataclasses import replace

import pandas as pd

from .contract import Market, ObservationSchedule, Snowball
from .greeks import GREEK_NAMES, GreekBumps, PDERisk
from .pde import PDEGrid, solve_pde


def parameter_risk(
    contract: Snowball, market: Market, grid: PDEGrid, bumps: GreekBumps = GreekBumps()
) -> pd.DataFrame:
    """Hold other inputs fixed in each sweep. Tenor uses new synthetic issue calendars."""
    scenarios = {
        "maturity": (0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
        "volatility": (0.09, 0.11, 0.13, 0.16, 0.19, 0.23),
        "ki_ratio": (0.75, 0.8, 0.85, 0.88, 0.9, 0.93),
        "ko_ratio": (1.01, 1.03, 1.05, 1.07, 1.09, 1.11),
        "coupon": (0.10, 0.15, 0.1882, 0.20, 0.25, 0.30),
        "dividend": (0.0, 0.025, 0.05, 0.075, 0.11, 0.15),
    }
    rows = []
    for parameter, values in scenarios.items():
        for level in values:
            if parameter == "maturity":
                terms = replace(contract, schedule=ObservationSchedule.regular(level))
                inputs = market
            elif parameter in ("dividend", "volatility"):
                terms, inputs = contract, replace(market, **{parameter: level})
            else:
                terms, inputs = replace(contract, **{parameter: level}), market
            risk = PDERisk(solve_pde(terms, inputs, grid), bumps)
            for knocked_in in (False, True):
                curve = risk.evaluate(terms.initial_spot, knocked_in=knocked_in)
                rows.append({
                    "parameter": parameter,
                    "level": level,
                    "ki_state": "post-KI" if knocked_in else "pre-KI",
                    **{name: float(getattr(curve, name)[0]) for name in ("value", *GREEK_NAMES)},
                })
    return pd.DataFrame(rows)


def volatility_carry(
    contract: Snowball, market: Market, grid: PDEGrid, volatilities, dividends
) -> pd.DataFrame:
    rows = []
    for vol in volatilities:
        for q in dividends:
            solution = solve_pde(contract, replace(market, volatility=vol, dividend=q), grid)
            rows.append({
                "volatility": vol,
                "dividend": q,
                "pre_ki_price": solution.price,
                "post_ki_price": float(solution.value(knocked_in=True)),
            })
    return pd.DataFrame(rows)
