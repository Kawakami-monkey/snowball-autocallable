"""Publication figures for the synthetic research experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, PercentFormatter

from .payoff import Scenario, classify_payoffs
from .validation import REPORT_PRICES

INK, TEAL, RED, GOLD, INDIGO, GRAY = "#17324d", "#087e8b", "#b64e34", "#b58019", "#5958a6", "#708090"
STATE_COLORS = {"pre-KI": TEAL, "post-KI": INDIGO}
CASES = (
    "KO without prior KI", "Neither barrier", "KI, recovered at maturity",
    "KI, loss at maturity", "KI followed by KO",
)
CASE_COLORS = (TEAL, INK, GOLD, RED, INDIGO)


def research_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#bbc6ce",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.6,
        "grid.color": "#a4b5c4",
        "legend.frameon": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "savefig.dpi": 200,
    })


def _figure(title, *shape, size=(8.4, 4.3), **kwargs):
    fig, axes = plt.subplots(*shape, figsize=size, layout="constrained", **kwargs)
    fig.suptitle(title, x=0.02, ha="left", weight="bold", fontsize=14)
    return fig, axes


def _save(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", facecolor="white")
    plt.close(fig)


def _barriers(ax, contract, *, vertical=False):
    line = ax.axvline if vertical else ax.axhline
    line(contract.ki_barrier / contract.initial_spot, color=RED, lw=1, ls="--", alpha=0.8)
    line(contract.ko_barrier / contract.initial_spot, color=TEAL, lw=1, ls="--", alpha=0.8)


def payoff_scenarios(output, contract):
    times = np.array(contract.schedule.times)
    anchors = [
        ([0, 3, 6, 12], [1, 0.95, 1.06, 0.90]),
        ([0, 4, 8, 12], [1, 0.94, 1.01, 0.99]),
        ([0, 3, 7, 12], [1, 0.82, 0.93, 1.01]),
        ([0, 4, 8, 12], [1, 0.82, 0.96, 0.90]),
        ([0, 3, 8, 10, 12], [1, 0.82, 0.94, 1.05, 0.93]),
    ]
    months = times * 12
    paths = np.stack([np.interp(months, x, y) for x, y in anchors]) * contract.initial_spot
    payoffs = classify_payoffs(contract, paths)
    np.testing.assert_array_equal(payoffs.scenario, list(Scenario))
    fig, axes = _figure("Five possible settlements of the same Snowball", 2, 3,
                        size=(11.2, 5.6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat[:5]):
        end = payoffs.payment_step[i]
        values = paths[i] / contract.initial_spot
        _barriers(ax, contract)
        ax.plot(months[:end + 1], values[:end + 1], color=CASE_COLORS[i], lw=2)
        observations = np.array([k for k in contract.schedule.ko_steps if k <= end])
        ax.scatter(months[observations], values[observations], s=11, color=GRAY, zorder=3)
        ax.scatter(months[end], values[end], s=35, color=CASE_COLORS[i], zorder=4)
        ax.set(title=f"{i + 1}   {CASES[i]}", ylim=(0.79, 1.09), xlim=(0, 12.2),
               xticks=(0, 3, 6, 9, 12))
        net, gross = payoffs.net_profit[i] / contract.notional, payoffs.gross_redemption[i] / contract.notional
        ax.text(0.03, 0.06, f"Net {net:+.1%}    Redemption {gross:.1%}",
                transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", edgecolor="none"))
    legend_ax = axes.flat[5]
    legend_ax.axis("off")
    legend_ax.legend(handles=[
        Line2D([0], [0], color=RED, ls="--", label="Daily knock-in level"),
        Line2D([0], [0], color=TEAL, ls="--", label="Monthly knock-out level"),
        Line2D([0], [0], color=GRAY, marker="o", lw=0, label="Call observation"),
        Line2D([0], [0], color=INK, marker="o", lw=0, label="Cash settlement"),
    ], loc="center left")
    fig.supxlabel("Months since issue")
    fig.supylabel("Spot / issue reference")
    _save(fig, output, "01_payoff_scenarios")


def simulated_paths(output, contract, paths):
    payoffs = classify_payoffs(contract, paths)
    months = np.array(contract.schedule.times) * 12
    fig, ax = _figure("Risk-neutral paths stop at cash settlement")
    _barriers(ax, contract)
    for i, end in enumerate(payoffs.payment_step):
        color = CASE_COLORS[payoffs.scenario[i] - 1]
        ax.plot(months[:end + 1], paths[i, :end + 1] / contract.initial_spot,
                color=color, alpha=0.62, lw=1)
        ax.scatter(months[end], paths[i, end] / contract.initial_spot, color=color, s=13)
    ax.set(xlabel="Months since issue", ylabel="Spot / issue reference",
           xticks=np.arange(0, 13), xlim=(0, 12.25))
    fig.legend(handles=[
        Line2D([0], [0], color=color, label=label) for color, label in zip(CASE_COLORS, CASES)
    ], loc="outside lower center", ncols=3, fontsize=9)
    _save(fig, output, "02_simulated_paths")


def distributions(output, contract, mc):
    pv = mc.discounted / contract.notional
    fig, ax = _figure("Distribution of discounted investor cash flows")
    ax.hist(pv, bins=140, weights=np.ones(len(pv)) / len(pv), color=TEAL, alpha=0.82)
    ax.axvline(pv.mean(), color=INK, lw=1.6, label=f"Mean PV {pv.mean():.3%}")
    ax.set(xlabel="Discounted net cash flow / notional", ylabel="Probability per bin")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    _save(fig, output, "03_discounted_payoff_distribution")

    cash = mc.payoffs.net_profit / contract.notional
    worst = np.sort(cash)
    quantile = np.quantile(cash, 0.01)
    tail_mean = worst[:int(np.ceil(0.01 * len(cash)))].mean()
    fig, (ax, tail) = _figure("Investor payoff and downside tail", 1, 2, size=(10.2, 4.1))
    ax.hist(cash, bins=120, color=INK, weights=np.ones(len(cash)) / len(cash))
    ax.set(xlabel="Net cash flow / notional", ylabel="Probability per bin")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    loss_count = np.searchsorted(worst, 0.0, side="left")
    loss_prob = np.arange(1, loss_count + 1) / len(cash)
    tail.plot(worst[:loss_count], loss_prob, color=RED, lw=1.7)
    tail.axvline(quantile, color=GRAY, ls="--", label=f"1% quantile {quantile:.1%}")
    tail.axvline(tail_mean, color=RED, ls=":", label=f"Worst 1% mean {tail_mean:.1%}")
    tail.set(xlabel="Net cash flow threshold", ylabel="Unconditional probability ≤ threshold",
             yscale="log", ylim=(0.0001, 0.2))
    tail.xaxis.set_major_formatter(PercentFormatter(1))
    tail.yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
    tail.legend(loc="lower right", fontsize=9)
    _save(fig, output, "04_payoff_downside_tail")


def event_probabilities(output, contract, mc):
    payoffs = mc.payoffs
    fig, (ax, dates) = _figure("Barrier outcomes and the timing of the first call", 1, 2, size=(11, 4.0))
    positions = np.arange(5)
    probs = np.bincount(payoffs.scenario, minlength=6)[1:] / len(payoffs.scenario)
    bars = ax.barh(positions, probs, color=CASE_COLORS)
    ax.bar_label(bars, [f"{p:.2%}" for p in probs], padding=4)
    ax.set(yticks=positions, yticklabels=CASES, xlim=(0, max(probs) * 1.18),
           xlabel="Probability of path outcome")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.invert_yaxis()
    ko_steps = list(contract.schedule.ko_steps)
    calls = np.array([(payoffs.first_ko_step == i).mean() for i in ko_steps])
    labels = np.arange(1, len(ko_steps) + 1)
    dates.bar(labels, calls, color=TEAL, width=0.72)
    dates.set(xlabel="Monthly call observation", ylabel="Unconditional first-call probability",
              xticks=labels)
    dates.yaxis.set_major_formatter(PercentFormatter(1))
    _save(fig, output, "05_event_probabilities")


def convergence(output, mc, pde):
    fig, ax = _figure("Monte Carlo path convergence")
    iid = mc.loc[~mc.antithetic]
    paired = mc.loc[mc.antithetic]
    for frame, color, label, offset in (
        (iid, TEAL, "Independent draws, 95% CI", 1),
        (paired, INDIGO, "Antithetic pairs, 95% CI", 1.07),
    ):
        ax.errorbar(frame.n_paths * offset, frame.price * 10_000,
                    yerr=(frame.ci_high - frame.price) * 10_000,
                    marker="o", color=color, lw=1.6, capsize=4, label=label)
    ax.axhline(iid.reference_pde.iloc[0] * 10_000, color=INK, ls="--", label="Finest PDE")
    ax.axhline(REPORT_PRICES["baseline", "Monte Carlo"] * 10_000,
               color=GRAY, ls=":", label="Cinda MC")
    ax.set(xscale="log", xlabel="Simulated paths", ylabel="Net PV (bp of notional)")
    ax.set_xticks(iid.n_paths, [f"{n / 1000:g}k" for n in iid.n_paths])
    ax.legend(ncols=2, loc="best", fontsize=9)
    _save(fig, output, "06_mc_convergence")

    fig, (ax, time) = _figure("Crank–Nicolson grid convergence", 1, 2, size=(10.3, 4.1))
    spatial = pde.loc[pde.study == "space"]
    for (label, group), color in zip(spatial.groupby("scenario", sort=False), (TEAL, INDIGO)):
        ax.plot(group.space_steps, group.signed_error * 10_000, marker="o",
                label=label.capitalize(), color=color)
    ax.axhline(0, color=GRAY, ls=":")
    ax.set(xscale="log", xlabel="Log-spot intervals", ylabel="PV minus finest space grid (bp)")
    base = spatial.loc[spatial.scenario == "baseline"]
    ax.set_xticks(base.space_steps, [str(n) for n in base.space_steps])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend()
    temporal = pde.loc[pde.study == "time"]
    time.plot(temporal.max_effective_substeps, temporal.signed_error * 10_000,
              marker="o", color=TEAL)
    time.axhline(0, color=GRAY, ls=":")
    time.set(xscale="log", xlabel="Time subdivisions per observation interval",
             ylabel="PV minus finest time grid (bp)")
    time.set_xticks(temporal.max_effective_substeps, [str(n) for n in temporal.max_effective_substeps])
    time.xaxis.set_minor_formatter(NullFormatter())
    _save(fig, output, "07_pde_convergence")


def greek_profiles(output, contract, spot_risk, bump_risk):
    fig, axes = _figure("Delta depends on spot, remaining life and KI history", 1, 2,
                        size=(10.6, 4.1), sharey=True)
    for (state, frame), ax in zip(spot_risk.groupby("ki_state", sort=False), axes):
        _barriers(ax, contract, vertical=True)
        for (step, curve), color in zip(frame.groupby("step", sort=False), (INK, TEAL, GOLD, INDIGO)):
            remaining = (contract.schedule.n_steps - step) / 21
            ax.plot(curve.spot_ratio, curve.delta, lw=1.7, color=color,
                    label=f"{remaining:.2f} months left" if step else "At issue")
        ax.set(title=state, xlabel="Spot / issue reference")
        ax.axhline(0, color=GRAY, lw=0.75)
    axes[0].set_ylabel("Investor Delta (shares per contract)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=9)
    _save(fig, output, "08_delta_vs_spot")

    fig, axes = _figure("Gamma one trading day before the first call", 1, 2,
                        size=(10.6, 4.1), sharey=True)
    for (state, frame), ax in zip(bump_risk.groupby("ki_state", sort=False), axes):
        _barriers(ax, contract, vertical=True)
        for (bump, curve), color in zip(frame.groupby("spot_bump", sort=False), (INDIGO, TEAL, GOLD)):
            ax.plot(curve.spot_ratio, curve.gamma, color=color, lw=1.5, label=f"{bump:.2%} spot bump")
        ax.axhline(0, color=GRAY, lw=0.75)
        ax.set(title=state, xlabel="Spot / issue reference")
    axes[0].set_ylabel("Investor Gamma (value / spot²)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=9)
    _save(fig, output, "09_gamma_vs_spot")

    fig, axes = _figure("Volatility, calendar time and rate sensitivity at issue", 1, 3,
                        size=(11.1, 3.65))
    issue = spot_risk.loc[spot_risk.step == 0]
    for ax, key, factor, label in zip(
        axes, ("vega", "theta", "rho"), (100, 10_000 / 252, 100),
        ("Vega (bp per vol point)", "Theta (bp per trading day)", "Rho (bp per 100 bp)"),
    ):
        _barriers(ax, contract, vertical=True)
        for state, group in issue.groupby("ki_state", sort=False):
            ax.plot(group.spot_ratio, factor * group[key], color=STATE_COLORS[state], label=state)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set(xlabel="Spot / issue reference", ylabel=label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    _save(fig, output, "10_vega_theta_rho")


def hedge_paths(output, contract, traces):
    fig, axes = _figure("Issuer inventory around discrete barrier events", 2, 2,
                        size=(10.5, 5.5), sharex="col")
    for i, (path, frame) in enumerate(traces.groupby("example", sort=False)):
        daily = frame.loc[frame.strategy == "Daily"]
        top, bottom = axes[:, i]
        _barriers(top, contract)
        top.plot(daily.month, daily.spot / contract.initial_spot, color=INK, lw=1.45)
        top.set(title=path, ylabel="Spot / issue reference")
        for (strategy, group), color in zip(frame.groupby("strategy", sort=False), (TEAL, INDIGO)):
            bottom.step(group.month, group.shares, where="post", color=color, label=strategy, lw=1.35)
        for flag, color, label in (("ki", RED, "First KI"), ("ko", TEAL, "First KO")):
            event = daily.loc[daily[flag], "month"]
            if len(event):
                for ax in (top, bottom):
                    ax.axvline(event.iloc[0], color=color, lw=1.1, ls=":", label=label)
        bottom.set(xlabel="Months since issue", ylabel="Hedge shares")
    fig.legend(handles=[
        Line2D([0], [0], color=TEAL, label="Daily"),
        Line2D([0], [0], color=INDIGO, label="Weekly"),
        Line2D([0], [0], color=RED, ls=":", label="First KI"),
        Line2D([0], [0], color=TEAL, ls=":", label="First KO"),
    ], loc="outside lower center", ncols=4)
    _save(fig, output, "11_hedge_positions")


def hedge_distribution(output, pnl, summary):
    fig, (ax, costs) = _figure("Issuer hedging P&L at a common maturity", 1, 2, size=(11.1, 4.25))
    bins = np.linspace(min(x.min() for x in pnl.values()), max(x.max() for x in pnl.values()), 105)
    for (label, values), color in zip(pnl.items(), (GRAY, TEAL, INDIGO)):
        ax.hist(values, bins=bins, histtype="step", lw=1.6, density=True, color=color, label=label)
    ax.set(xlabel="Maturity-equivalent P&L (bp)", ylabel="Probability density",
           title="Frictionless strategies")
    ax.legend()
    for (strategy, group), color, shift in zip(
        summary.loc[summary.strategy != "Unhedged"].groupby("strategy", sort=False),
        (TEAL, INDIGO), (-0.15, 0.15),
    ):
        y = np.arange(len(group)) + shift
        costs.hlines(y, group.p05_maturity_bp, group.p95_maturity_bp, color=color, lw=2)
        costs.scatter(group.median_maturity_bp, y, color=color, label=strategy)
    costs.axvline(0, color=GRAY, ls=":")
    levels = sorted(summary.cost_bps.unique())
    costs.set(xlabel="Maturity-equivalent P&L (bp)", ylabel="Cost per trade",
              yticks=np.arange(len(levels)), yticklabels=[f"{x:g} bp" for x in levels],
              title="Median and 5th–95th percentiles")
    costs.legend()
    _save(fig, output, "12_hedging_pnl")


def carry_heatmap(output, sensitivity):
    fig, axes = _figure("Volatility and effective carry at a fixed 20% coupon", 1, 2,
                        size=(10.5, 4.2), sharex=True, sharey=True)
    price_columns = ("pre_ki_price", "post_ki_price")
    prices = 100 * sensitivity[list(price_columns)]
    norm = TwoSlopeNorm(vmin=prices.min().min(), vcenter=0, vmax=prices.max().max())
    for column, state, ax in zip(price_columns, STATE_COLORS, axes):
        panel = sensitivity.pivot(index="dividend", columns="volatility", values=column)
        vol, q, pv = 100 * panel.columns.to_numpy(), 100 * panel.index.to_numpy(), 100 * panel.to_numpy()
        mesh = ax.contourf(vol, q, pv, levels=np.linspace(norm.vmin, norm.vmax, 23),
                           cmap="RdBu", norm=norm, extend="neither")
        ax.contour(vol, q, pv, levels=[0], colors=INK, linewidths=1.25)
        ax.scatter(13, 0, marker="o", s=40, facecolors="white", edgecolors=INK, clip_on=False)
        ax.set(title=state, xlabel="Volatility (%)")
    axes[0].set_ylabel("Effective dividend yield (%)")
    fig.colorbar(mesh, ax=axes, label="Net PV (% of notional)")
    _save(fig, output, "13_volatility_carry")


def price_comparison(output, benchmarks):
    fig, axes = _figure("Pricing engines against the Cinda benchmarks", 1, 2,
                        size=(11.4, 4.0))
    for (scenario, frame), ax in zip(benchmarks.groupby("scenario", sort=False), axes):
        mc, pde = frame.iloc[0], frame.iloc[1]
        ax.errorbar(mc.net_pv * 10_000, 0, xerr=(mc.ci_high - mc.net_pv) * 10_000,
                    marker="o", capsize=4, color=TEAL, lw=1.7)
        ax.scatter(pde.net_pv * 10_000, 1, marker="D", color=INDIGO, zorder=3)
        labels = ["Our Monte Carlo", "Our PDE"]
        for row in frame.dropna(subset=["report_net_pv"]).itertuples():
            ax.scatter(row.report_net_pv * 10_000, len(labels), marker="s", facecolor="none",
                       edgecolor=GRAY, s=43)
            labels.append(f"Cinda {row.method}")
        ax.set(title=scenario.capitalize(), xlabel="Net PV (bp of notional)",
               yticks=np.arange(len(labels)), yticklabels=labels, ylim=(len(labels) - 0.55, -0.55))
        ax.margins(x=0.12)
    _save(fig, output, "14_mc_pde_comparison")


def parameter_sensitivities(output, sensitivity):
    labels = {
        "maturity": ("Tenor", "Years", 1),
        "volatility": ("Volatility", "Annualized (%)", 100),
        "ki_ratio": ("Knock-in barrier", "% of reference", 100),
        "ko_ratio": ("Knock-out barrier", "% of reference", 100),
        "coupon": ("Coupon", "Annualized (%)", 100),
        "dividend": ("Dividend yield", "Annualized (%)", 100),
    }
    fig, axes = _figure("One-factor valuation sweeps at the reference spot", 2, 3,
                        size=(10.7, 6.25))
    for (parameter, panel), ax in zip(sensitivity.groupby("parameter", sort=False), axes.flat):
        title, label, scale = labels[parameter]
        for state, group in panel.groupby("ki_state", sort=False):
            ax.plot(group.level * scale, group.value * 100, marker="o", ms=3,
                    color=STATE_COLORS[state], label=state)
        ax.axhline(0, color=GRAY, lw=0.75)
        ax.set(title=title, xlabel=label, ylabel="Net PV (% of notional)")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=2)
    _save(fig, output, "15_parameter_sensitivities")
