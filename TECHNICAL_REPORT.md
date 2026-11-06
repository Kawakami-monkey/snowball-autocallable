# Pricing and hedging a discrete Snowball

## 1. Contract and cash-flow convention

The reference product is a one-year note on the CSI 500 with a 20% annual coupon, a down barrier at 85% of the issue reference $S_0$ and an up barrier at 103%. I use $S_0=1$ and notional $N=1$ for the experiments. Under the baseline calendar, KI is checked at trading closes $j=1,\ldots,252$ and KO at $j=21,42,\ldots,252$. Both model time and coupon accrual use $j/252$. Cinda's term sheet specifies actual days divided by 365 and the last trading close of each month, while its numerical tables specify 252 daily and 12 monthly observations. The exact dates of the numerical experiment are unspecified. The implementation also accepts supplied dates on ACT/365F and explicit grids with a separate coupon accrual clock. Cash settlement occurs at the observation close. [Cinda, pp. 4–8 and 16](https://bigdata-s3.wmcloud.com/researchreport/2021-07/b025f87ecb207bafca79ade86110b98d.pdf)

Let $K$ indicate a first scheduled KO with $S\ge H=1.03S_0$ and let $D$ indicate that $S<L=0.85S_0$ at a scheduled KI date before the contract settles. The settlement time $\tau$ is the first KO date if called, and $T$ otherwise. With coupon rate $c$ and accrued year fraction $a(t)$, the investor's net cash flow is

$$
X =
\begin{cases}
Nc\,a(\tau), & K,\\
Nc\,a(T), & K^c\cap D^c,\\
-N(1-S_T/S_0)^+, & K^c\cap D.
\end{cases}
$$

A KO can follow KI and pays the same accrued coupon as an otherwise identical KO without KI. The maturity put term vanishes when the underlying recovers to the issue reference or above. The five cases and their settlement times are illustrated with synthetic paths. [Payoff diagrams](figures/01_payoff_scenarios.png)

The net option value is $V_{\rm net}=\mathbb{E}^{\mathbb Q}[e^{-r\tau}X]$. The note's gross redemption is $N+X$, and its present value is $V_{\rm net}+N\mathbb{E}^{\mathbb Q}[e^{-r\tau}]$. If the investor deposits par at inception, the value after that outlay is $V_{\rm gross}-N$. The baseline PDE gives $V_{\rm gross}=1.03665214$ and $V_{\rm net}=0.05085573$. For the effective-carry terms, $V_{\rm gross}=0.94442355$ and $V_{\rm net}=-0.03932494$. [Net and gross values](results/benchmarks.csv)

## 2. Risk-neutral Monte Carlo

The model uses deterministic rate $r$, dividend yield $q$ and volatility $\sigma$. On every supplied interval of length $\Delta t$, an independent standard normal $Z$ gives the exact transition

$$
\log S_{t+\Delta t}=\log S_t+(r-q-\tfrac12\sigma^2)\Delta t+\sigma\sqrt{\Delta t}\,Z.
$$

The first event time is determined from each barrier's own index set. KI after the first KO is discarded. This simulation uses exact transitions at all monitored closes and carries no SDE time-stepping bias for the discrete contract. For $M$ paths on $n$ observation intervals and batch size $B$, working storage is $O(Bn+M)$ including the retained settlement outcomes. Pricing averages the discounted cash flows with a normal 95% interval $\hat V\pm 1.96s/\sqrt{M}$, where $s$ is the sample standard deviation. For antithetics, adjacent paths use shocks $(Z,-Z)$. Pair means are the independent observations used in the standard error.

On the 300,000-path baseline run the independent standard error is 1.806 bp of notional and the antithetic standard error is 1.812 bp for the same path count. In this run, antithetics reduce the number of normals generated and leave the sampling variance similar. The price path convergence series reuses the same seed across path counts, so its points share random draws. Its intervals shrink at the expected scale while the realized errors vary. [Path convergence and timings](results/mc_convergence.csv) · [Simulated paths](figures/02_simulated_paths.png)

## 3. Two-state Crank–Nicolson solver

Between fixing times, the clean continuation value $U_0(t,S)$ and the previously knocked-in value $U_1(t,S)$ each satisfy

$$
\partial_t U_h+\tfrac12\sigma^2 S^2\partial_{SS}U_h+(r-q)S\partial_SU_h-rU_h=0.
$$

The terminal values conditional on survival after applying the fixing are $U_0(T,S)=Nc\,a(T)$ and $U_1(T,S)=-N(1-S/S_0)^+$. To price gross redemption, add $N$ to both terminal values and to every call settlement. Backward stepping applies a fixing to the continuation values on its later side. At a KO date and spot $S\ge H$, replace both values with the call payment. At a KI date and spot $S<L$, replace $U_0$ with $U_1$ on paths that survive the KO check. Other nodes retain their continuation. The live surfaces in the API are conditional on survival immediately after the fixing. The before-fixing API applies these pointwise rules and the specified equality conventions.

Set $x=\log(S/S_0)$ and $\mu=r-q-\sigma^2/2$. Factor out the discount $e^{-r\Delta t}$ over each substep and discretize the remaining operator $\tfrac12\sigma^2\partial_{xx}+\mu\partial_x$. The centered spatial generator $A$ has off-diagonal coefficients

$$
a_-=\frac{\sigma^2}{2\Delta x^2}-\frac{\mu}{2\Delta x},
\qquad
a_+=\frac{\sigma^2}{2\Delta x^2}+\frac{\mu}{2\Delta x}
$$

and diagonal $-(a_-+a_+)$. A drift-directed one-sided stencil is used if a centered off-diagonal would be negative. With transformed boundary terms $b$, a Crank–Nicolson substep solves

$$
(I-\tfrac12\Delta t A)u_{\rm earlier}
  =(I+\tfrac12\Delta t A)u_{\rm later}+b
$$

for the transformed earlier value and then applies the exact discount. The implementation uses a banded tridiagonal solve. It keeps the explicit weights nonnegative by refining the time interval to satisfy $\Delta t(a_-+a_+)/2\le1$. At maturity and after a fixing projection, two backward-Euler half steps replace the first Crank–Nicolson step. At a barrier jump, a log cell averages the fractions on each side. When both barriers occupy one cell, the KO and KI fractions are disjoint and sum to at most one. This projection reduces sensitivity to a barrier moving across a node under grid refinement.

The upper boundary assumes the next scheduled call is certain at sufficiently high spot and discounts its payment to the current step. After the last call check, it uses the surviving high-spot maturity payoff. At the lower boundary the downside leg uses the deep-in-the-money approximation $N[e^{-r(T-t)}-(S/S_0)e^{-q(T-t)}]$. The clean state uses the same low-spot exposure if another KI check remains. Otherwise it retains its maturity coupon. The gross calculation adds the discounted principal. The log-domain width is configurable. Refining the time and space grids separately and widening the domain at fixed spacing test these boundary and discretization choices. [PDE convergence](results/pde_convergence.csv)

Cinda's explicit recurrence on p. 11 prints the left neighbor twice. The right-neighbor coefficient multiplies $V_{j+1}$. The implicit presentation uses inverse-matrix notation. The scheme here derives and solves the linear system directly, with each event applied on its scheduled date. [Cinda, pp. 10–16](https://bigdata-s3.wmcloud.com/researchreport/2021-07/b025f87ecb207bafca79ade86110b98d.pdf)

## 4. Four-leg decomposition

Let $P=N(1-S_T/S_0)^+$. The maturity loss is $-P\,\mathbf{1}_{K^c\cap D}$. Splitting the no-KO put across KI histories gives

$$
-P\,\mathbf{1}_{K^c\cap D}=-P\,\mathbf{1}_{K^c}+P\,\mathbf{1}_{K^c\cap D^c}.
$$

Combining this equality with the two coupon cases gives $V_{\rm net}=V_1+V_2-V_3+V_4$. The terms are present values of the following cash flows.

| Leg | Cash flow and date | PDE | MC | Cinda |
| --- | --- | ---: | ---: | ---: |
| $V_1$ | $Nc\,a(\tau)\mathbf{1}_{K}$ at $\tau$ | 0.042575 | 0.042552 | 0.04213 |
| $V_2$ | $Nc\,a(T)\mathbf{1}_{K^c\cap D^c}$ at $T$ | 0.026497 | 0.026436 | 0.02633 |
| $V_3$ | $P\mathbf{1}_{K^c}$ at $T$, short in the portfolio | 0.025594 | 0.025619 | 0.02501 |
| $V_4$ | $P\mathbf{1}_{K^c\cap D^c}$ at $T$ | 0.007378 | 0.007342 | 0.00796 |
| Net | $V_1+V_2-V_3+V_4$ | 0.050856 | 0.050711 | 0.05141 |

The report labels $V_1$ an up-and-out call. Its stated payoff is the cash coupon paid at first KO, which determines the event condition here. The four-leg PDE is propagated with its own single-state survival conditions and agrees with the two-state net PDE to rounding error in both benchmark cases. The Monte Carlo leg identity also holds path by path. [Cinda, pp. 13–16](https://bigdata-s3.wmcloud.com/researchreport/2021-07/b025f87ecb207bafca79ade86110b98d.pdf) · [Leg estimates and intervals](results/decomposition.csv)

## 5. Validation and unresolved benchmark differences

The baseline MC–PDE gap is 1.45 bp, or 0.80 Monte Carlo standard errors. The effective-carry gap is 1.68 bp, or 0.64 standard errors. The absolute differences from the published baseline MC, baseline PDE and effective-carry PDE values are 1.49, 5.54 and 20.15 bp. The corresponding relative errors are 0.29%, 1.08% and 5.40%. The published baseline MC value is inside the simulation's 95% interval. The published PDE references are above the corresponding intervals. [Benchmark values and errors](results/benchmarks.csv) · [Cross-method checks](results/cross_method.csv)

The finest production grid uses 2,560 log-spot intervals over $[-1,1]$. Increasing the grid from 1,280 to 2,560 changes the baseline and carry values by 0.113 and 0.013 bp. At 400 intervals, refining from 2 to 32 time subdivisions changes the baseline PV by 0.883 bp. Widening the baseline half-domain from 0.65 to 1.35 at fixed spacing changes its issue PV by less than $10^{-12}$ for the tested grid. Runtime is recorded per solve, including the imposed time refinement. The report gives a ten-run MC average on unspecified hardware, while the archived experiment reports wall times from one host. [PDE grid experiment](results/pde_convergence.csv) · [MC timings](results/mc_convergence.csv) · [Run environment](results/metadata.json)

| Baseline metric | Simulated | Cinda |
| --- | ---: | ---: |
| KO probability | 73.66% | 73.79% |
| No KI and no KO | 13.62% | 13.69% |
| KI and no KO | 12.72% | 12.52% |
| Terminal loss probability | 12.53% | 12.32% |
| Mean elapsed months conditional on KO | 3.518 | 2.59 |
| Mean life in months | 5.752 | 5.74 |

The timing identity for a one-year contract is $E[12\tau]=p_{\rm KO}E[12\tau\mid K]+(1-p_{\rm KO})12$. The reported 73.79% and 2.59 imply 5.056 months. Using 5.74 as the mean life instead implies a conditional call time of 3.516 months, close to the simulated 3.518. A zero-indexed month number would reduce the simulated conditional mean to 2.518 and could explain most of the one-month gap. The source's exact convention is unresolved. Also, the reported average KO coupon of 4.32% and KO probability give an upper bound of 0.031877 for $V_1$ under positive discount rates. The reported $V_1$ is 0.04213. [Cinda, pp. 8–9 and 16](https://bigdata-s3.wmcloud.com/researchreport/2021-07/b025f87ecb207bafca79ade86110b98d.pdf) · [Event estimates and uncertainty](results/event_statistics.csv) · [Timing identities](results/report_timing_identities.csv)

Shifting all interim call dates by one trading day moves the baseline PDE by −2.29 bp for earlier calls and +2.43 bp for later calls. Omitting the final call check reduces KO probability to 72.31% and lowers PV by 2.14 bp on the same grid. These are controlled alternatives to the baseline schedule. The precise source calendar and PDE resolution remain unknown. Under the stated inputs, the converged PDE and MC confidence intervals support the independent values for both scenarios. [Calendar experiment](results/calendar_diagnostics.csv)

The baseline payoff has positive atoms at the accrued call coupons and at the full coupon, with a continuous loss tail. Across 300,000 risk-neutral paths the 1% payoff quantile is −23.99% and the mean of the worst 1% is −27.31%. These statistics describe the undiscounted model cash flow. [Payoff tail](figures/04_payoff_downside_tail.png) · [Tail statistics](results/payoff_tail.csv)

## 6. Greeks and state dependence

The PDE uses centered absolute spot bumps of $0.005S_0$ for Delta and Gamma, 0.005 absolute volatility bumps for Vega, and 0.001 rate bumps for Rho. Bumps preserve issue levels and accrued coupon dates. Calendar Theta is the local $\partial_t U$ at fixed spot, fixed KI state and fixed schedule. It follows from the PDE between fixings. A scheduled fixing can produce a discrete value jump, in which case a roll across the fixing includes an event effect. The Monte Carlo time roll shortens only the first remaining interval by a quarter of that interval. All Monte Carlo bumps reuse path shocks, and intervals are computed on the pathwise finite-difference samples with pair-based standard errors.

| Issue-reference Greek | Pre-KI | Post-KI conditional | Units |
| --- | ---: | ---: | --- |
| Delta | −0.291 | +0.238 | Shares per unit contract |
| Gamma | −12.60 | −10.48 | PV per spot unit squared |
| Vega | −94.75 | −29.09 | bp per volatility point |
| Theta | +4.63 | +3.25 | bp per synthetic trading day |
| Rho | +19.17 | +39.65 | bp per 100 bp rate increase |

At spot $0.90S_0$, a higher spot reduces the chance of a downside settlement and the model gives positive Delta of about 1.56 before KI and 1.05 after KI. At $S_0$, higher spot also makes the short first-call accrual more likely relative to staying alive for a longer coupon. The clean-state Delta is negative there. After KI, the recovery and downside-loss channels strengthen the positive exposure at that spot. The sign and slope change with the next fixing date and accrued coupon. At the baseline reference spot Vega is negative in both states. Deeper after KI, extra dispersion can increase the chance of recovery or a call, and the model gives positive Vega of about 11 bp per volatility point at $0.80S_0$. Theta also changes sign across spot. Curves conditioned on a clean state below KI at a fixing are hypothetical marks for that history. [Delta profiles](figures/08_delta_vs_spot.png) · [Vega, Theta and Rho](figures/10_vega_theta_rho.png)

One trading day before the first call, peak absolute clean Gamma across the plotted spot range is about 147 with a 0.25% spot bump and 113 with a 1% bump. A central difference magnifies price errors by the inverse squared bump. In Monte Carlo, paths that switch settlement cases across bumps add further dispersion. At issue, the pre-KI MC Gamma is −18.81 with standard error 5.12, while the PDE estimate is −12.60. The separate, smaller CRN run's unbumped prices sit above the PDE values by 2.81 and 2.34 standard errors for the two states. The main 300,000-path comparisons use a separate seed. [Gamma bump plot](figures/09_gamma_vs_spot.png) · [CRN estimates](results/issue_greeks.csv)

Controlled one-factor sweeps show that Delta at reference spot turns positive as volatility rises from 9% to 23%, and as the KI level rises from 75% to 93%. Increasing the coupon from 10% to 30% raises value and makes the clean-state Delta more negative in this experiment. The higher KO barrier increases the clean-state PV over the sampled levels. After KI, varying the KI level leaves the conditional value invariant by construction. Tenor sweeps use new issue calendars, while the fixed-contract spot curves keep the original dates and vary remaining life. Each sweep records the full vector of Greeks in both states. [One-factor figures](figures/15_parameter_sensitivities.png) · [Parameter and Greek results](results/parameter_risk.csv)

## 7. Self-financing issuer hedge

The issuer shorts the investor claim at its model value $\pi$ and holds the investor Delta in spot shares when rebalancing. Let $h_j$ denote shares after trading and $B_j$ the cash account. Between rebalances continuous dividends are reinvested, so $h^-_j=h_{j-1}e^{q\Delta t}$. Cash accrues at $r$. At a scheduled rebalance the change $\Delta h_j=h_j-h^-_j$ costs $S_j\Delta h_j+\kappa S_j|\Delta h_j|$, where $\kappa$ is the proportional fee. Thus

$$
B_j=B_{j-1}e^{r\Delta t}-S_j\Delta h_j-\kappa S_j|\Delta h_j|.
$$

The initial trade uses $B^-_0=\pi$ and $h^-_0=0$. At settlement, set $h_j=0$, charge the liquidation cost and pay the investor cash flow. The remaining issuer cash is the settlement P&L. Discounting it to issue measures replication error before costs when $\kappa=0$. In the cost runs, adding back discounted fees recovers the frictionless replication residual for the same trading rule. P&L is carried forward from each settlement to the common maturity $T$ when comparing distributions. Cash-only positions, daily rebalancing and every-fifth-observation rebalancing use the same 10,000 paths. Each strategy processes barrier events on the contract schedule.

| Strategy | Frictionless P&L SD, bp at $T$ | Frictionless replication RMSE, PV bp | Mean PV turnover / notional | Mean P&L at $T$, 20 bp fee |
| --- | ---: | ---: | ---: | ---: |
| Unhedged | 1,025 | 995 | 0.00 | +5 bp |
| Daily | 172 | 167 | 16.08 | −331 bp |
| Weekly | 325 | 316 | 7.89 | −159 bp |

At the first KI on a live path the daily strategy trades 3.06 shares on average in absolute value, compared with 0.13 shares on other non-settlement rebalance dates. The model can require a sharp reduction in the hedge once the downside state is fixed. A weekly hedge may carry its earlier exposure until the next scheduled rebalance. Liquidation takes place on every actual settlement date. The 20 bp fee consumes an average of 322 PV bp for daily hedging and 158 PV bp for weekly hedging. These outcomes depend on the assumed model, risk-neutral spot paths and the ability to execute at the fixing price. [Hedge paths](figures/11_hedge_positions.png) · [P&L distributions](figures/12_hedging_pnl.png) · [Hedge statistics](results/hedge_summary.csv)

## 8. Effective carry and model limits

Cinda's secondary terms use an 18.82% coupon, 19% volatility, 3% risk-free rate and 11% effective dividend yield. The report combines an estimate of equity-index futures discount and a return on capital released by futures margining to motivate the carry input. Within this model, $q$ changes the forward curve to $S_0 e^{(r-q)T}$ while the discount rate stays at $r$. At these terms the PDE moves from −0.03672 at $q=10.5\%$ to −0.03933 at 11% and −0.04198 at 11.5%. This quantifies how a half-point shift in effective carry changes PV by an amount comparable to the gap from the reported −0.03731. The benchmark uses the report's stated 11% without calibration. [Cinda, pp. 19–20](https://bigdata-s3.wmcloud.com/researchreport/2021-07/b025f87ecb207bafca79ade86110b98d.pdf) · [Effective-carry sweep](results/effective_carry_diagnostics.csv)

A dealer's economics depend on actual dividends, financing curves, the futures basis term structure, daily variation margin, collateral terms, rolls, execution and basis volatility. The scalar proxy changes risk-neutral spot dynamics. Explicit futures hedging would track the futures contract and its margin cash flows separately. The project uses constant Black–Scholes inputs and synthetic spot paths, with a frictionless funding account and a proportional execution fee. Its valuation and replication experiments exclude market calibration, stochastic volatility and jumps, credit, liquidity, and asynchronous settlement. [Volatility and carry surface](figures/13_volatility_carry.png)
