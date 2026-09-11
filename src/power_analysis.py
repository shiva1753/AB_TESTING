"""
power_analysis.py
=================
Phase 6 — Power Analysis

Answers three questions:
  1. How many users did we NEED for a reliable experiment?
  2. How much power did our actual experiment have?
  3. What is the Minimum Detectable Effect (MDE) for our sample?
"""

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
import warnings
warnings.filterwarnings("ignore")


def required_sample_size(
    baseline_cvr: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Calculate the required sample size per group for a two-proportion z-test.

    Parameters
    ----------
    baseline_cvr : float  — control group conversion rate (e.g. 0.125)
    mde          : float  — minimum detectable effect (absolute, e.g. 0.02 = 2pp)
    alpha        : float  — significance level (default 0.05)
    power        : float  — desired statistical power (default 0.80)

    Returns required n per group and total n.
    """
    p1 = baseline_cvr
    p2 = baseline_cvr + mde

    # Effect size (Cohen's h for proportions)
    effect_size = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

    analysis = NormalIndPower()
    n_per_group = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided"
    )
    n_per_group = int(np.ceil(n_per_group))

    return {
        "baseline_cvr":     baseline_cvr,
        "mde_absolute":     mde,
        "mde_pp":           round(mde * 100, 2),
        "treatment_cvr":    round(p2, 4),
        "alpha":            alpha,
        "power":            power,
        "effect_size_h":    round(abs(effect_size), 4),
        "n_per_group":      n_per_group,
        "n_total":          n_per_group * 2,
    }


def achieved_power(
    n_control: int,
    n_treatment: int,
    baseline_cvr: float,
    observed_lift: float,
    alpha: float = 0.05,
) -> dict:
    """
    Calculate the actual statistical power our experiment achieved,
    given the observed effect size and sample sizes.

    Power = P(reject H0 | H1 is true)
    A well-powered experiment has power >= 0.80.
    """
    p1 = baseline_cvr
    p2 = baseline_cvr + observed_lift

    effect_size = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    n_per_group = min(n_control, n_treatment)  # use conservative (smaller) group

    analysis = NormalIndPower()
    power = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        nobs1=n_per_group,
        ratio=1.0,
        alternative="two-sided"
    )

    return {
        "n_control":       n_control,
        "n_treatment":     n_treatment,
        "baseline_cvr":    baseline_cvr,
        "observed_lift":   round(observed_lift, 4),
        "effect_size_h":   round(abs(effect_size), 4),
        "achieved_power":  round(power, 4),
        "adequate_power":  power >= 0.80,
        "verdict": (
            f"✅  Well-powered experiment (power = {power:.1%})"
            if power >= 0.80 else
            f"⚠️  Underpowered experiment (power = {power:.1%} < 80%). "
            f"Results may be unreliable."
        ),
    }


def minimum_detectable_effect(
    n_per_group: int,
    baseline_cvr: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Given a fixed sample size, what is the smallest effect we can reliably detect?

    This answers: "Was our experiment sensitive enough?"
    If MDE > observed lift, the experiment was under-powered for that effect size.
    """
    analysis = NormalIndPower()

    # Solve for effect size
    effect_size_h = analysis.solve_power(
        alpha=alpha,
        power=power,
        nobs1=n_per_group,
        ratio=1.0,
        alternative="two-sided"
    )

    # Convert Cohen's h back to absolute difference in proportions
    # h = 2*arcsin(sqrt(p2)) - 2*arcsin(sqrt(p1))
    phi1 = 2 * np.arcsin(np.sqrt(baseline_cvr))
    phi2 = phi1 + effect_size_h
    p2   = np.sin(phi2 / 2) ** 2
    mde  = p2 - baseline_cvr

    return {
        "n_per_group":      n_per_group,
        "baseline_cvr":     baseline_cvr,
        "mde_absolute":     round(mde, 4),
        "mde_pp":           round(mde * 100, 2),
        "mde_relative_pct": round(mde / baseline_cvr * 100, 1),
        "effect_size_h":    round(effect_size_h, 4),
    }


def power_curve(
    baseline_cvr: float,
    mde_range: list[float] | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> list[dict]:
    """
    Generate a power curve: required sample size vs. MDE.
    Useful for visualizing sample size trade-offs.
    """
    if mde_range is None:
        mde_range = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]

    rows = []
    for mde in mde_range:
        result = required_sample_size(baseline_cvr, mde, alpha, power)
        rows.append({
            "mde_pp":       round(mde * 100, 2),
            "n_per_group":  result["n_per_group"],
            "n_total":      result["n_total"],
        })
    return rows


def run_power_analysis(
    n_control: int,
    n_treatment: int,
    baseline_cvr: float,
    observed_lift: float,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Master runner for all power analysis components."""

    sep = "=" * 55
    if verbose:
        print(sep)
        print("  PHASE 6 — POWER ANALYSIS")
        print(sep)

    # Required sample size for this observed effect
    req = required_sample_size(baseline_cvr, observed_lift, alpha, power=0.80)
    if verbose:
        print(f"\n[A] Required Sample Size")
        print(f"    Baseline CVR : {baseline_cvr:.1%}")
        print(f"    MDE (lift)   : {observed_lift:.2%} ({observed_lift*100:.1f}pp)")
        print(f"    Required n   : {req['n_per_group']:,} per group ({req['n_total']:,} total)")
        print(f"    We had       : {n_control:,} + {n_treatment:,} = {n_control+n_treatment:,} total")
        adequately_sized = (n_control >= req["n_per_group"] and
                            n_treatment >= req["n_per_group"])
        print(f"    Adequate?    : {'✅  Yes' if adequately_sized else '⚠️  No — underpowered'}")

    # Achieved power
    ach = achieved_power(n_control, n_treatment, baseline_cvr, observed_lift, alpha)
    if verbose:
        print(f"\n[B] Achieved Power")
        print(f"    {ach['verdict']}")

    # MDE for our actual sample
    n_min = min(n_control, n_treatment)
    mde_r = minimum_detectable_effect(n_min, baseline_cvr, alpha, power=0.80)
    if verbose:
        print(f"\n[C] Minimum Detectable Effect (for our n={n_min:,} per group)")
        print(f"    MDE = {mde_r['mde_pp']:.2f}pp ({mde_r['mde_relative_pct']:.1f}% relative)")
        detectable = observed_lift >= mde_r["mde_absolute"]
        print(f"    Observed lift ({observed_lift*100:.2f}pp) {'≥' if detectable else '<'} MDE "
              f"({mde_r['mde_pp']:.2f}pp) → "
              f"{'✅  Detectable' if detectable else '⚠️  Below MDE'}")

    # Power curve
    curve = power_curve(baseline_cvr)

    if verbose:
        print(f"\n[D] Power Curve (n per group needed at 80% power)")
        print(f"    {'MDE':>8}  {'n/group':>10}  {'n total':>10}")
        print(f"    {'-'*32}")
        for row in curve:
            print(f"    {row['mde_pp']:>6.2f}pp  {row['n_per_group']:>10,}  {row['n_total']:>10,}")

        print(f"\n{sep}")
        print("  Power analysis complete")
        print(sep)

    return {
        "required_sample": req,
        "achieved_power":  ach,
        "mde":             mde_r,
        "power_curve":     curve,
    }
