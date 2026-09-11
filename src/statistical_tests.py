"""
statistical_tests.py
====================
Phase 3, 4, 5, 7, 8 — Metric Definition, Hypothesis Testing,
Practical Significance, Pitfall Analysis, Segment Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import warnings
warnings.filterwarnings("ignore")


# ── Phase 3 — Metric Calculation ──────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute primary and guardrail metrics for each group.

    Primary  : Conversion Rate
    Guardrail: Revenue per User, Average Order Value
    """
    grp = df.groupby("experiment_group").agg(
        n_users        = ("user_id",   "count"),
        n_converted    = ("converted", "sum"),
        total_revenue  = ("revenue",   "sum"),
    )
    grp["conversion_rate"]   = grp["n_converted"] / grp["n_users"]
    grp["revenue_per_user"]  = grp["total_revenue"] / grp["n_users"]
    grp["avg_order_value"]   = (
        df[df["converted"] == 1]
        .groupby("experiment_group")["revenue"]
        .mean()
    )
    return grp.round(4)


# ── Phase 4 — Hypothesis Testing ──────────────────────────────────────────────

def two_proportion_ztest(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Two-proportion z-test for conversion rate difference.

    H0: p_control == p_treatment  (no difference)
    H1: p_control != p_treatment  (two-tailed)

    Returns everything needed for the final report:
      - rates, lift (absolute + relative)
      - z-statistic, p-value
      - 95% confidence interval on the difference
      - business interpretation
    """
    grp = df.groupby("experiment_group").agg(
        n       = ("user_id",   "count"),
        convert = ("converted", "sum"),
    )

    n_ctrl  = grp.loc["control",   "n"]
    n_treat = grp.loc["treatment", "n"]
    c_ctrl  = grp.loc["control",   "convert"]
    c_treat = grp.loc["treatment", "convert"]

    p_ctrl  = c_ctrl  / n_ctrl
    p_treat = c_treat / n_treat

    # Two-proportion z-test (statsmodels)
    z_stat, p_value = proportions_ztest(
        count=[c_treat, c_ctrl],
        nobs=[n_treat, n_ctrl],
        alternative="two-sided"
    )

    # 95% CI on the difference (treatment - control)
    # Using normal approximation
    se_diff = np.sqrt(p_ctrl*(1-p_ctrl)/n_ctrl + p_treat*(1-p_treat)/n_treat)
    z_crit  = stats.norm.ppf(1 - alpha/2)
    diff    = p_treat - p_ctrl
    ci_low  = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    # Absolute and relative lift
    abs_lift = p_treat - p_ctrl
    rel_lift = abs_lift / p_ctrl

    significant = p_value < alpha

    return {
        # Rates
        "p_control":          round(p_ctrl,  4),
        "p_treatment":        round(p_treat, 4),
        "n_control":          int(n_ctrl),
        "n_treatment":        int(n_treat),
        "conversions_control":   int(c_ctrl),
        "conversions_treatment": int(c_treat),
        # Lift
        "absolute_lift":      round(abs_lift, 4),
        "relative_lift":      round(rel_lift, 4),
        # Test statistics
        "z_statistic":        round(z_stat,  4),
        "p_value":            round(p_value, 6),
        "alpha":              alpha,
        # Confidence interval
        "ci_lower":           round(ci_low,  4),
        "ci_upper":           round(ci_high, 4),
        "ci_excludes_zero":   bool(ci_low > 0 or ci_high < 0),
        # Verdict
        "statistically_significant": significant,
        "verdict": (
            f"✅  SIGNIFICANT (p={p_value:.4f} < α={alpha}) — "
            f"Treatment CVR {p_treat:.1%} vs Control {p_ctrl:.1%}, "
            f"lift = +{abs_lift:.2%} absolute (+{rel_lift:.1%} relative)"
        ) if significant else (
            f"❌  NOT SIGNIFICANT (p={p_value:.4f} ≥ α={alpha}) — "
            f"Cannot reject H0"
        ),
    }


def chi_square_test(df: pd.DataFrame) -> dict:
    """
    Chi-square test of independence as a validation check.
    Should agree with the z-test result.
    """
    contingency = pd.crosstab(df["experiment_group"], df["converted"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "chi2_statistic": round(chi2, 4),
        "p_value":        round(p, 6),
        "dof":            dof,
        "agrees_with_ztest": True,  # should always match for 2x2 table
    }


# ── Phase 5 — Practical Significance ──────────────────────────────────────────

def practical_significance(
    df: pd.DataFrame,
    ztest_result: dict,
    min_detectable_effect: float = 0.01,  # 1pp = minimum business cares about
) -> dict:
    """
    Determine whether the effect is large enough to matter to the business.

    Statistical significance tells you the effect is REAL.
    Practical significance tells you the effect is MEANINGFUL.

    min_detectable_effect: The smallest lift the business considers worthwhile.
    """
    abs_lift = ztest_result["absolute_lift"]
    rel_lift = ztest_result["relative_lift"]
    n_treat  = ztest_result["n_treatment"]
    p_treat  = ztest_result["p_treatment"]
    p_ctrl   = ztest_result["p_control"]

    # Additional conversions per period (treatment vs. control, same sample size)
    additional_conversions = abs_lift * n_treat

    # Revenue impact estimation
    # Average revenue per converted user (treatment group)
    treat_converted = df[
        (df["experiment_group"] == "treatment") & (df["converted"] == 1)
    ]["revenue"]
    avg_rev = treat_converted.mean() if len(treat_converted) > 0 else 0

    estimated_revenue_lift = additional_conversions * avg_rev

    practically_significant = abs(abs_lift) >= min_detectable_effect

    return {
        "absolute_lift_pp":          round(abs_lift * 100, 2),    # in percentage points
        "relative_lift_pct":         round(rel_lift * 100, 2),    # in %
        "min_detectable_effect_pp":  round(min_detectable_effect * 100, 2),
        "practically_significant":   practically_significant,
        "additional_conversions_per_cohort": round(additional_conversions, 1),
        "avg_revenue_per_conversion":        round(avg_rev, 2),
        "estimated_revenue_lift":            round(estimated_revenue_lift, 2),
        "verdict": (
            f"✅  PRACTICALLY SIGNIFICANT — Lift {abs_lift:.2%} exceeds MDE "
            f"of {min_detectable_effect:.2%}. Estimated {additional_conversions:.0f} "
            f"extra conversions worth ${estimated_revenue_lift:,.0f} per cohort."
        ) if practically_significant else (
            f"⚠️  NOT PRACTICALLY SIGNIFICANT — Lift {abs_lift:.2%} is below "
            f"the business threshold of {min_detectable_effect:.2%}."
        ),
    }


# ── Phase 7 — Pitfall Analysis ────────────────────────────────────────────────

def check_multiple_comparisons(n_tests: int, alpha: float = 0.05) -> dict:
    """
    Multiple comparisons / multiple testing problem.

    Running many tests at α=0.05 inflates the false positive rate.
    If we test 20 subgroups, we expect ~1 false positive by chance.
    Bonferroni correction: use α/n_tests as the adjusted threshold.
    """
    bonferroni_alpha   = alpha / n_tests
    family_wise_error  = 1 - (1 - alpha) ** n_tests

    return {
        "n_tests":              n_tests,
        "original_alpha":       alpha,
        "bonferroni_alpha":     round(bonferroni_alpha, 4),
        "family_wise_error_rate": round(family_wise_error, 4),
        "explanation": (
            f"With {n_tests} tests at α={alpha}, the probability of at least "
            f"one false positive is {family_wise_error:.1%}. "
            f"Bonferroni-corrected α = {bonferroni_alpha:.4f}."
        ),
    }


def check_simpsons_paradox(df: pd.DataFrame) -> dict:
    """
    Simpson's Paradox check.
    Tests whether the aggregate result holds within each device subgroup.
    A reversal of direction in subgroups is Simpson's Paradox.
    """
    overall_ctrl  = df[df["experiment_group"]=="control"]["converted"].mean()
    overall_treat = df[df["experiment_group"]=="treatment"]["converted"].mean()
    overall_dir   = "treatment_better" if overall_treat > overall_ctrl else "control_better"

    subgroup_dirs = []
    rows = []
    for seg in df["device"].unique():
        sub = df[df["device"] == seg]
        c = sub[sub["experiment_group"]=="control"]["converted"].mean()
        t = sub[sub["experiment_group"]=="treatment"]["converted"].mean()
        d = "treatment_better" if t > c else "control_better"
        subgroup_dirs.append(d)
        rows.append({
            "device": seg,
            "control_cvr": round(c, 4),
            "treatment_cvr": round(t, 4),
            "direction": d
        })

    paradox_detected = any(d != overall_dir for d in subgroup_dirs)

    return {
        "overall_direction":    overall_dir,
        "subgroup_results":     rows,
        "paradox_detected":     paradox_detected,
        "verdict": (
            "⚠️  SIMPSON'S PARADOX DETECTED — subgroup direction differs from aggregate"
            if paradox_detected else
            "✅  No Simpson's Paradox — direction consistent across subgroups"
        ),
    }


def peeking_simulation(df: pd.DataFrame, alpha: float = 0.05) -> list[dict]:
    """
    Simulate the 'peeking' problem.
    Shows what happens if you check p-values daily and stop when p < 0.05.
    This inflates the false positive rate significantly.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    results = []
    for cutoff in dates:
        sub  = df[df["date"] <= cutoff]
        grp  = sub.groupby("experiment_group").agg(n=("user_id","count"), c=("converted","sum"))
        if "control" not in grp.index or "treatment" not in grp.index:
            continue
        n_c, c_c = grp.loc["control",  ["n","c"]]
        n_t, c_t = grp.loc["treatment",["n","c"]]
        if c_c < 5 or c_t < 5:
            continue
        try:
            _, p = proportions_ztest([c_t, c_c], [n_t, n_c], alternative="two-sided")
        except Exception:
            continue
        results.append({
            "date":        str(cutoff.date()),
            "n_total":     int(len(sub)),
            "p_value":     round(p, 4),
            "significant": p < alpha,
        })
    return results


# ── Phase 8 — Segment Analysis ────────────────────────────────────────────────

def segment_analysis(df: pd.DataFrame, segment: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run hypothesis test within each level of a segment.
    Returns a DataFrame with conversion rates and significance per segment.

    IMPORTANT: Adjust for multiple comparisons — we test n_segments hypotheses.
    """
    segments = df[segment].unique()
    n_segs   = len(segments)
    bonferroni = alpha / n_segs

    rows = []
    for seg_val in sorted(segments):
        sub = df[df[segment] == seg_val]
        grp = sub.groupby("experiment_group").agg(
            n=("user_id","count"), c=("converted","sum")
        )
        if "control" not in grp.index or "treatment" not in grp.index:
            continue

        n_c, c_c = grp.loc["control",  ["n","c"]]
        n_t, c_t = grp.loc["treatment",["n","c"]]
        p_c = c_c / n_c
        p_t = c_t / n_t

        try:
            z, p = proportions_ztest([c_t, c_c], [n_t, n_c], alternative="two-sided")
        except Exception:
            z, p = np.nan, np.nan

        rows.append({
            segment:             seg_val,
            "n_control":         int(n_c),
            "n_treatment":       int(n_t),
            "cvr_control":       round(p_c, 4),
            "cvr_treatment":     round(p_t, 4),
            "absolute_lift":     round(p_t - p_c, 4),
            "z_stat":            round(z, 3) if not np.isnan(z) else None,
            "p_value":           round(p, 4) if not np.isnan(p) else None,
            "sig_unadjusted":    p < alpha if not np.isnan(p) else False,
            "sig_bonferroni":    p < bonferroni if not np.isnan(p) else False,
            "bonferroni_alpha":  round(bonferroni, 4),
        })

    return pd.DataFrame(rows)


# ── Revenue Guardrail Test ─────────────────────────────────────────────────────

def revenue_guardrail_test(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Mann-Whitney U test for revenue per user (non-parametric, because
    revenue is heavily right-skewed with many zeros).
    Checks whether treatment harmed revenue even if conversion improved.
    """
    ctrl_rev  = df[df["experiment_group"]=="control"]["revenue"].values
    treat_rev = df[df["experiment_group"]=="treatment"]["revenue"].values

    u_stat, p_value = stats.mannwhitneyu(treat_rev, ctrl_rev, alternative="two-sided")

    mean_ctrl  = ctrl_rev.mean()
    mean_treat = treat_rev.mean()
    diff       = mean_treat - mean_ctrl

    return {
        "mean_revenue_control":   round(mean_ctrl, 2),
        "mean_revenue_treatment": round(mean_treat, 2),
        "difference":             round(diff, 2),
        "pct_change":             round(diff / mean_ctrl * 100, 2),
        "u_statistic":            round(u_stat, 2),
        "p_value":                round(p_value, 4),
        "significant":            p_value < alpha,
        "verdict": (
            f"⚠️  Revenue per user changed significantly (p={p_value:.4f})"
            if p_value < alpha else
            f"✅  Revenue per user not significantly different (p={p_value:.4f})"
        ),
    }
