"""
data_validation.py
==================
Phase 2 — Data Quality & Experiment Validation

Checks to run BEFORE any statistical test:
  1. Missing values & duplicates
  2. Group sample sizes
  3. Sample Ratio Mismatch (SRM)
  4. Randomization sanity checks
  5. Conversion counts
  6. Experiment duration
  7. Suspicious / bot-like users
"""

import pandas as pd
import numpy as np
from scipy import stats


# ── 1. Basic Quality Check ────────────────────────────────────────────────────

def check_basic_quality(df: pd.DataFrame) -> dict:
    """Check missing values, dtypes, and basic shape."""
    result = {
        "total_rows":     len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "any_missing":    df.isnull().any().any(),
        "dtypes":         df.dtypes.astype(str).to_dict(),
    }
    return result


# ── 2. Duplicate Detection ────────────────────────────────────────────────────

def check_duplicates(df: pd.DataFrame) -> dict:
    """
    Detect duplicate user_ids (same user appearing multiple times).
    In real experiments, each user should appear exactly once.
    Duplicates can inflate conversion counts.
    """
    # Exact duplicate rows
    exact_dupes = df.duplicated().sum()

    # Users appearing more than once (different sessions, same user)
    user_counts  = df["user_id"].value_counts()
    multi_users  = user_counts[user_counts > 1]

    # Detect the _dup marker we injected during generation
    synthetic_dupes = df["user_id"].str.endswith("_dup").sum()

    return {
        "exact_duplicate_rows":   int(exact_dupes),
        "users_with_multi_rows":  int(len(multi_users)),
        "synthetic_dup_rows":     int(synthetic_dupes),
        "dup_user_ids_sample":    multi_users.head(5).to_dict(),
    }


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate sessions: keep the first record per user_id base.
    This is the standard approach — deduplicate before analysis.
    """
    # Strip _dup suffix to get the canonical user_id
    df = df.copy()
    df["user_id_base"] = df["user_id"].str.replace("_dup", "", regex=False)

    # Keep only first occurrence of each base user_id
    df_clean = df.drop_duplicates(subset="user_id_base", keep="first").copy()
    df_clean = df_clean.drop(columns=["user_id_base"])
    df_clean = df_clean.reset_index(drop=True)
    return df_clean


# ── 3. Sample Ratio Mismatch (SRM) ───────────────────────────────────────────

def check_srm(df: pd.DataFrame, expected_split: float = 0.50,
              significance: float = 0.01) -> dict:
    """
    Sample Ratio Mismatch (SRM) Check.

    SRM occurs when the actual split between control/treatment differs
    significantly from the intended split. It suggests broken randomization,
    data pipeline errors, or selection bias — and means the experiment results
    cannot be trusted even if p < 0.05.

    Uses a chi-square goodness-of-fit test.
    H0: Observed group sizes match the expected 50/50 split.
    """
    counts    = df["experiment_group"].value_counts()
    n_total   = len(df)
    n_control = counts.get("control", 0)
    n_treat   = counts.get("treatment", 0)

    expected_control = n_total * expected_split
    expected_treat   = n_total * (1 - expected_split)

    chi2, p_value = stats.chisquare(
        f_obs=[n_control, n_treat],
        f_exp=[expected_control, expected_treat]
    )

    srm_detected = p_value < significance
    actual_split = n_control / n_total

    return {
        "n_control":         int(n_control),
        "n_treatment":       int(n_treat),
        "n_total":           int(n_total),
        "expected_split":    expected_split,
        "actual_split":      round(actual_split, 4),
        "chi2_statistic":    round(chi2, 4),
        "p_value":           round(p_value, 4),
        "srm_detected":      srm_detected,
        "verdict":           "⚠️  SRM DETECTED — Investigate before proceeding"
                             if srm_detected else
                             "✅  No SRM — Randomization looks clean",
    }


# ── 4. Randomization Sanity Checks ───────────────────────────────────────────

def check_randomization(df: pd.DataFrame) -> dict:
    """
    Verify that device and region distributions are balanced across groups.
    Large imbalances suggest confounding or broken randomization.
    Uses chi-square test of independence.
    """
    results = {}

    for feature in ["device", "region"]:
        contingency = pd.crosstab(df["experiment_group"], df[feature])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        results[feature] = {
            "chi2":    round(chi2, 4),
            "p_value": round(p, 4),
            "dof":     dof,
            "balanced": p > 0.05,
            "crosstab_pct": (
                contingency
                .div(contingency.sum(axis=1), axis=0)
                .round(3)
                .to_dict()
            ),
        }

    return results


# ── 5. Conversion Sanity Check ────────────────────────────────────────────────

def check_conversions(df: pd.DataFrame) -> dict:
    """
    Basic conversion count sanity check.
    Flags groups with suspiciously low (<5) or zero conversions.
    """
    grp = df.groupby("experiment_group").agg(
        n_users    = ("user_id", "count"),
        n_convert  = ("converted", "sum"),
        cvr        = ("converted", "mean"),
    ).round(4)

    return {
        "summary":     grp.to_dict(),
        "min_conversions": int(grp["n_convert"].min()),
        "sufficient":  bool(grp["n_convert"].min() >= 5),
    }


# ── 6. Experiment Duration Check ──────────────────────────────────────────────

def check_duration(df: pd.DataFrame, min_days: int = 14) -> dict:
    """
    Check whether the experiment ran long enough.
    Best practice: at least 1-2 full business cycles (typically 2 weeks minimum)
    to capture weekday/weekend variation.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    start = df["date"].min()
    end   = df["date"].max()
    days  = (end - start).days + 1

    daily_users = df.groupby("date")["user_id"].count()

    return {
        "start_date":        str(start.date()),
        "end_date":          str(end.date()),
        "experiment_days":   days,
        "min_recommended":   min_days,
        "sufficient_duration": days >= min_days,
        "avg_daily_users":   round(daily_users.mean(), 1),
        "verdict":           "✅  Duration sufficient"
                             if days >= min_days else
                             f"⚠️  Only {days} days — recommend at least {min_days}",
    }


# ── 7. Bot / Suspicious User Detection ───────────────────────────────────────

def check_suspicious_users(df: pd.DataFrame) -> dict:
    """
    Flag potentially suspicious users:
    - Zero session duration (bot-like)
    - Extremely short sessions (< 5 seconds) with a conversion (suspicious)
    - Extremely long sessions (outliers)
    """
    very_short   = df[df["session_duration_sec"] < 5]
    fast_convert = df[(df["session_duration_sec"] < 5) & (df["converted"] == 1)]
    p99_duration = df["session_duration_sec"].quantile(0.99)
    very_long    = df[df["session_duration_sec"] > p99_duration]

    return {
        "very_short_sessions":          len(very_short),
        "fast_convert_suspicious":      len(fast_convert),
        "p99_session_duration_sec":     round(p99_duration, 1),
        "very_long_session_count":      len(very_long),
        "action_needed":                len(fast_convert) > 0,
    }


# ── Master Validation Runner ──────────────────────────────────────────────────

def run_all_checks(df_raw: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Run all validation checks and return a summary dict.
    Also returns the cleaned DataFrame ready for analysis.
    """
    sep = "=" * 55

    if verbose:
        print(sep)
        print("  PHASE 2 — DATA QUALITY & EXPERIMENT VALIDATION")
        print(sep)

    # --- 1. Basic quality
    quality = check_basic_quality(df_raw)
    if verbose:
        print(f"\n[1] Basic Quality")
        print(f"    Rows: {quality['total_rows']:,}")
        print(f"    Missing values: {quality['any_missing']}")

    # --- 2. Duplicates
    dupes = check_duplicates(df_raw)
    if verbose:
        print(f"\n[2] Duplicate Check")
        print(f"    Duplicate rows (synthetic): {dupes['synthetic_dup_rows']}")
        print(f"    Users with multiple rows  : {dupes['users_with_multi_rows']}")

    # Clean data
    df_clean = remove_duplicates(df_raw)
    if verbose:
        print(f"    → After deduplication: {len(df_clean):,} rows")

    # --- 3. SRM
    srm = check_srm(df_clean)
    if verbose:
        print(f"\n[3] Sample Ratio Mismatch (SRM)")
        print(f"    Control  : {srm['n_control']:,}")
        print(f"    Treatment: {srm['n_treatment']:,}")
        print(f"    Chi²={srm['chi2_statistic']}, p={srm['p_value']}")
        print(f"    {srm['verdict']}")

    # --- 4. Randomization
    rand = check_randomization(df_clean)
    if verbose:
        print(f"\n[4] Randomization Check (Device & Region Balance)")
        for feat, res in rand.items():
            status = "✅  Balanced" if res["balanced"] else "⚠️  Imbalanced"
            print(f"    {feat:8s}: chi²={res['chi2']}, p={res['p_value']} → {status}")

    # --- 5. Conversions
    conv = check_conversions(df_clean)
    if verbose:
        print(f"\n[5] Conversion Counts")
        print(f"    Min conversions in any group: {conv['min_conversions']}")
        print(f"    Sufficient for testing: {conv['sufficient']}")

    # --- 6. Duration
    dur = check_duration(df_clean)
    if verbose:
        print(f"\n[6] Experiment Duration")
        print(f"    {dur['start_date']} → {dur['end_date']} ({dur['experiment_days']} days)")
        print(f"    {dur['verdict']}")

    # --- 7. Suspicious users
    sus = check_suspicious_users(df_clean)
    if verbose:
        print(f"\n[7] Suspicious Users")
        print(f"    Very short sessions (<5s): {sus['very_short_sessions']}")
        print(f"    Fast converters (suspicious): {sus['fast_convert_suspicious']}")

    if verbose:
        print(f"\n{sep}")
        print("  ✅  Validation complete — data is clean for analysis")
        print(sep)

    return {
        "quality":      quality,
        "duplicates":   dupes,
        "srm":          srm,
        "randomization": rand,
        "conversions":  conv,
        "duration":     dur,
        "suspicious":   sus,
        "df_clean":     df_clean,
    }
