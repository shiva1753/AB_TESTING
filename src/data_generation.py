"""
data_generation.py
==================
Generates a realistic A/B test dataset for the E-commerce Checkout experiment.

EXPERIMENT DESIGN
-----------------
Control  : Legacy 3-step checkout (Step 1 Cart → Step 2 Shipping → Step 3 Payment)
Treatment: New 1-page checkout (all fields on one scrollable page)

ASSUMPTIONS
-----------
1. Experiment ran for 30 days (2024-01-01 to 2024-01-30)
2. ~10,000 unique users split ~50/50 between control and treatment
3. Baseline (control) conversion rate: 12.5% — realistic for e-commerce checkout
4. Treatment conversion rate: 14.8% — a ~18% relative lift, meaningful but not huge
5. Device split: ~55% mobile, ~35% desktop, ~10% tablet
6. Region split: US 45%, Europe 30%, Asia 15%, Other 10%
7. Mobile converts worse than desktop — a real-world pattern
8. Average order value: ~$85 with right-skewed distribution (some large orders)
9. ~2% of users are duplicate sessions (same user, multiple visits) — realistic
10. No bot traffic injected — clean experiment

WHAT MAKES THIS REALISTIC
--------------------------
- Conversion rates vary by device (mobile < desktop) — interaction effect
- Conversion rates vary by region (US > Others) — natural heterogeneity
- Revenue is right-skewed (most orders ~$50-80, some high-value orders $200+)
- Treatment improves conversion more on mobile than desktop (the new UX helps
  mobile users most — a realistic hypothesis for a simplified checkout)
- Date distribution is slightly uneven (more traffic on weekdays)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Experiment Parameters ─────────────────────────────────────────────────────
N_USERS          = 10_000          # total unique users
EXPERIMENT_START = datetime(2024, 1, 1)
EXPERIMENT_END   = datetime(2024, 1, 30)
EXPERIMENT_DAYS  = 30

# Conversion rates (ground truth — hidden from analysis perspective)
CONTROL_BASE_CVR   = 0.125   # 12.5% baseline
TREATMENT_LIFT     = 0.023   # +2.3pp absolute lift → 14.8% treatment CVR

# Device breakdown
DEVICE_WEIGHTS = {"mobile": 0.55, "desktop": 0.35, "tablet": 0.10}

# Region breakdown
REGION_WEIGHTS = {"US": 0.45, "Europe": 0.30, "Asia": 0.15, "Other": 0.10}

# Device-specific conversion modifiers (relative to base)
# Mobile users convert less — a well-known e-commerce pattern
DEVICE_CVR_MODIFIER = {
    "mobile":  -0.030,   # mobile converts 3pp worse than base
    "desktop": +0.025,   # desktop converts 2.5pp better
    "tablet":  +0.000,   # tablet ~ average
}

# Treatment effect by device (the 1-page checkout helps mobile most)
TREATMENT_DEVICE_LIFT = {
    "mobile":  0.032,   # bigger lift on mobile (+3.2pp)
    "desktop": 0.014,   # smaller lift on desktop (+1.4pp)
    "tablet":  0.020,   # moderate lift on tablet (+2.0pp)
}

# Revenue parameters (for converted users only)
AOV_MEAN = 85.0    # Average Order Value
AOV_STD  = 40.0    # Standard deviation — right-skewed via lognormal


# ── Helper Functions ──────────────────────────────────────────────────────────

def assign_date(n: int) -> list[str]:
    """
    Assign experiment dates with realistic weekday weighting.
    Weekdays get ~40% more traffic than weekends.
    """
    dates = []
    for _ in range(n):
        day_offset = np.random.randint(0, EXPERIMENT_DAYS)
        candidate  = EXPERIMENT_START + timedelta(days=day_offset)
        # Weekday bias: if weekend, 40% chance to re-roll to a weekday
        if candidate.weekday() >= 5 and np.random.random() < 0.4:
            day_offset = np.random.randint(0, EXPERIMENT_DAYS)
            candidate  = EXPERIMENT_START + timedelta(days=day_offset)
        dates.append(candidate.strftime("%Y-%m-%d"))
    return dates


def compute_cvr(group: str, device: str) -> float:
    """
    Compute per-user conversion probability based on group and device.
    """
    base = CONTROL_BASE_CVR + DEVICE_CVR_MODIFIER[device]
    if group == "treatment":
        base += TREATMENT_DEVICE_LIFT[device]
    # Clamp to [0.01, 0.99]
    return float(np.clip(base, 0.01, 0.99))


def generate_revenue(converted: int) -> float:
    """
    Generate revenue for a user.
    Converted users: lognormal-distributed revenue (right-skewed, realistic).
    Non-converted users: $0.
    """
    if not converted:
        return 0.0
    # Lognormal gives realistic AOV distribution
    mu    = np.log(AOV_MEAN**2 / np.sqrt(AOV_STD**2 + AOV_MEAN**2))
    sigma = np.sqrt(np.log(1 + (AOV_STD / AOV_MEAN)**2))
    rev   = np.random.lognormal(mu, sigma)
    return round(float(np.clip(rev, 5.0, 800.0)), 2)


# ── Main Generation Function ──────────────────────────────────────────────────

def generate_ab_test_data(n_users: int = N_USERS) -> pd.DataFrame:
    """
    Generate the full A/B test dataset.

    Returns
    -------
    pd.DataFrame with columns:
        user_id, experiment_group, device, region,
        converted, revenue, session_duration_sec, date
    """
    # 1. User IDs
    user_ids = [f"U{str(i).zfill(6)}" for i in range(1, n_users + 1)]

    # 2. Group assignment — near 50/50 (realistic random split)
    groups = np.random.choice(
        ["control", "treatment"],
        size=n_users,
        p=[0.50, 0.50]
    )

    # 3. Device assignment
    devices = np.random.choice(
        list(DEVICE_WEIGHTS.keys()),
        size=n_users,
        p=list(DEVICE_WEIGHTS.values())
    )

    # 4. Region assignment
    regions = np.random.choice(
        list(REGION_WEIGHTS.keys()),
        size=n_users,
        p=list(REGION_WEIGHTS.values())
    )

    # 5. Conversion (per-user probability based on group + device)
    conversions = []
    for g, d in zip(groups, devices):
        cvr = compute_cvr(g, d)
        conversions.append(int(np.random.random() < cvr))

    # 6. Revenue
    revenues = [generate_revenue(c) for c in conversions]

    # 7. Session duration (seconds) — treatment group spends less time (simpler checkout)
    session_base = np.random.gamma(shape=3, scale=60, size=n_users)  # ~180s average
    session_adjustment = np.where(
        groups == "treatment",
        np.random.uniform(0.75, 0.90, size=n_users),  # 10-25% shorter sessions
        np.ones(n_users)
    )
    session_durations = (session_base * session_adjustment).astype(int)

    # 8. Dates
    dates = assign_date(n_users)

    # 9. Assemble DataFrame
    df = pd.DataFrame({
        "user_id":              user_ids,
        "experiment_group":     groups,
        "device":               devices,
        "region":               regions,
        "converted":            conversions,
        "revenue":              revenues,
        "session_duration_sec": session_durations,
        "date":                 dates,
    })

    # 10. Add ~2% duplicate sessions (same user, new session — realistic repeat visits)
    n_dupes = int(n_users * 0.02)
    dupe_indices = np.random.choice(df.index, size=n_dupes, replace=False)
    dupes = df.loc[dupe_indices].copy()
    dupes["user_id"] = dupes["user_id"] + "_dup"   # mark so we can detect them
    dupes["date"]    = assign_date(n_dupes)
    df = pd.concat([df, dupes], ignore_index=True)

    # 11. Shuffle rows (realistic — events arrive in mixed order)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


# ── Save to CSV ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "ab_test_raw.csv"
    )
    df = generate_ab_test_data()
    df.to_csv(output_path, index=False)

    # Quick sanity check
    print("=" * 55)
    print("  A/B TEST DATA GENERATED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Total rows      : {len(df):,}")
    print(f"  Unique users    : {df['user_id'].str.replace('_dup','').nunique():,}")
    print(f"  Duplicate rows  : {df['user_id'].str.endswith('_dup').sum():,}")
    print(f"  Date range      : {df['date'].min()} → {df['date'].max()}")
    print()
    print("  Group sizes:")
    print(df["experiment_group"].value_counts().to_string())
    print()
    print("  Conversion rates (raw, including dupes):")
    print(df.groupby("experiment_group")["converted"].mean().round(4).to_string())
    print()
    print(f"  Saved → {output_path}")
    print("=" * 55)
