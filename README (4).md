# E-commerce Checkout A/B Test Analysis

**Portfolio Project · Data Analyst · Statistical Experimentation**

---

## Business Problem

An e-commerce company redesigned its checkout flow from a **multi-step (3-page) checkout** to a **simplified one-page checkout**. The business wants to know:

> *Did the new checkout genuinely improve conversion rate, or is the observed difference just random chance?*

**Final decision: SHIP / DON'T SHIP / RUN LONGER**

---

## Project Structure

```
ecommerce-ab-test/
├── data/
│   └── ab_test_raw.csv          ← Generated experiment dataset (10,200 rows)
├── notebooks/
│   └── ab_test_analysis.ipynb   ← Main analysis notebook (all 10 phases)
├── src/
│   ├── data_generation.py       ← Realistic A/B test data simulator
│   ├── data_validation.py       ← SRM, quality checks, duplicate detection
│   ├── statistical_tests.py     ← Z-test, CI, practical significance, segments
│   └── power_analysis.py        ← Sample size, achieved power, MDE
├── visualizations/
│   ├── ab_test_dashboard.png    ← 6-panel analysis dashboard
│   └── peeking_simulation.png  ← Peeking pitfall demonstration
├── requirements.txt
└── README.md
```

---

## Experiment Design

| Parameter | Value |
|---|---|
| Experiment | Multi-step (Control) vs One-page (Treatment) checkout |
| Period | 2024-01-01 → 2024-01-30 (30 days) |
| Users | 10,000 unique users (~5,000 per group) |
| Randomization | Server-side, 50/50 split |
| Primary Metric | Conversion Rate |
| Guardrail Metrics | Revenue per User, Average Order Value |
| Significance level | α = 0.05, two-tailed |
| Target Power | 80% |

---

## Analysis Phases

### Phase 1 — Data Generation
Simulated a realistic experiment dataset using NumPy with documented assumptions:
- Baseline conversion rate: 12.5% (realistic e-commerce benchmark)
- Device-specific conversion differences (mobile < desktop — real-world pattern)
- Right-skewed revenue distribution (lognormal, AOV ~$85)
- 2% duplicate sessions (realistic repeat visits)

### Phase 2 — Data Quality & Experiment Validation
Ran all critical pre-analysis checks **before** any statistical test:
- ✅ No missing values
- ✅ 200 duplicate sessions removed
- ✅ No Sample Ratio Mismatch (χ²=2.31, p=0.13)
- ✅ Device and region balanced across groups
- ✅ 30-day duration covers full weekday/weekend cycles
- ✅ Minimal suspicious user activity

### Phase 3 — Metric Definition
- **Primary:** Conversion Rate
- **Guardrails:** Revenue per User, Average Order Value

### Phase 4 — Hypothesis Testing

| Metric | Control | Treatment |
|---|---|---|
| Users | 5,076 | 4,924 |
| Conversions | 592 | 730 |
| Conversion Rate | 11.66% | 14.83% |
| Absolute Lift | — | **+3.17pp** |
| Relative Lift | — | **+27.2%** |
| P-value | — | **0.000003** |
| 95% CI | — | [+1.83pp, +4.49pp] |
| Z-statistic | — | 4.67 |

**Result: Statistically significant (p << 0.05). CI excludes zero.**

### Phase 5 — Practical Significance
The lift of **+3.17pp exceeds our 1pp business threshold** — practically meaningful.

Estimated business impact per 10,000-user cohort:
- ~156 additional conversions
- ~$13,100 additional revenue

### Phase 6 — Power Analysis
- Required sample: 1,800 per group for this effect size
- Actual sample: ~5,000 per group
- **Achieved power: 99.6%** — well-powered, results are reliable

### Phase 7 — Experiment Pitfalls
Analyzed and explained 6 critical A/B testing pitfalls:
1. **Sample Ratio Mismatch** — detected via chi-square; none found
2. **Peeking** — simulated daily p-value evolution showing early stopping risk
3. **Multiple Comparisons** — Bonferroni correction applied across 8 segment tests
4. **Simpson's Paradox** — checked subgroup direction consistency; no reversal found
5. **Selection Bias** — mitigated by server-side randomization
6. **Insufficient Sample Size** — power analysis confirms adequate n

### Phase 8 — Segment Analysis

**By Device** (Bonferroni α = 0.0167):
| Device | Control CVR | Treatment CVR | Lift | Significant? |
|---|---|---|---|---|
| Mobile | 8.29% | 13.33% | +5.05pp | ✅ Yes |
| Desktop | 16.33% | 16.98% | +0.65pp | ❌ No |
| Tablet | 13.68% | 15.34% | +1.66pp | ❌ No |

**Key insight:** The lift is driven by **mobile users** — the 1-page checkout eliminates scrolling friction on small screens.

**By Region** (Bonferroni α = 0.0125):
- US and Other: Significant after Bonferroni
- Europe and Asia: Positive but subgroups may be underpowered

### Phase 9 — Visualizations
- Conversion rate comparison (bar chart)
- **Effect size with 95% CI (most important chart)**
- Revenue per user comparison
- Treatment effect by device
- Treatment effect by region
- Power curve (sample size vs MDE)
- Peeking simulation (daily p-value)

### Phase 10 — Business Recommendation

## ✅ DECISION: SHIP

The one-page checkout is a clear improvement:
- Statistically significant (p = 3×10⁻⁶)
- Practically significant (3.17pp > 1pp threshold)
- Revenue per user **increased** — no quality trade-off
- Mobile uplift is especially strong
- Experiment was well-designed and well-powered

**Post-launch monitoring plan:**
- Daily conversion rate for 30 days
- Weekly revenue per user and AOV
- Alert if CVR drops below 13%

---

## Technical Stack

| Tool | Usage |
|---|---|
| Python 3.11 | Core language |
| pandas | Data manipulation |
| NumPy | Data simulation, numerical ops |
| SciPy | Statistical tests (z-test, chi-square, Mann-Whitney) |
| statsmodels | Proportions z-test, power analysis |
| Matplotlib / Seaborn | Visualizations |
| Jupyter Notebook | Analysis environment |

---

## Key Interview Talking Points

| Question | Answer |
|---|---|
| Why validate before testing? | SRM and data issues invalidate results even with p < 0.05 |
| Why z-test? | Large n, binary outcome, two independent groups — CLT applies |
| Why show CI, not just p-value? | p-value answers "is it real?"; CI answers "how big is it?" |
| Stat sig vs practical sig? | Large n makes tiny effects significant — need a business threshold |
| What is peeking? | Stopping early when p < 0.05 inflates false positive rate |
| What is SRM? | Split deviates from intended — suggests broken randomization |
| What is Simpson's Paradox? | Aggregate result reverses within subgroups — check consistency |
| How did you handle multiple comparisons? | Bonferroni correction across segment tests |
| Was the experiment adequately powered? | 99.6% power — far exceeded 80% target |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the notebook
jupyter notebook notebooks/ab_test_analysis.ipynb
```

Or run the modules directly:
```bash
python src/data_generation.py   # generate data
python src/data_validation.py   # run validation checks
```

---

*This project demonstrates: experiment design, statistical inference, effect size analysis, confidence intervals, power analysis, and A/B testing pitfall identification — core skills for a Data Analyst role.*
