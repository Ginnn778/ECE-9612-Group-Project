# Modeling Strategy Recommendation
## Steam Review Helpfulness Prediction (CS:GO Dataset)

> **Date:** 2026-03-21  
> **Dataset:** `big_reviews.csv` — 4,374,931 rows total; 50,000-row sample used for EDA & experiments  
> **Target:** `votes_up` (number of upvotes a review receives)

---

## 1. Problem Diagnosis — Why Raw Regression Is Hard

directly regressing on raw `votes_up` is genuinely difficult. Here is why, based on our actual data:

| Observation | Evidence |
|---|---|
| Extreme right skew | `votes_up` skewness = **36.60** on raw values |
| Zero-dominated | **87.3%** of reviews have `votes_up = 0` |
| Only 12.7% non-zero | The "interesting" minority drives all the signal |
| Percentile jump | p99 = 5, but p100 = 297 — a massive outlier tail |
| Weak feature correlations | Best Spearman r ≈ 0.21 (`votes_funny`), most features < 0.09 vs `votes_up` |
| All models plateau | Ridge MAE ≈ 0.52, LightGBM ≈ 0.52, LightGBM+TF-IDF ≈ 0.52 (Section 9 results) |

Even after `log1p` transform (MAE drops from 0.54 → 0.37 in log space), the real-scale MAE barely budges because 87% of predictions are just predicting "zero."

---

## 2. Evaluating the Two Proposals

### 2.1 Proposal A — Capped Target (Truncated Regression)

**Idea:** Cap `votes_up` at some ceiling (e.g., 99th percentile = 5) and regress on the capped value.

**Assessment: ⚠️ Partially useful, but limited**

- **Pro:** Reduces the influence of extreme outliers; slightly simplifies the regression task.
- **Con:** The cap is arbitrary. Capping at 5 loses all information about reviews with 10, 50, 297 upvotes — precisely the "viral" reviews that matter most for understanding helpfulness.
- **Con:** Does not solve the zero-inflation problem. 87.3% of data still has target = 0.
- **Verdict:** Acceptable as a preprocessing step (clip outliers for training stability) but **not recommended as a standalone strategy**.

---

### 2.2 Proposal B — Two-Stage Model (Classifier + Regressor)

**Idea:**
1. **Stage 1 (Binary Classifier):** Predict whether `votes_up > 10` ("high-quality review").
2. **Stage 2 (Regressor):** On the subset where Stage 1 predicts "yes," predict the exact `votes_up` count.

**Assessment: ✅ Sound direction, needs refinement**

**Strengths:**
- Directly addresses the zero-inflation problem by separating the "will this review get any traction?" question from "how much traction?".
- Stage 2 trains on a cleaner, less skewed subpopulation.
- Architecturally interpretable and modular — each stage can be tuned independently.

**Concerns and refinements:**
- **Threshold choice:** `votes_up > 10` is a reasonable start, but at p99 = 5 in our 50k sample, `votes_up > 10` may be very rare (~1–2% of data). Consider `votes_up > 0` (12.7% positive rate, already explored) or `votes_up > 3` as alternatives. **We recommend trying `votes_up > 0` first** since it aligns with existing EDA.
- **Class imbalance in Stage 1:** At 12.7% positive rate (`votes_up > 0`), use `class_weight='balanced'` — already validated to improve weighted F1 from 0.9694 → 0.9704.
- **Error propagation:** Stage 1 false positives feed noisy inputs to Stage 2. Final pipeline evaluation should use end-to-end MAE, not per-stage metrics alone.
- **Stage 2 data size:** If threshold is `votes_up > 10`, training data for Stage 2 shrinks drastically. Ensure sufficient samples for stable training.

---

## 3. Our Recommendation — Hybrid Three-Path Strategy

Based on EDA findings and experiment results, we recommend pursuing **all three paths in parallel** and reporting comparison results. This makes the project academically richer and more defensible.

### Path 1 — Direct Regression with log1p (Baseline)
> Already implemented in Sections 9–10.

- **Target:** `log1p(votes_up)`, evaluate with `expm1(pred)` → MAE in original space
- **Model:** LightGBM (`n_estimators=500`, `num_leaves=63`, `learning_rate=0.05`)
- **Features:** Structural features + `voted_up` boolean
- **Result so far:** MAE ≈ 0.52 (original scale) — use as baseline floor

### Path 2 — Two-Stage Model (Recommended Main Contribution)
> Implement as the primary novel contribution of the project.

**Stage 1 — Binary Classifier: "Will this review get upvotes?"**
- **Target:** `(votes_up > 0)` → binary label (positive rate 12.7%)
- **Model:** `LightGBM Classifier` with `class_weight='balanced'`
- **Features:** All structural features + TF-IDF (max 3000 features, sublinear_tf)
- **Key features (from importance analysis):** `weighted_vote_score`, `num_games_owned`, `num_reviews`, `playtime_forever`, `playtime_at_review`

**Stage 2 — Regressor: "How many upvotes will it get?"**
- **Subset:** Only rows where `votes_up > 0` (≈6,250 rows in 50k sample; ~560k in full dataset)
- **Target:** `log1p(votes_up)` on this subset — skew is much lower in this subpopulation
- **Model:** `LightGBM Regressor`
- **Features:** Same structural + TF-IDF features

**Inference pipeline:**
```
Input review
    → Stage 1 classifier → P(votes_up > 0)
        if P < 0.5:  predict votes_up = 0
        if P ≥ 0.5:  → Stage 2 regressor → expm1(pred) → votes_up estimate
```

**Evaluation:** Compute MAE and RMSE on the full test set using the combined pipeline output.

### Path 3 — Capped / Quantile Target (Ablation)

- Cap `votes_up` at p99 = 5 for training; evaluate on uncapped test set
- Or reframe as **ordinal regression**: predict bucket {0, 1–2, 3–5, 6–10, >10}
- Use as an ablation to show why uncapped two-stage beats truncation

---

## 4. Feature Engineering Recommendations

Based on Sections 2–10 analysis:

| Feature | Recommendation | Reason |
|---|---|---|
| `playtime_forever`, `playtime_at_review` | ✅ Keep, apply `log1p` | Skew=110–115, log1p reduces to ~0.3 |
| `num_games_owned`, `num_reviews` | ✅ Keep, apply `log1p` | Spearman r=0.08–0.09 with target; skew ~15–25 |
| `weighted_vote_score` | ✅ Keep | Top feature importance (2068); Spearman r=0.89 with `votes_up` — **but check for leakage** |
| `voted_up` | ✅ Keep as boolean | Useful signal; importance=165 |
| `votes_funny` | ⛔ Exclude | Spearman r=0.21 but co-occurs with `votes_up` → **data leakage** |
| `comment_count` | ⛔ Exclude | 98.7% zeros, near-zero information |
| `appid` | ⛔ Exclude | Constant (CS:GO only) |
| `last_played` | ⚠️ Use with caution | Unix timestamp, not playtime duration |
| TF-IDF on `review` | ⚠️ Conditional | Max_features=3000, sublinear_tf. Brings only -0.2% gain in single-model regression, but may help Stage 1 classifier more meaningfully |
| Review length (`review_len`) | ✅ Engineer | Spearman r=0.09 with target; cheap to compute |
| Language (`language`) | ✅ One-hot top-10 | English dominates (48%), but lang shows some median variance |

> ⚠️ **Critical note on `weighted_vote_score`:** This feature has Spearman r=0.89 with `votes_up`. It is computed by Steam using vote counts, so it is almost certainly a **leaky feature** (partially derived from `votes_up` itself). Always include a no-leakage ablation that excludes it.

---

## 5. Model Selection Summary

| Stage | Task | Recommended Model | Why |
|---|---|---|---|
| Stage 1 | Binary classification | **LightGBM Classifier** | Handles sparse TF-IDF, class imbalance, fast |
| Stage 1 (baseline) | Binary classification | Logistic Regression | Interpretable, quick sanity check |
| Stage 2 | Regression on votes_up > 0 | **LightGBM Regressor** | Best performer in Section 9; handles skewed log target |
| Stage 2 (baseline) | Regression | Ridge Regression | Fast, already validated (MAE=0.37 in log space) |
| Optional | Full regression | LightGBM + TF-IDF | Already implemented; use as Path 1 baseline |

---

## 6. Evaluation Metrics

| Metric | Use case |
|---|---|
| **MAE (original scale)** | Primary regression metric — penalizes absolute errors |
| **RMSE (original scale)** | Secondary — more sensitive to large outliers |
| **Weighted F1** | Stage 1 classifier — accounts for class imbalance |
| **Precision/Recall @ threshold** | Stage 1 — tune threshold for precision vs. recall tradeoff |
| **End-to-end MAE** | Final pipeline evaluation — most important single number |

---

## 7. Recommended Implementation Roadmap

```
Week 1: 
  ✅ EDA complete (Sections 1–8 done)
  ✅ Baseline experiments complete (Sections 9–10 done)
  → Implement Stage 1 LightGBM classifier (votes_up > 0)
  → Evaluate Stage 1 in isolation (F1, AUC-ROC)

Week 2:
  → Filter training data to votes_up > 0 subset
  → Train Stage 2 LightGBM regressor on subset
  → Build inference pipeline (Stage 1 → Stage 2)
  → Evaluate end-to-end MAE vs. baseline

Week 3:
  → Ablation: votes_up > 3 vs votes_up > 10 as Stage 1 threshold
  → Ablation: with vs. without weighted_vote_score (leakage check)
  → Ablation: with vs. without TF-IDF in Stage 1
  → Report Path 1 vs. Path 2 vs. Path 3 comparison table
  → Write final report
```

---

## 8. Summary

| Proposal | Verdict | Action |
|---|---|---|
| Raw regression (current) | ⚠️ Hard but valid baseline | Keep as Path 1 |
| Capped target | ⚠️ Limited; loses tail info | Use as ablation only |
| Two-stage model | ✅ **Recommended main approach** | Implement as Path 2 |
| Two-stage threshold | Start with `votes_up > 0` | Ablate with `> 3`, `> 10` |
| Model choice | LightGBM for both stages | Already validated in experiments |
| TF-IDF text features | Marginal gain in regression; test in classifier | Include in Stage 1 experiment |
| `weighted_vote_score` | High importance but **potential leakage** | Run ablation with and without |
