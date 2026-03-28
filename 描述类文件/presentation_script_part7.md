# Presentation: Part 7 — Classification Models & Two-Stage Pipeline
**Speaker scope:** Classification models (7.1–7.2) + Two-Stage Pipeline design & implementation (7.4)  
**Time budget:** ~1 min 40 sec (medium pace, ~250 words)

---

## PPT Slides

---

### Slide 1 — Classification Models: Feature Engineering Drives Performance

**Title:** Classification Models: What Features Actually Matter?

**Layout:** Left column = feature diagram | Right column = result highlight box

**Left — Feature Groups (visual icon list):**

```
📝 Text Features
   ├── TF-IDF  (review vocabulary)
   └── BERT    (sentence semantics)

🎮 Behavioral Features
   ├── playtime_forever / playtime_at_review
   ├── review_length
   └── num_games_owned / num_reviews

📦 Game Metadata
   ├── price (log-scaled)
   ├── days_since_release (log-scaled)
   └── genre dummies (one-hot)
```

**Right — Key Result Box (large font):**

```
┌──────────────────────────────────────┐
│  LR + TF-IDF only                    │
│  ROC-AUC = 0.812                     │
│                                      │
│  ↓  add behavioral + metadata        │
│                                      │
│  LR + Full Features         ⭐        │
│  ROC-AUC = 0.845  (+0.033)           │
│  Accuracy = 0.770   F1 = 0.768       │
└──────────────────────────────────────┘
```

**Bottom caption:**
> Feature engineering > model complexity for this task.  
> Linear models (LR / Ridge / SVC) all plateau at ~Acc 0.763 with full features.

---

### Slide 2 — Two-Stage Pipeline: Design & Implementation

**Title:** Two-Stage Pipeline: Combining Classification + Regression

**Layout:** Top = pipeline flow diagram | Bottom = implementation detail table

**Top — Pipeline Flow (horizontal diagram):**

```
  All Reviews
      │
      ▼
┌─────────────┐        Stage 1 (7.1.4 LR)
│ Classifier  │ ──────► P(helpful) ∈ [0, 1]
│ Full Feature│                │
└─────────────┘                │
                               ▼
┌─────────────────────────────────────────┐
│  Helpful Subset Only (is_helpful = 1)   │
│  Stage 2 Regression  ──► ŷ_votes_up     │  ← trained on helpful reviews only
│  · Ridge (log target)                   │
│  · TweedieRegressor  (power=1.5)        │
│  · LightGBM Tweedie                     │
└─────────────────────────────────────────┘
                               │
                               ▼
         Final Prediction = P(helpful) × ŷ_votes_up
```

**Bottom — Key Design Choices Table:**

| Decision | Choice | Reason |
|---|---|---|
| Stage-2 training data | Helpful reviews only | Remove noise from zero-vote reviews |
| Regression target | `log1p(votes_up)` | Stabilise heavy right-skewed distribution |
| TweedieRegressor power | 1.5 (Poisson–Gamma) | Native support for non-negative count with heavy tail |
| Combination formula | `P(helpful) × ŷ` | Suppress over-prediction on non-helpful reviews |

---

## 演讲稿（中英文对照，约 250 词 / 1分40秒）

---

> "My section covers two parts: the classification models we built in 7.1 and 7.2, and the design of the Two-Stage Pipeline in 7.4.
>
> For classification, we treated the problem as predicting whether a review would be considered helpful — a binary label.
>
> We started with simple Logistic Regression on TF-IDF text features, getting a ROC-AUC of 0.812. Adding BERT embeddings gave us similar results. The real improvement came when we introduced behavioral and game metadata features — things like playtime at review time, review length, game price, days since release, and genre.
>
> That single addition pushed our best model — Logistic Regression with full features — to **ROC-AUC 0.845 and accuracy 0.770**. That's a gain of 0.033 AUC, larger than any model switch we tried.
>
> We also tested RidgeClassifier, LinearSVC, and an MLP. The two linear models matched LR almost exactly, telling us linear classifiers have reached a ceiling on this feature set. The MLP, trained on BERT features only, actually scored the worst at 0.674 accuracy — confirming that behavioral signals are essential, not just text.
>
> Now for the Two-Stage Pipeline. The core idea is: not every review deserves a vote count prediction. So Stage 1 uses the best classifier to output a probability — P(helpful) — for each review. Stage 2 then trains a regression model *exclusively on helpful reviews*, predicting `log1p(votes_up)`. We tried three Stage-2 regressors: Ridge, TweedieRegressor with power 1.5 — which natively handles non-negative skewed count data — and LightGBM with a Tweedie objective.
>
> The final prediction combines both stages: **P(helpful) multiplied by the regression output**. This multiplication acts as a soft gate — it suppresses over-prediction on reviews the classifier considers non-helpful.
>
> The comparison results will be covered in the next section."

---

## 时间节奏参考

| 段落 | 内容 | 建议时长 |
|---|---|---|
| 开场 | 说明自己讲的范围 | 10 秒 |
| 分类模型特征工程 | TF-IDF → Full Features，ROC-AUC 提升 | 30 秒 |
| 高级分类模型对比 | Ridge / SVC / MLP 结论 | 15 秒 |
| Two-Stage 设计思路 | 为什么要两阶段 | 15 秒 |
| Stage 1 实现 | 分类器输出 P(helpful) | 10 秒 |
| Stage 2 实现 | 三种回归器选择 + Tweedie 理由 | 20 秒 |
| 组合公式 | P × ŷ 的直觉解释 | 10 秒 |
| 过渡 | 把结论图交给下一位 | 5 秒 |
| **合计** | | **~1 分 35 秒** |
