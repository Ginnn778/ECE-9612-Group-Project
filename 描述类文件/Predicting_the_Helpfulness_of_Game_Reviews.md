# Predicting the Helpfulness of Game Reviews
## 项目总文档

> 本文档整合了项目所有规划与分析文档，包含：问题诊断、建模方案、模型策略推荐、Notebook 实现指南。

---

## 目录

1. [问题诊断](#一问题诊断)
2. [模型选型：系统性对比](#二模型选型系统性对比)
3. [核心策略：两阶段 Pipeline](#三核心策略两阶段-pipeline)
4. [特征工程](#四特征工程)
5. [评估指标](#五评估指标)
6. [Notebook 实现指南（逐 Part 说明）](#六notebook-实现指南逐-part-说明)
7. [实施优先级与路线图](#七实施优先级与路线图)

---

## 一、问题诊断

`votes_up` 存在极度右偏分布，**87.3% 的评论 votes_up = 0**。无论模型复杂度如何，所有单模型方案均在 MAE ≈ 0.52 处停滞不前。根本原因是**零值膨胀**——单一模型无法同时学习「这条评论会不会得到票」和「会得到多少票」这两个本质不同的问题。

| 观察 | 证据 |
|---|---|
| 极端右偏 | `votes_up` 偏度 = **36.60** |
| 零值主导 | **87.3%** 的评论 `votes_up = 0` |
| 仅 12.7% 非零 | "有帮助"的少数派驱动了所有信号 |
| 百分位数跳跃 | p99 = 5，但 max = 12371 — 极端长尾 |
| 特征相关性弱 | 最佳 Spearman r ≈ 0.21（`votes_funny`），多数特征 < 0.09 |
| 所有模型停滞 | Ridge MAE ≈ 0.52，LightGBM ≈ 0.52，LightGBM+TF-IDF ≈ 0.52 |

即使在进行 `log1p` 变换后（对数空间 MAE 从 0.54 降至 0.37），原始尺度 MAE 几乎不变，因为 87% 的预测只是在预测"零"。

---

## 二、模型选型：系统性对比

在确定最终方案前，对 **7 个候选模型**在统一条件下进行横向评估，以实验数据为依据选定最终模型。

### 2.1 预实验快速对比

| 方案 | 模型 | 特征数 | MAE（原始尺度） | 训练时间 |
|---|---|---|---|---|
| 基线回归（原始 target） | Ridge | 9 | 0.5428 | <0.01s |
| log1p 回归 | Ridge | 9 | ~0.52 | 0.01s |
| log1p 回归 | LightGBM | 9 | 0.5244 | 0.36s |
| log1p 回归 | LightGBM + TF-IDF | 3009 | **0.5191** | 3.94s |

### 2.2 全量模型对比实验（Part 7.2.1）

**7 个候选模型，统一条件下网格/随机搜索：**

| # | 模型 | 搜索方式 | 关键超参数范围 |
|---|---|---|---|
| 1 | Linear Regression | — | 无超参数 |
| 2 | Ridge Regression | GridSearchCV (5-fold) | `alpha ∈ [0.01, 0.1, 1, 10, 100]` |
| 3 | Random Forest | RandomizedSearchCV (3-fold) | `n_estimators ∈ [100,300]`，`max_depth ∈ [10,20,None]` |
| 4 | Gradient Boosting (sklearn) | 手动对比 | `lr ∈ [0.05,0.1]`，`n_estimators ∈ [100,300]`，`max_depth ∈ [4,6]` |
| 5 | XGBoost | 手动对比 | `n_estimators=200`，`max_depth ∈ [4,6]`，`lr=0.05` |
| 6 | LightGBM | 手动网格 | `num_leaves ∈ [31,63,127]`，`n_estimators ∈ [200,500]`，`lr ∈ [0.05,0.1]` |
| 7 | LightGBM + TF-IDF | 继承最优 | 同 LightGBM 最优参数 + TF-IDF `max_features=3000` |

**统一实验条件：**
- 训练/测试划分：80/20，`random_state=42`
- 目标变量：`log1p(votes_up)`，评估时 `expm1` 反变换回原始尺度
- 评估指标：MAE（原始尺度）、RMSE、训练时间

**对比结果表（运行后填入实际数值）：**

| 模型 | 最优超参数 | MAE | RMSE | 训练时间 | 备注 |
|---|---|---|---|---|---|
| Linear Regression | — | — | — | — | 基线 |
| Ridge | alpha=? | — | — | — | |
| Random Forest | depth=?, n=? | — | — | — | |
| Gradient Boosting | lr=?, n=? | — | — | — | |
| XGBoost | depth=?, n=? | — | — | — | |
| LightGBM | leaves=?, n=? | — | — | — | |
| LightGBM + TF-IDF | 同上 | — | — | — | |

> 💡 实验完成后，**加粗标注最终选定的模型**，并在下方写一段选型说明。

### 2.3 各模型排除原因

| 模型 | 排除原因 |
|---|---|
| 线性回归 / Ridge | 假设线性相关；votes_up 极度非线性，无法捕获特征交互 |
| 随机森林 | 训练慢、内存高；不原生支持稀疏矩阵，TF-IDF 效率差 |
| Gradient Boosting (sklearn) | 效率低于 LightGBM/XGBoost，不支持稀疏矩阵 |
| XGBoost | 性能与 LightGBM 相近，但训练速度慢 3–5 倍 |
| 深度学习 / BERT | 训练成本高；结构特征主导，NLP 驱动不适用 |

---

## 三、核心策略：两阶段 Pipeline

将问题拆解为两个独立的子问题：

```
输入评论
    │
    ▼
【第一阶段】最优分类器
    问题：这条评论会不会获得赞同票？（votes_up > 0）
    │
    ├── P < 0.5  →  预测 votes_up = 0          （占 ~87% 的情况）
    │
    └── P ≥ 0.5  →  【第二阶段】最优回归器
                        问题：具体会获得多少票？
                        训练目标：log1p(votes_up)
                        输出：expm1(pred) → 最终 votes_up 预测值
```

### 三路径并行策略

| 路径 | 方法 | 目的 |
|---|---|---|
| 路径 1（基线） | 最优单模型直接回归 + log1p | 来自 Part 7.2.1，MAE ≈ 0.52 |
| 路径 2（主方案） | 两阶段 Pipeline（最优分类器 + 最优回归器） | 主要学术贡献 |
| 路径 3（消融） | 路径 2 + 不同阈值 / 截断目标 | 验证设计决策有效性 |

### Stage 1：二分类器

```python
# 标签：该评论是否获得了赞同票？
y_cls = (df["votes_up"] > 0).astype(int)   # 正样本率 ~12.7%

# 用 balanced 权重处理类别不平衡
clf = BestClassifier(class_weight="balanced", random_state=42)
clf.fit(X_train, y_cls_train)
```

评估指标：**加权 F1、AUC-ROC、Precision/Recall**

### Stage 2：回归器（仅用 votes_up > 0 子集）

```python
mask = df["votes_up"] > 0
reg = BestRegressor(random_state=42)
reg.fit(X_train[mask], np.log1p(y_train_raw[mask]))
```

评估指标：**原始尺度 MAE**（`expm1(pred)` 后）

### 完整 Pipeline 推理

```python
def two_stage_predict(X, clf, reg, threshold=0.5):
    proba = clf.predict_proba(X)[:, 1]
    result = np.zeros(len(X))
    pos_mask = proba >= threshold
    if pos_mask.sum() > 0:
        result[pos_mask] = np.expm1(reg.predict(X[pos_mask]))
    return result
```

### 消融实验设计

| 消融编号 | 变量 | 对比条件 | 目的 |
|---|---|---|---|
| Ablation 1 | Stage 1 阈值 | `votes_up > 0` vs `> 3` vs `> 10` | 最优分类边界 |
| Ablation 2 | `weighted_vote_score` | 含 vs 不含 | 验证数据泄漏 |
| Ablation 3 | TF-IDF 在 Stage 1 | 含 vs 不含 | 文本对分类贡献 |
| Ablation 4 | 两阶段 vs 单模型 | Pipeline vs 直接回归 | 两阶段方案有效性 |

---

## 四、特征工程

### 4.1 各类特征

| 特征 | 预处理 | 用途 | 备注 |
|---|---|---|---|
| `playtime_forever`、`playtime_at_review` | `log1p` | ✅ 两阶段均使用 | 偏度 110–115，log1p 降至 ~0.3 |
| `playtime_last_two_weeks` | `log1p` | ✅ 两阶段均使用 | |
| `num_games_owned`、`num_reviews` | `log1p` | ✅ 两阶段均使用 | Spearman r≈0.08–0.09 |
| `voted_up` | 布尔 → int | ✅ 两阶段均使用 | 重要性=165 |
| `review_length`、`word_count` | 原始值 | ✅ 两阶段均使用 | EDA 5.2 中已计算 |
| `language` | One-hot（top 语言） | ✅ 两阶段均使用 | EDA 5.5 中已分析 |
| `price` | `log1p` | ✅ 游戏级特征 | 跨 80 款游戏有效变化 |
| `days_since_release` | `log1p` | ✅ 时间特征 | EDA 5.5.3 中已计算 |
| `positive`、`negative`、`pct_pos_total` | 原始 / `log1p` | ✅ 游戏人气特征 | 跨游戏有效变化 |
| TF-IDF on `review` | max_features=3000，sublinear_tf | ✅ Stage 1 必选；Stage 2 待消融 | 单模型回归增益仅 −0.2% |
| `weighted_vote_score` | 原始值 | ⚠️ 仅消融实验 | Spearman r=0.89，高泄漏风险 |
| `votes_funny` | — | ❌ 排除 | 与 votes_up 共现，数据泄漏 |
| `comment_count` | — | ❌ 排除 | 98.7% 为零，近零信息量 |

> ⚠️ **`weighted_vote_score` 泄漏警告**：由 Steam 基于 `votes_up`/`votes_down` 计算，Spearman r=0.89。必须排除在正式特征外，仅在消融实验中测试含/不含的效果差异。

### 4.2 特征矩阵汇总

```
结构特征（数值型，log1p 变换）：
    playtime_forever, playtime_at_review, playtime_last_two_weeks,
    num_games_owned, num_reviews, voted_up, review_length, word_count

游戏元数据特征：
    language（One-hot）, price（log1p）, days_since_release（log1p）,
    positive, negative, pct_pos_total, recommendations

文本特征（3000 维，稀疏矩阵）：
    TF-IDF(review, max_features=3000, sublinear_tf=True)

潜在泄漏特征（仅消融实验）：
    weighted_vote_score
```

---

## 五、评估指标

### 三层评估体系

| 层次 | 指标 | 评估对象 |
|---|---|---|
| Stage 1 单独评估 | 加权 F1、AUC-ROC、Precision/Recall | 分类器质量 |
| Stage 2 单独评估 | MAE（原始尺度）、RMSE | 回归器质量（votes_up > 0 子集） |
| **Pipeline 端到端** | **MAE（原始尺度）** | **最终核心指标** |

**成功标准：** 两阶段 Pipeline 端到端 MAE < 单模型基线 MAE（≈ 0.52）

### 最终对比表（实验后填入）

| 方案 | 端到端 MAE | 相对基线提升 | 备注 |
|---|---|---|---|
| 单模型最优（基线，来自 7.2.1） | — | — | 待填入模型名称及参数 |
| 两阶段 Pipeline（主方案） | — | — | 分类器 + 回归器模型待定 |
| + 移除 weighted_vote_score | — | — | 泄漏验证 |
| + threshold=3 | — | — | 阈值消融 |
| + threshold=10 | — | — | 阈值消融 |

---

## 六、Notebook 实现指南（逐 Part 说明）

### Part 1：Introduction

**当前状态：** 文字介绍已存在，无需修改。  
**可选补充：** 末尾简要说明最终采用两阶段 Pipeline 的背景动机。

---

### Part 2：Dataset Description

| 字段 | 说明 |
|---|---|
| 数据来源 | Steam Reviews（80 款游戏，多语言） |
| 总行数 | 4,374,931 条 |
| 游戏数量 | 80 款（top 游戏包括 Rocket League、Dead by Daylight、ETS2 等） |
| 目标变量 | `votes_up`：偏度=36.60，87.3% 为零，p99=5，max=12371 |
| 游戏元数据 | `games_march2025_cleaned.csv`（89,618 行，47 列） |

**合并方式：** reviews 与 games 按 `appid` 左连接，结果 71 列。

---

### Part 3：Problem Formulation

| 字段 | 内容 |
|---|---|
| 任务类型 | **两阶段混合任务**：Stage 1 = 二分类；Stage 2 = 回归 |
| 目标变量 | `votes_up`（评估用原始值），训练时用 `log1p(votes_up)` |
| 核心挑战 | ① 零值膨胀（87.3%）；② 极端长尾；③ 特征与 target 相关性弱；④ 正样本类别不平衡（12.7%） |
| 解决思路 | 拆分为「是否有帮助」（分类）+「帮助程度」（回归）两个子问题 |

---

### Part 4：Data Preprocessing

**4.1 库导入**  
确认导入所有候选模型库：`lightgbm`、`xgboost`、`sklearn.ensemble`、`scipy.sparse`。

**4.2 数据加载**  
已实现 Colab/本地双路径兼容（`RUNNING_IN_COLAB` flag + `os.getcwd()` 本地路径）。

**4.3 数据清洗**

新增两步：
```python
# 生成两阶段标签
df["is_helpful"] = (df["votes_up"] > 0).astype(int)   # Stage 1 标签
df["target_log"] = np.log1p(df["votes_up"])             # Stage 2 目标值

# 排除潜在泄漏特征（建模时不加入特征矩阵）
EXCLUDE_COLS = ["votes_funny", "weighted_vote_score",
                "recommendationid", "steamid"]
```

**4.3 数据变换**

数值特征 log1p 变换：
```python
LOG_COLS = ["playtime_forever", "playtime_at_review",
            "playtime_last_two_weeks", "num_games_owned", "num_reviews"]
for col in LOG_COLS:
    if col in df.columns:
        df[col] = np.log1p(df[col].clip(lower=0).fillna(0))
```

**4.4 数据集成**  
已按 `appid` 完成合并，无需修改。

---

### Part 5：EDA（探索性数据分析）

各节内容总结（详见 notebook 代码）：

| 节 | 内容 | 关键结论 |
|---|---|---|
| 5.1 Target Distribution | votes_up 分布、log 变换对比 | 偏度=36.60，零值膨胀严重，log1p 改善分布 |
| 5.2 Text Analysis | 评论长度/词数分布、与 votes_up 相关性 | 长度特征与 votes_up 弱相关（r≈0.05–0.15），仍作为辅助特征 |
| 5.3 Reviewer Behavior | num_reviews、playtime 分布、活跃度 vs 帮助度 | 结构特征是主要信号，reviewer 经验有弱正相关 |
| 5.4 Engagement Analysis | votes_funny、comment_count、weighted_vote_score | votes_funny/comment_count 可用；weighted_vote_score 高泄漏风险（r=0.89），排除 |
| 5.5 Game-level Analysis | 语言分布、价格、发布时间、游戏人气 | 80 款游戏，game-level 特征（price、pct_pos_total 等）跨行有效变化，可用作特征 |

---

### Part 6：Feature Engineering

**6.1 文本特征**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000, sublinear_tf=True, min_df=5)
X_text = tfidf.fit_transform(df["review"])   # 稀疏矩阵 (N, 3000)
df["review_length"] = df["review"].str.len()
df["word_count"] = df["review"].str.split().str.len()
```

**6.2 时间特征**
```python
# days_since_release 已在 EDA 5.5.3 中计算并写入 df
df["days_since_release_log"] = np.log1p(df["days_since_release"].clip(lower=0).fillna(0))
```

**6.3 Reviewer 特征**  
已在 Part 4.3 中完成 log1p 变换：`playtime_forever`、`playtime_at_review`、`playtime_last_two_weeks`、`num_games_owned`、`num_reviews`

**6.4 Engagement 特征**  
`weighted_vote_score` 单独维护，不默认加入特征矩阵，仅在消融实验中使用。

**6.5 游戏元数据**
```python
# 语言 → One-hot
lang_dummies = pd.get_dummies(df["language"], prefix="lang")

# price → log1p
df["price_log"] = np.log1p(df["price"].fillna(0))

# 游戏人气指标（跨 80 款游戏有效变化）
# positive, negative, pct_pos_total, recommendations 直接使用
```

---

### Part 7：Modeling Approach

**7.1 Baseline Model**

保持现有线性回归内容，补充引用：
```python
# 来自 Modeling_Decision_Analysis.ipynb 的基线结果：
# Ridge (log1p target): MAE ≈ 0.52（原始尺度）→ 两阶段方案对比基准
```

**7.2.1 系统性模型对比（选型依据）**

运行 7 个模型对比实验（见第二节），结果填入对比表后：
- 选取 MAE 最低的模型作为两阶段核心模型
- 若两模型 MAE 差距 < 0.005，优先选训练速度更快的

**7.2.2 两阶段 Pipeline 实现**

Stage 1（`BestClassifier` 占位，以实验结果为准）：
```python
clf = BestClassifier(class_weight="balanced", random_state=42)
clf.fit(X_train, y_cls_train)
```

Stage 2（`BestRegressor` 占位，以实验结果为准）：
```python
mask_train = y_train_raw > 0
reg = BestRegressor(random_state=42)
reg.fit(X_train[mask_train], np.log1p(y_train_raw[mask_train]))
```

完整推理：
```python
def two_stage_predict(X, clf, reg, threshold=0.5):
    proba = clf.predict_proba(X)[:, 1]
    result = np.zeros(len(X))
    pos_mask = proba >= threshold
    if pos_mask.sum() > 0:
        result[pos_mask] = np.expm1(reg.predict(X[pos_mask]))
    return result
```

**7.2.3 消融实验**（见第三节消融表）

---

### Part 8：Evaluation Strategy

**8.1 三层评估指标**（见第五节）

**8.2 最终对比表**（实验完成后填入，见第五节对比表）

---

### Part 9：Analysis and Discussion

**讨论要点：**
1. 两阶段 Pipeline 是否有效降低了 MAE？降低幅度是否显著？
2. `weighted_vote_score` 去除后效果如何？是否确认存在数据泄漏？
3. Stage 1 阈值（`> 0` vs `> 3` vs `> 10`）对最终结果的影响？
4. 文本特征（TF-IDF）在分类 vs 回归任务中贡献有何不同？
5. 项目局限性：数据涵盖 80 款游戏，但仍以热门游戏为主，跨类型泛化能力待验证。

---

## 七、实施优先级与路线图

### 优先级表

| 优先级 | Part | 工作量 | 说明 |
|---|---|---|---|
| 🔴 高 | Part 7.2 | 大 | 两阶段模型核心代码，全新实现 |
| 🔴 高 | Part 3 | 小 | 填充 Problem Formulation 内容 |
| 🟡 中 | Part 4.3 | 中 | 补充泄漏特征排除 + 两阶段标签生成 |
| 🟡 中 | Part 8 | 小 | 扩充为三层评估指标说明 |
| 🟢 低 | Part 5 | 小 | EDA 已完成（5.1–5.5） |
| 🟢 低 | Part 9 | 中 | 实验完成后填入对比表与讨论 |

### 实施路线图

```
Week 1:
  ✅ EDA 完成（Parts 5.1–5.5）
  ✅ 基线实验完成（MAE ≈ 0.52）
  → 运行 Part 7.2.1 的 7 模型对比实验，确定最优模型
  → 实现 Stage 1 分类器，在验证集上评估 F1 / AUC-ROC

Week 2:
  → 过滤 votes_up > 0 子集，训练 Stage 2 回归器
  → 构建完整推理 Pipeline（two_stage_predict）
  → 评估端到端 MAE vs 单模型基线

Week 3:
  → 消融 Ablation 1：阈值 > 0 vs > 3 vs > 10
  → 消融 Ablation 2：含 vs 不含 weighted_vote_score
  → 消融 Ablation 3：Stage 1 含 vs 不含 TF-IDF
  → 汇总路径 1 / 路径 2 / 路径 3 对比表
  → 撰写 Part 9 Analysis and Discussion
```

---

*文档最后更新：2026-03-22*
