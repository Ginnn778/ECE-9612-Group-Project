# Notebook 实现指南
## `Predicting_the_Helpfulness_of_Game_Reviews.ipynb` 各节修改说明

> 本文档按 notebook 目录结构逐节说明 `modeling_plan.md` 的落地方式。

---

## Part 1：Introduction

**当前状态：** 文字介绍已存在，无需修改。  
**可选补充：** 在末尾简要说明最终采用两阶段 Pipeline 的背景动机（一两句话即可）。

---

## Part 2：Dataset Description

### 2.1 Dataset 1: Steam Reviews

**当前状态：** 已有描述框架。  
**补充内容：**

| 字段 | 说明 |
|---|---|
| 数据来源 | Steam Reviews 2020（CS:GO, appid=620） |
| 原始总行数 | 4,374,931 条 |
| 使用样本 | 50,000 条（约 1.14%） |
| 目标变量 | `votes_up`：偏度=36.60，87.3% 为零，p99=5，max=297 |

### 2.2 Dataset 2: Steam Games Metadata

**当前状态：** 已有描述框架。  
**补充内容：** 说明从该数据集中引入的特征：`genre`（类型）、`price`（价格）、`release_date`（用于计算 `days_since_release`）。

### 2.3 Data Integration

**当前状态：** 已有合并逻辑（按 `appid` 合并）。  
**确认无需修改。**

---

## Part 3：Problem Formulation

**当前状态：** 仅有空占位符，需要完整填写。  
**需要补充的内容：**

| 字段 | 内容 |
|---|---|
| 任务类型 | **两阶段混合任务**：Stage 1 = 二分类；Stage 2 = 回归 |
| 目标变量 | `votes_up`（评估用原始值），训练时用 `log1p(votes_up)` |
| 核心挑战 | ① 零值膨胀（87.3% 为零）；② 极端长尾（p99=5 但 max=297）；③ 特征与 target 相关性弱（最高 Spearman r=0.21）；④ 正样本类别不平衡（12.7%） |
| 解决思路 | 将问题拆分为「是否有帮助」（分类）+「帮助程度」（回归）两个子问题 |

---

## Part 4：Data Preprocessing

### 4.1 Technical Preliminaries

**当前状态：** 已有库导入。  
**补充：** 确认导入所有候选模型库：`lightgbm`、`xgboost`、`sklearn.ensemble`、`scipy.sparse`（视 7.2.1 对比结果按需保留）。

### 4.2 Data Loading

**当前状态：** 已使用 Google Drive 路径加载 `big_reviews.csv`。  
**无需修改。**

### 4.2 Dataset Overview and Sampling

**当前状态：** 已有 shape/dtypes 预览。  
**补充：** 打印 `votes_up` 分布统计（mean、std、skewness、% zeros）。

### 4.3 Data Cleaning

**当前状态：** 已有文本清洗与时间戳转换。  
**新增两步：**

1. **生成两阶段标签**
   ```python
   df["is_helpful"] = (df["votes_up"] > 0).astype(int)   # Stage 1 标签
   df["target_log"] = np.log1p(df["votes_up"])             # Stage 2 目标值
   ```

2. **排除潜在泄漏特征**
   ```python
   # votes_funny 与 votes_up 共现 → 排除
   # weighted_vote_score 由 Steam 基于投票数计算，Spearman r=0.89 → 单独消融
   EXCLUDE_COLS = ["votes_funny", "comment_count", "appid",
                   "recommendationid", "steamid"]
   ```

### 4.3 Data Transformation

**当前状态：** 已有数值/布尔类型转换。  
**新增：数值特征 log1p 变换**

```python
LOG_COLS = ["playtime_forever", "playtime_at_review",
            "playtime_last_two_weeks", "num_games_owned", "num_reviews"]
for col in LOG_COLS:
    if col in df.columns:
        df[col] = np.log1p(df[col].clip(lower=0).fillna(0))
```

### 4.4 Data Integration

**当前状态：** 已按 `appid` 完成合并。  
**确认无需修改。**

---

## Part 5：Exploratory Data Analysis

> 💡 本节结论已在 `Modeling_Decision_Analysis.ipynb` Sections 2–8 中完成，可直接引用图表，无需重复计算。

### 5.1 Target Distribution

- `votes_up` 偏度=36.60，87.3% 为零
- `log1p` 变换后偏度降至约 5.23，分布明显改善
- **结论**：零值膨胀问题严重，单一模型无法有效拟合

### 5.2 Text Analysis

- 评论中位长度约 29 字符，分布极度右偏
- TF-IDF 在单模型回归中增益仅 −0.2%
- **结论**：文本特征对回归贡献有限，但可能对 Stage 1 分类有帮助（待消融验证）

### 5.3 Reviewer Behavior

- `playtime_forever`、`num_games_owned` 是较重要的行为特征
- **结论**：结构特征是主要信号来源

### 5.4 Engagement Analysis

- `votes_funny` 与 `votes_up` 高度相关但存在共现泄漏
- `weighted_vote_score` Spearman r=0.89 → 标注为潜在泄漏特征，进入消融实验

### 5.5 Game-level Analysis

- 语言分布：英语 48%，俄语次之
- **结论**：语言字段可作为分类特征之一

---

## Part 6：Feature Engineering

### 6.1 Text Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000, sublinear_tf=True, min_df=5)
X_text = tfidf.fit_transform(df["review_cleaned"])
# 输出：稀疏矩阵 (N, 3000)

df["review_len"] = df["review_cleaned"].str.len()   # 文本长度特征
```

### 6.2 Temporal Features

```python
# 需要从 games 数据集的 release_date 计算
df["days_since_release"] = (df["timestamp_created"] - df["release_timestamp"]).dt.days
df["days_since_release"] = np.log1p(df["days_since_release"].clip(lower=0).fillna(0))
```

### 6.3 Reviewer Features

```python
# 已在 Part 4.3 中完成 log1p 变换：
# playtime_forever, playtime_at_review, playtime_last_two_weeks,
# num_games_owned, num_reviews
```

### 6.4 Engagement Features

```python
# weighted_vote_score：单独维护，不默认加入特征矩阵
# 仅在消融实验「含 vs 不含」场景中使用
```

### 6.5 Game Metadata

```python
# genre → One-hot 编码
genre_dummies = pd.get_dummies(df["genre"], prefix="genre")

# price → log1p 变换
df["price_log"] = np.log1p(df["price"].fillna(0))
```

**最终特征矩阵汇总：**

```
结构特征（7 维）：
    playtime_forever, playtime_at_review, playtime_last_two_weeks,
    num_games_owned, num_reviews, voted_up, review_len

游戏元数据特征：
    genre（One-hot）, price_log, days_since_release

文本特征（3000 维，稀疏）：
    TF-IDF(review_cleaned, max_features=3000)

潜在泄漏特征（仅消融使用）：
    weighted_vote_score
```

---

## Part 7：Modeling Approach

### 7.1 Baseline Model

**保持现有线性回归内容，补充实验结论引用：**

```python
# 来自 Modeling_Decision_Analysis.ipynb Section 9 的基线结果（无需重跑）：
# Ridge (log1p target): MAE ≈ 0.52（原始尺度）
# → 作为两阶段方案的对比基准（baseline）
```

---

### 7.2 Advanced Models

#### 7.2.1 系统性模型对比（选型依据）

在确定两阶段 Pipeline 的核心模型之前，对以下 **7 个候选模型**进行统一条件下的横向对比：

| # | 模型 | 搜索策略 | 关键超参数范围 |
|---|---|---|---|
| 1 | Linear Regression | — | 无 |
| 2 | Ridge Regression | GridSearchCV (5-fold) | `alpha ∈ [0.01, 0.1, 1, 10, 100]` |
| 3 | Random Forest | RandomizedSearchCV (3-fold, n_iter=10) | `n_estimators ∈ [100,300]`，`max_depth ∈ [10,20,None]`，`min_samples_leaf ∈ [1,5]` |
| 4 | Gradient Boosting (sklearn) | 手动对比 | `n_estimators ∈ [100,300]`，`lr ∈ [0.05,0.1]`，`max_depth ∈ [4,6]` |
| 5 | XGBoost | 手动对比 | `n_estimators=200`，`max_depth ∈ [4,6]`，`lr=0.05` |
| 6 | LightGBM | 手动网格 | `num_leaves ∈ [31,63,127]`，`n_estimators ∈ [200,500]`，`lr ∈ [0.05,0.1]` |
| 7 | LightGBM + TF-IDF | 继承最优参数 | 同 #6 最优参数 + TF-IDF `max_features=3000` |

**统一实验条件：**

```python
# 所有模型遵守相同条件：
# - 划分：80/20，random_state=42
# - 目标变量：log1p(votes_up)，评估时 expm1 反变换回原始尺度
# - 结构特征：7 维（均已 log1p 变换 + fillna(0)）
# - 评估指标：MAE（原始尺度）、RMSE、训练时间（秒）

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time

results = []

def evaluate(name, model, X_tr, y_tr, X_te, y_te_raw):
    t0 = time.time()
    model.fit(X_tr, np.log1p(y_tr))
    t1 = time.time()
    pred = np.expm1(model.predict(X_te))
    mae = mean_absolute_error(y_te_raw, pred)
    rmse = np.sqrt(mean_squared_error(y_te_raw, pred))
    results.append({"Model": name, "MAE": mae, "RMSE": rmse,
                    "Train_time": f"{t1-t0:.2f}s"})
```

**对比结果表（运行后填入实际数值）：**

| 模型 | 最优超参数 | MAE（原始尺度） | RMSE | 训练时间 |
|---|---|---|---|---|
| Linear Regression | — | — | — | — |
| Ridge | alpha=? | — | — | — |
| Random Forest | depth=?, n=? | — | — | — |
| Gradient Boosting | lr=?, n=? | — | — | — |
| XGBoost | depth=?, n=? | — | — | — |
| LightGBM | leaves=?, n=? | — | — | — |
| LightGBM + TF-IDF | 同上 | — | — | — |

**决策规则：**
- 选取 MAE 最低的模型作为两阶段 Pipeline 的核心模型
- 若两模型 MAE 差距 < 0.005，优先选训练速度更快的
- Random Forest / GBM 若内存超限，记录原因后排除

> 💡 7.2.1 实验完成后，在结果表中**加粗标注最终选定的模型**，并在下方写一段选型说明（为什么选这个，其余模型差在哪里）。

---

#### 7.2.2 两阶段 Pipeline 实现

> ⚠️ **模型待定**：Stage 1（分类器）和 Stage 2（回归器）的具体模型类型，**以 7.2.1 对比实验的结果为准**，选取各自任务中 MAE / F1 最优的模型填入。下方代码以伪代码形式表示 Pipeline 结构，`BestClassifier` 和 `BestRegressor` 为占位符。

**Stage 1：二分类器（votes_up > 0？）**

```python
# 候选：LGBMClassifier / XGBClassifier / RandomForestClassifier / LogisticRegression
# 选取在验证集上加权 F1 最高的模型
# 正样本率仅 12.7%，所有模型均需设置类别权重或 class_weight='balanced'

clf = BestClassifier(
    # 超参数来自 7.2.1 对比实验中 Stage 1 的最优配置
    class_weight="balanced",   # 或 scale_pos_weight（XGBoost）
    random_state=42,
)
clf.fit(X_train, y_cls_train)   # y_cls = (votes_up > 0).astype(int)

# Stage 1 评估：加权 F1、AUC-ROC、Precision/Recall
```

**Stage 2：回归器（仅在 votes_up > 0 子集上训练）**

```python
# 候选：LGBMRegressor / XGBRegressor / RandomForestRegressor / Ridge
# 选取在 votes_up > 0 子集验证集上 MAE（原始尺度）最低的模型

mask_train = y_train_raw > 0
reg = BestRegressor(
    # 超参数来自 7.2.1 对比实验中 Stage 2 的最优配置
    random_state=42,
)
reg.fit(X_train[mask_train], np.log1p(y_train_raw[mask_train]))

# Stage 2 评估：MAE（原始尺度，expm1 反变换）
```

**完整 Pipeline 推理函数（与具体模型类型无关）：**

```python
def two_stage_predict(X, clf, reg, threshold=0.5):
    proba = clf.predict_proba(X)[:, 1]
    result = np.zeros(len(X))
    pos_mask = proba >= threshold
    if pos_mask.sum() > 0:
        result[pos_mask] = np.expm1(reg.predict(X[pos_mask]))
    return result

y_pred = two_stage_predict(X_test, clf, reg)
print(f"Two-Stage Pipeline MAE: {mean_absolute_error(y_test_raw, y_pred):.4f}")
print(f"Baseline (single best model) MAE: ~0.52")
```

---

#### 7.2.3 消融实验

| 消融编号 | 变量 | 对比条件 | 目的 |
|---|---|---|---|
| Ablation 1 | Stage 1 阈值 | `votes_up > 0` vs `> 3` vs `> 10` | 验证最优分类边界 |
| Ablation 2 | `weighted_vote_score` | 含 vs 不含 | 验证是否存在数据泄漏 |
| Ablation 3 | TF-IDF 在 Stage 1 | 含 vs 不含 | 验证文本对分类的贡献 |
| Ablation 4 | 两阶段 vs 单模型 | Pipeline vs 直接回归 | 验证两阶段方案的有效性 |

```python
# Ablation 1 示例（以最优分类器为准）
for threshold in [0, 3, 10]:
    y_cls = (y_raw > threshold).astype(int)
    clf_tmp = BestClassifier(class_weight="balanced", ...)
    clf_tmp.fit(X_train, y_cls[train_idx])
    y_pred_tmp = two_stage_predict(X_test, clf_tmp, reg)
    print(f"Threshold={threshold}: MAE={mean_absolute_error(y_test_raw, y_pred_tmp):.4f}")
```

---

## Part 8：Evaluation Strategy

### 8.1 Metrics

**采用三层评估体系：**

| 层次 | 指标 | 评估对象 |
|---|---|---|
| Stage 1 单独评估 | 加权 F1、AUC-ROC、Precision / Recall | 分类器质量 |
| Stage 2 单独评估 | MAE（原始尺度）、RMSE | 回归器质量（votes_up > 0 子集） |
| **Pipeline 端到端** | **MAE（原始尺度）** | **最终核心指标** |

**成功标准：** 两阶段 Pipeline 端到端 MAE < 单模型基线 MAE（≈ 0.52）

### 8.2 Model Comparison

**对比维度：**

| 比较路径 | 方法 | 端到端 MAE |
|---|---|---|
| 路径 1（单模型基线） | 最优单模型直接回归 + log1p（来自 7.2.1） | ~0.52（已有） |
| 路径 2（主方案） | 两阶段 Pipeline（最优分类器 + 最优回归器） | 待实验 |
| 路径 3（消融对照） | 路径 2 + 移除 `weighted_vote_score` | 待实验 |
| 路径 4（阈值对照） | 路径 2 + threshold=3 | 待实验 |

---

## Part 9：Analysis and Discussion

**需要展示的最终对比结论，实验完成后填入：**

| 方案 | 端到端 MAE | 相对基线提升 | 备注 |
|---|---|---|---|
| 单模型最优（基线，来自 7.2.1） | — | — | 实验后填入模型名称及参数 |
| 两阶段 Pipeline（主方案） | — | — | 分类器 + 回归器模型名称待定 |
| + 移除 weighted_vote_score | — | — | 泄漏验证 |
| + threshold=3 | — | — | 阈值消融 |

**讨论要点：**
1. 两阶段 Pipeline 是否有效降低了 MAE？降低幅度是否显著？
2. `weighted_vote_score` 去除后效果如何？是否确认存在数据泄漏？
3. Stage 1 阈值（`> 0` vs `> 3` vs `> 10`）对最终结果的影响？
4. 文本特征（TF-IDF）在分类 vs 回归任务中贡献有何不同？
5. 项目局限性：样本仅覆盖 CS:GO 单款游戏，跨游戏泛化能力待验证。

---

## 实施优先级

| 优先级 | Part | 工作量 | 说明 |
|---|---|---|---|
| 🔴 高 | Part 7.2 | 大 | 两阶段模型核心代码，全新实现 |
| 🔴 高 | Part 3 | 小 | 填充 Problem Formulation 内容 |
| 🟡 中 | Part 4.3 | 中 | 补充泄漏特征排除 + 两阶段标签生成 |
| 🟡 中 | Part 8 | 小 | 扩充为三层评估指标说明 |
| 🟢 低 | Part 5 | 小 | 直接引用 Modeling_Decision_Analysis 结论图表 |
| 🟢 低 | Part 9 | 中 | 实验完成后填入对比表与讨论 |
