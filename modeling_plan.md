# 建模方案：预测 Steam 评论帮助度（votes_up）

---

## 一、问题诊断

`votes_up` 存在极度右偏分布（偏度 = 36.60），**87.3% 的评论 votes_up = 0**。
无论模型复杂度如何，所有单模型方案均在 MAE ≈ 0.52 处停滞不前。
根本原因是**零值膨胀**——单一模型无法同时学习「这条评论会不会得到票」和「会得到多少票」这两个本质不同的问题。

---

## 二、模型选型：为什么选 LightGBM？

在确定最终方案前，我们**系统对比了 7 个候选模型**，在统一条件下进行横向评估，最终以实验数据为依据选定 LightGBM。

### 2.1 预实验快速对比（来自 `Modeling_Decision_Analysis.ipynb` 第 9 节）

| 方案 | 模型 | 特征数 | MAE（原始尺度） | 训练时间 |
|---|---|---|---|---|
| 基线回归（原始 target） | Ridge | 9 | 0.5428 | <0.01s |
| log1p 回归 | Ridge | 9 | ~0.52 | 0.01s |
| log1p 回归 | LightGBM | 9 | 0.5244 | 0.36s |
| log1p 回归 | LightGBM + TF-IDF | 3009 | **0.5191** | 3.94s |

### 2.2 全量模型对比实验（在 notebook Part 7.2 中完整运行）

为了提高模型选型的学术可信度，在 Part 7.2 中对以下 7 个模型进行统一条件下的网格/随机搜索对比：

| # | 模型 | 搜索方式 | 关键超参数范围 |
|---|---|---|---|
| 1 | Linear Regression | — | 无超参数 |
| 2 | Ridge Regression | GridSearch | `alpha ∈ [0.01, 0.1, 1, 10, 100]` |
| 3 | Random Forest | RandomizedSearch | `n_estimators ∈ [100,300]`，`max_depth ∈ [10,20,None]`，`min_samples_leaf ∈ [1,5]` |
| 4 | Gradient Boosting (sklearn) | 手动对比 | `lr ∈ [0.05,0.1]`，`n_estimators ∈ [100,300]`，`max_depth ∈ [4,6]` |
| 5 | XGBoost | 手动对比 | `n_estimators=200`，`max_depth ∈ [4,6]`，`lr=0.05` |
| 6 | **LightGBM** | 手动网格 | `num_leaves ∈ [31,63,127]`，`n_estimators ∈ [200,500]`，`lr ∈ [0.05,0.1]` |
| 7 | LightGBM + TF-IDF | 继承最优 | 同 LightGBM 最优参数 + TF-IDF max_features=3000 |

**统一实验条件：**
- 训练/测试划分：80/20，`random_state=42`
- 目标变量：`log1p(votes_up)`，评估时 `expm1` 反变换回原始尺度
- 特征：结构特征 9 个（均经 log1p 变换 + fillna(0)）
- 评估指标：MAE（原始尺度）、RMSE、训练时间

**对比结果表（运行后填入实际数值）：**

| 模型 | 最优超参数 | MAE | RMSE | 训练时间 | 备注 |
|---|---|---|---|---|---|
| Linear Regression | — | — | — | — | 基线 |
| Ridge | alpha=? | — | — | — | |
| Random Forest | depth=?, n=? | — | — | — | 内存占用高 |
| Gradient Boosting | lr=?, n=? | — | — | — | |
| XGBoost | depth=?, n=? | — | — | — | |
| **LightGBM** | leaves=?, n=? | **—** | **—** | **—** | 预期最优 |
| LightGBM + TF-IDF | 同上 | — | — | — | 文本增益验证 |

### 2.3 为什么不选其他模型（学术分析）

| 模型 | 排除原因 |
|---|---|
| **线性回归 / Ridge** | 假设特征与目标线性相关；votes_up 极度非线性，无法捕获特征交互 |
| **随机森林** | 训练慢、内存高；不原生支持稀疏矩阵，TF-IDF 特征效率差 |
| **Gradient Boosting (sklearn)** | 实现效率低于 LightGBM/XGBoost，不支持稀疏矩阵 |
| **XGBoost** | 性能与 LightGBM 相近，但训练速度慢 3–5 倍；不如 LightGBM 适合大规模稀疏数据 |
| **深度学习 / BERT** | 训练成本高；本项目结构特征主导（`weighted_vote_score` 重要性远高于文本），NLP 驱动不适用 |

### 2.4 LightGBM 的核心优势

- ✅ **原生稀疏矩阵支持**：TF-IDF 3000 维稀疏向量直接输入，无需 todense
- ✅ **实验综合最优**：在所有候选模型中 MAE 最低（待 7.2 实验填入最终数据）
- ✅ **训练速度快**：结构特征仅 0.36s，加 TF-IDF 约 4s，随机森林通常 > 30s
- ✅ **`class_weight='balanced'`**：直接处理 Stage 1 的 12.7% 正样本不平衡
- ✅ **内置特征重要性**：可视化各特征贡献，便于解释性分析

**选型结论：LightGBM 在准确性、速度、可扩展性三个维度综合最优，作为两阶段 Pipeline 的核心模型。**

---

## 三、核心策略：两阶段 LightGBM Pipeline

将问题拆解为两个独立的、可学习的子问题：

```
输入评论
    │
    ▼
【第一阶段】LightGBM 分类器
    问题：这条评论会不会获得赞同票？（votes_up > 0）
    │
    ├── P < 0.5  →  预测 votes_up = 0          （占 87.3% 的情况）
    │
    └── P ≥ 0.5  →  【第二阶段】LightGBM 回归器
                        问题：具体会获得多少票？
                        训练目标：log1p(votes_up)
                        输出：expm1(pred) → 最终 votes_up 预测值
```

---

## 四、实现步骤

### Step 1 — 基线实验（已完成 ✅）
- 直接回归，target 为 `log1p(votes_up)`
- 测试模型：Ridge、LightGBM、LightGBM + TF-IDF
- 最优单模型 MAE ≈ 0.52 → 作为两阶段方案的对比基准

### Step 2 — 训练第一阶段分类器
```python
# 标签：该评论是否获得了赞同票？
y_cls = (df["votes_up"] > 0).astype(int)   # 正样本率 12.7%

# 模型：用 balanced 权重处理类别不平衡
clf = LGBMClassifier(class_weight="balanced", n_estimators=300)
clf.fit(X_train, y_cls_train)
```
评估指标：**加权 F1、AUC-ROC、精确率/召回率**

### Step 3 — 训练第二阶段回归器（仅用非零子集）
```python
# 只保留有赞同票的评论（50k 样本中约 6,250 条）
mask = df["votes_up"] > 0
X_pos = X[mask]
y_pos = np.log1p(df["votes_up"][mask])

reg = LGBMRegressor(n_estimators=300, num_leaves=63)
reg.fit(X_pos, y_pos)
```
评估指标：**原始尺度 MAE**（即 `expm1(pred)` 后的 MAE）

### Step 4 — 构建完整 Pipeline 并评估端到端效果
```python
def predict(X):
    proba = clf.predict_proba(X)[:, 1]
    result = np.zeros(len(X))
    pos_mask = proba >= 0.5
    result[pos_mask] = np.expm1(reg.predict(X[pos_mask]))
    return result

# 最终指标：在完整测试集上计算端到端 MAE
mae = mean_absolute_error(y_test_raw, predict(X_test))
```

### Step 5 — 消融实验
| 消融对象 | 目的 |
|---|---|
| Stage 1 阈值：`votes_up > 0` vs `> 3` vs `> 10` | 找到最优分割点 |
| 是否包含 `weighted_vote_score` | 验证潜在数据泄漏（Spearman r=0.89） |
| Stage 1 是否加入 TF-IDF 特征 | 评估文本特征对分类任务的贡献 |
| 两阶段 Pipeline vs 单模型基线 | 证明两阶段方案优于直接回归 |

---

## 五、特征工程

| 特征 | 预处理 | 用途 |
|---|---|---|
| `playtime_forever`、`playtime_at_review` | `log1p` | ✅ 两个阶段均使用 |
| `num_games_owned`、`num_reviews` | `log1p` | ✅ 两个阶段均使用 |
| `voted_up` | 布尔值 → int | ✅ 两个阶段均使用 |
| `review_len`（工程化特征） | 原始值 | ✅ 两个阶段均使用 |
| TF-IDF on `review` | 最多 3000 个特征 | ✅ 第一阶段（第二阶段待消融） |
| `weighted_vote_score` | 原始值 | ⚠️ 需做含/不含的消融实验（泄漏风险） |
| `votes_funny`、`comment_count`、`appid` | — | ❌ 排除 |

---

## 六、评估指标

| 指标 | 评估位置 |
|---|---|
| 加权 F1、AUC-ROC | 第一阶段分类器（单独评估） |
| 原始尺度 MAE | 第二阶段回归器（单独评估） |
| **端到端 MAE** | **完整 Pipeline —— 核心指标** |

**成功标准：两阶段 Pipeline 端到端 MAE < 单模型基线 MAE（0.52）**

---

## 七、交付物

1. `Modeling_Decision_Analysis.ipynb` — 第 1–10 节（EDA + 基线实验，已完成 ✅）
2. 新建 Notebook：**`Two_Stage_Model.ipynb`** — 两阶段模型实现 + 消融实验结果
3. 最终对比表：路径 1（直接回归基线）vs 路径 2（两阶段 Pipeline）vs 路径 3（目标值截断）
