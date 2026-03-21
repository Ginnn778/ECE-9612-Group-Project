# 模型策略推荐
## Steam 评论帮助度预测 (CS:GO 数据集)

> **日期:** 2026-03-21  
> **数据集:** `big_reviews.csv` — 共 4,374,931 行；用于 EDA 和实验的 50,000 行样本  
> **目标:** `votes_up` (评论获得的赞同票数)

---

## 1. 问题诊断 — 为什么原始回归很难

直接对原始 `votes_up` 进行回归确实很困难。原因如下，基于我们的实际数据：

| 观察 | 证据 |
|---|---|
| 极端右偏 | 原始值的 `votes_up` 偏度 = **36.60** |
| 零值主导 | **87.3%** 的评论有 `votes_up = 0` |
| 仅 12.7% 非零 | “有趣”的少数派驱动了所有信号 |
| 百分位数跳跃 | p99 = 5，但 p100 = 297 — 一个巨大的异常值尾部 |
| 特征相关性弱 | 最佳 Spearman r ≈ 0.21 (`votes_funny`)，大多数特征与 `votes_up` 的相关性 < 0.09 |
| 所有模型都停滞不前 | Ridge MAE ≈ 0.52, LightGBM ≈ 0.52, LightGBM+TF-IDF ≈ 0.52 (第 9 节结果) |

即使在进行 `log1p` 变换后（对数空间中 MAE 从 0.54 降至 0.37），实际尺度的 MAE 几乎没有变化，因为 87% 的预测只是预测“零”。

---

## 2. 评估两个提议

### 2.1 提议 A — 目标值封顶 (截断回归)

**想法：** 将 `votes_up` 封顶到某个上限（例如，99 百分位数 = 5），然后对封顶后的值进行回归。

**评估：⚠️ 部分有用，但有限**

- **优点：** 降低了极端异常值的影响；稍微简化了回归任务。
- **缺点：** 封顶值是任意的。将 5 作为上限会丢失关于拥有 10、50、297 个赞同票的评论的所有信息 — 这些正是理解帮助度最关键的“病毒式”评论。
- **缺点：** 未解决零值膨胀问题。87.3% 的数据目标值仍为 0。
- **结论：** 可作为预处理步骤（为训练稳定性裁剪异常值），但**不建议作为独立策略**。

---

### 2.2 提议 B — 两阶段模型 (分类器 + 回归器)

**想法：**
1. **阶段 1 (二元分类器)：** 预测评论是否为“高质量评论”（即是否会获得赞同票）。
2. **阶段 2 (回归器)：** 在阶段 1 预测为“是”的子集中，预测确切的 `votes_up` 计数。

**评估：✅ 方向正确，需要完善**

**优点：**
- 通过将“评论是否会获得关注？”的问题与“获得多少关注？”的问题分开，直接解决了零值膨胀问题。
- 阶段 2 在一个更干净、偏斜度更低的子群体上进行训练。
- 架构上可解释且模块化 — 每个阶段都可以独立调整。

**顾虑与完善：**
- **阈值选择：** `votes_up > 10` 是一个合理的起点，但在我们 50k 样本的 p99 = 5 下，`votes_up > 10` 可能非常罕见（约占数据的 1-2%）。考虑使用 `votes_up > 0`（12.7% 的正面率，已探索）或 `votes_up > 3` 作为替代方案。**我们建议首先尝试 `votes_up > 0`**，因为它与现有 EDA 一致。
- **阶段 1 的类别不平衡：** 在 12.7% 的正面率 (`votes_up > 0`) 下，使用 `class_weight='balanced'` — 已验证可将加权 F1 从 0.9694 提高到 0.9704。
- **误差传播：** 阶段 1 的误报会向阶段 2 输入嘈杂的数据。最终的管道评估应使用端到端的 MAE，而不是仅依赖各阶段的指标。
- **阶段 2 的数据量：** 如果阈值是 `votes_up > 10`，阶段 2 的训练数据会急剧减少。确保有足够样本以实现稳定训练。

---

## 3. 我们的建议 — 混合三路径策略

基于 EDA 的发现和实验结果，我们建议**并行推进所有三个路径**并报告比较结果。这将使项目在学术上更丰富，更具说服力。

### 路径 1 — log1p 转换的直接回归 (基线)
> 已在第 9-10 节实现。

- **目标：** `log1p(votes_up)`，使用 `expm1(pred)` 进行评估 → 原始尺度的 MAE
- **模型：** LightGBM (`n_estimators=500`, `num_leaves=63`, `learning_rate=0.05`)
- **特征：** 结构特征 + `voted_up` 布尔值
- **当前结果：** MAE ≈ 0.52 (原始尺度) — 作为基线底线

### 路径 2 — 两阶段模型 (推荐的主要贡献)
> 实现为项目的主要新颖贡献。

**阶段 1 — 二元分类器：“该评论会获得赞同票吗？”**
- **目标：** `(votes_up > 0)` → 二元标签（正面率为 12.7%）
- **模型：** `LightGBM Classifier` 配合 `class_weight='balanced'`
- **特征：** 所有结构特征 + TF-IDF（最多 3000 个特征，sublinear_tf）
- **关键特征（来自重要性分析）：** `weighted_vote_score`, `num_games_owned`, `num_reviews`, `playtime_forever`, `playtime_at_review`

**阶段 2 — 回归器：“它会获得多少赞同票？”**
- **子集：** 仅限 `votes_up > 0` 的行（在 50k 样本中约 6,250 行；在完整数据集中约 560k 行）
- **目标：** 在此子集上进行 `log1p(votes_up)` — 此子群体中的偏度要低得多
- **模型：** `LightGBM Regressor`
- **特征：** 相同的结构特征 + TF-IDF 特征

**推理管道：**
```
Input review
    → Stage 1 classifier → P(votes_up > 0)
        if P < 0.5:  predict votes_up = 0
        if P ≥ 0.5:  → Stage 2 regressor → expm1(pred) → votes_up estimate
```

**评估：** 使用组合管道输出在完整测试集上计算 MAE 和 RMSE。

### 路径 3 — 封顶/分位数目标 (消融实验)

- 训练时将 `votes_up` 的上限设置为 p99 = 5；在未设上限的测试集上进行评估
- 或者重构为**有序回归**：预测分组 {0, 1–2, 3–5, 6–10, >10}
- 用作消融实验，以展示为何未设上限的两阶段模型优于截断

---

## 4. 特征工程建议

基于第 2-10 节的分析：

| 特征 | 建议 | 原因 |
|---|---|---|
| `playtime_forever`, `playtime_at_review` | ✅ 保留，应用 `log1p` | 偏度=110–115，log1p 降至 ~0.3 |
| `num_games_owned`, `num_reviews` | ✅ 保留，应用 `log1p` | 与目标的 Spearman r=0.08–0.09；偏度 ~15–25 |
| `weighted_vote_score` | ✅ 保留 | 顶部特征重要性 (2068)；与 `votes_up` 的 Spearman r=0.89 — **但请检查数据泄露** |
| `voted_up` | ✅ 保留为布尔值 | 有用信号；重要性=165 |
| `votes_funny` | ⛔ 排除 | Spearman r=0.21 但与 `votes_up` 共现 → **数据泄露** |
| `comment_count` | ⛔ 排除 | 98.7% 为零，信息量接近于零 |
| `appid` | ⛔ 排除 | 常量 (仅限 CS:GO) |
| `last_played` | ⚠️ 谨慎使用 | Unix 时间戳，而非游戏时长 |
| TF-IDF on `review` | ⚠️ 条件性 | Max_features=3000，sublinear_tf。在单模型回归中仅带来 -0.2% 的增益，但可能对 Stage 1 分类器更有意义 |
| 评论长度 (`review_len`) | ✅ 工程化 | 与目标的 Spearman r=0.09；计算成本低 |
| 语言 (`language`) | ✅ One-hot 前 10 | 英语占主导 (48%)，但语言显示出一些中位数方差 |

> ⚠️ **关于 `weighted_vote_score` 的重要提示：** 此特征与 `votes_up` 的 Spearman r=0.89。它由 Steam 使用投票数计算得出，因此几乎可以肯定是**泄露特征**（部分源自 `votes_up` 本身）。务必包含一个排除该特征的无泄露消融实验。

---

## 5. 模型选择总结

| 阶段 | 任务 | 推荐模型 | 原因 |
|---|---|---|---|
| 阶段 1 | 二元分类 | **LightGBM 分类器** | 处理稀疏 TF-IDF，类别不平衡，速度快 |
| 阶段 1 (基线) | 二元分类 | 逻辑回归 | 可解释，快速健全性检查 |
| 阶段 2 | 回归 (votes_up > 0) | **LightGBM 回归器** | 第 9 节中表现最佳；处理偏斜的 log 目标 |
| 阶段 2 (基线) | 回归 | Ridge 回归 | 速度快，已验证 (log 空间 MAE=0.37) |
| 可选 | 全程回归 | LightGBM + TF-IDF | 已实现；用作路径 1 基线 |

---

## 6. 评估指标

| 指标 | 用例 |
|---|---|
| **MAE (原始尺度)** | 主要回归指标 — 惩罚绝对误差 |
| **RMSE (原始尺度)** | 次要 — 对大异常值更敏感 |
| **加权 F1** | 阶段 1 分类器 — 考虑类别不平衡 |
| **精确率/召回率 @ 阈值** | 阶段 1 — 调整阈值以平衡精确率与召回率 |
| **端到端 MAE** | 最终流水线评估 — 最重要的单一数字 |

---

## 7. 推荐实施路线图

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

## 8. 总结

| 提案 | 结论 | 操作 |
|---|---|---|
| 原始回归 (当前) | ⚠️ 困难但有效的基线 | 保留为路径 1 |
| 目标值上限 | ⚠️ 有限；丢失尾部信息 | 仅用作消融实验 |
| 两阶段模型 | ✅ **推荐主方法** | 实现为路径 2 |
| 两阶段阈值 | 从 `votes_up > 0` 开始 | 与 `> 3`, `> 10` 进行消融实验 |
| 模型选择 | 两阶段均使用 LightGBM | 已在实验中验证 |
| TF-IDF 文本特征 | 回归增益边际；在分类器中测试 | 包含在阶段 1 实验中 |
| `weighted_vote_score` | 重要性高但**潜在泄露** | 运行包含和不包含它的消融实验 |