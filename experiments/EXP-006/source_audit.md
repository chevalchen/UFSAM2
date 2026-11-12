# EXP-006 论文与实现核对（2026-11-12）

## 来源及版本

- [SANSA 论文](https://arxiv.org/abs/2505.21795)、[官方 SANSA](https://github.com/ClaudiaCuttano/SANSA/tree/f562b5642940e08a57e3da2608b804b99af3763b)。本仓库 `000d450` 导入；EXP-006 的 `sansa.py` 相对导入只增加已有的显式 checkpoint 路径与延迟下载导入，推理计算未改。
- [PR-MaGIC 论文 v1](https://arxiv.org/html/2604.12113v1)、[官方代码](https://github.com/POSTECH-MinjaeLee/PR-MaGIC/tree/9616a96ee55ff984607b268f2624cd234239ce6a)。服务器实际 clone 到 `/tmp/exp006-prmagic`，核对 `Personalize-SAM/pr_magic_for_persam.py` 与 `Matcher/matcher/Matcher_PR.py`。
- 历史 MTP `219ac83` 的采点阈值为正 `.65`、负 `.35`，9×9 前景腐蚀（空则保留原前景）；负点取其他 multimask 在已选 mask 外的最大概率。质量为 `stability(logits,±.05) − .25*mean_abs_multimask_probability_disagreement`。只迁移这些规则，未迁移门控或旧 baseline。

## 公式和代码的具体差别

1. 论文式 (12) 是 `z[t+1]=z[t]+eta*grad(d)+sqrt(2*gamma*eta)*noise`，再以式 (13–14) 重匹配采点，式 (15) 始终用原始 `z0` 解码。**不是以梯度幅值直接定位点**。
2. `pr_magic_for_persam.py:505–525` 对选中低分辨率 logits 传入全 1 `grad_outputs`，等价于 logits **求和**，无 GT、无高置信锚点/正负对比/显式 KL loss。梯度逐元素裁剪到 ±.1，累积 `grad_map`，并加入上述噪声；解码用原始 predictor features，采点用 `features+grad_map`。梯度本身也在原始 features 上求，区别于公式逐步在更新后 `z[t]` 求导。
3. Matcher 实现对合并所用 logits 求和；把 SAM 梯度增量插值到 DINO 网格并通过截断/补零匹配通道，然后重做 patch matching/clustering。这不是学习到的 SAM→DINO 语义映射。SANSA 无 DINO 分支，不能直接照搬该映射。
4. 论文式 (1) 写 `KL(mu||rho)`；式 (3) 的 `rho*grad(log(rho/mu))` 漂移通常来自 `KL(rho||mu)` 的 Wasserstein 梯度流。这是理论表达的方向差异；代码没有计算任一密度 KL。把 decoder 当密度比判别器是代理假设，不能据此保证纠错方向。
5. 选择器式 (16–19) 对**像素掩蔽后的图像重新编码**再做全局相似度。官方 PerSAM 代码确实重新编码；其 `curr_mask` 来自本轮首个 `logit_high`，而计入候选 IoU 的是后续 cascade 的 `masks_np3[best_idx]`，两者并不完全相同。Matcher 也重新编码像素掩蔽的图像。不能把缓存 feature 的 mask pooling 当作其等价复现。
6. 官方 README 2026-10-13 更正 PerSAM 额外点的 `(row,col)`/`(x,y)` 转置及去重索引。arXiv v1 表格仍是旧版结果，不用于本项目增益证据。当前正/负标签与点预算也不同于 PerSAM 多正点和 Matcher 聚类点。

## 本轮实现及可解释性边界

G2 使用 `J=mean(z[A+])-mean(z[A-])`；A 为 M1 停止梯度的 `.9/.1` 高置信、16×16 分区极值，每域最多32个。只对临时 `F0.clone()` 求导，L2 通道梯度上采样后在 MTP 正/负域内排序。均值让两域不按面积加权；不加入噪声，不更新解码特征。它衡量“当前决策对特征的敏感位置”，**没有给出应补前景还是背景的有符号纠错依据**。正点域被 M1 高置信前景约束，按构造几乎不能覆盖漏分区；梯度可偏向边界，也可能强化现有误判。这些是要由 GT 点、oracle 和打乱对照检验的限制。

H2 是前向语义诊断：memory attention 前的 SANSA support 前景特征按 support 各自归一化平均构成原型，与同层 query 特征做 cosine；全图最大相似位置作正点，M1 前景内最低相似位置作负点。仍用原始 F0 解码，额外只一次 head，无反向。其搜索域与 G2 不同，因此只能说明扩大语义搜索是否有希望，不能单独归因于梯度。

首轮未实现 stochastic PR-MaGIC 复现：将 post-memory F0 与 pre-memory support 表征直接作 cosine 会混用表征空间；若后续增加特征更新版，应明确定义支持表征、梯度求值点、步长/噪声、同域前向控制和候选选择成本。
