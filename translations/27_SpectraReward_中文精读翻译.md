# 《Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation》中文精读翻译

## 论文信息

- **英文标题**：Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation
- **中文标题**：Read It Back：预训练 MLLM 即文生图零样本奖励模型
- **作者**：Runhui Huang、Qihui Zhang、Zhe Liu、Yu Gao、Jie Wu、Hengshuang Zhao
- **机构**：香港大学（The University of Hong Kong）、字节跳动 Seed（ByteDance Seed）、北京大学（Peking University）
- **arXiv 链接**：<https://arxiv.org/abs/2607.11886>
- **提出的框架**：SpectraReward（免训练零样本奖励：图像条件化的提示似然）

## 一句话核心总结

SpectraReward 把"图像条件化的提示平均对数似然"——即将生成图像喂给冻结的预训练 MLLM、再以 teacher forcing 把原提示"读回来"——作为免训练、零样本的文生图强化学习奖励，直接复用 MLLM 预训练习得的图文对齐能力；其特例 Self-SpectraReward 进一步让统一多模态模型用自身理解分支为自身生成分支打分，构成无需外部奖励模型的闭环自我提升框架，并实证揭示"奖励-策略对齐"这一因素足以匹敌乃至超越奖励模型的规模。

---

## 摘要

本文提出 **SpectraReward**——一个免训练（training-free）的奖励函数，可将预训练多模态大语言模型（Multimodal Large Language Model, MLLM）变成图像生成强化学习即开即用（off-the-shelf）的奖励模型。SpectraReward 不要求 MLLM 对生成图像作出评判，也不要求其回答分解出的验证问题，而是度量**原始提示能从生成图像中被恢复得多好**：只需一次以图像为条件、teacher-forced 的前向传播。我们使用图像条件化的提示平均对数似然作为奖励，直接复用 MLLM 预训练习得的图文对齐能力，无需偏好标注（preference labels）、无需奖励模型微调。我们进一步提出 **Self-SpectraReward**——统一多模态模型（unified multimodal models）下的一个特例：策略自身的理解分支充当其生成分支的奖励模型，构成一个不依赖外部奖励模型、也不依赖外部知识的闭环自我提升框架。广泛的实验验证了 SpectraReward：这是一项覆盖两个扩散模型、三种 RL 算法、来自四个 MLLM 家族（参数量 4B 至 235B）的九个奖励 MLLM 骨干、以及五个分布外（out-of-distribution）文生图基准的大规模图像生成 RL 研究。结果表明，SpectraReward 与 Self-SpectraReward 均显著且一致地提升生成性能，并优于此前基于 MLLM 派生奖励的训练方法。进一步分析揭示：更大的奖励 MLLM 并不总是更好，而 Self-SpectraReward 可以匹敌甚至超越规模大得多的外部奖励模型，说明**奖励-策略对齐**（reward-policy alignment）是图像生成 RL 生效的关键因素。项目主页：<https://huangrh99.github.io/SpectraReward/>

---

## 1 引言

图像生成近年发展迅猛，从专用的文生图模型 (Esser et al., 2024; Labs, 2024) 演进到将视觉理解与生成整合于单一架构的统一多模态模型（unified multimodal models, UMMs）(Deng et al., 2025; Chameleon Team, 2024; Chen et al., 2025; Wang et al., 2024; Cui et al., 2025; Huang et al., 2025; Pan et al., 2025)。与此同时，强化学习（RL）已作为一种有效的后训练（post-training）阶段并行兴起 (Liu et al., 2025; Xue et al., 2025; Xue et al., 2025; Zheng et al., 2025)，持续提升组合保真度（compositional fidelity）与指令遵循能力。然而，一个 RL 配方的成功立于两根互补的支柱之上：优化算法决定长时程训练的稳定性，而奖励模型决定训练策略最终所能逼近的上限。设计一个既高效又可靠的实用奖励模型，至今仍颇具挑战。

近期研究在图像生成的奖励建模方面取得可观进展。一条工作线基于大规模人类偏好标注构建奖励模型，用这些数据将视觉-文本表示或视觉-语言模型与人类判断对齐 (Kirstain et al., 2023; Xu et al., 2023; Wu et al., 2023; Wu et al., 2025; Wang et al., 2025)。这些方法固然有效，却依赖昂贵的标注、困难的数据收集与高成本的迭代。另一条工作线通过把预训练 MLLM 复用为零样本奖励源来绕开偏好训练 (Ku et al., 2024; Li et al., 2025; Huang et al., 2026)。直接的标量或 logit 反馈虽免训练，但对评判者的校准（judge calibration）与打分噪声敏感 (Ku et al., 2024; Li et al., 2025)；问题分解（question-decomposition）流水线通过逐一验证原子化的提示要求提升了可靠性，却引入了可观的工程复杂度 (Huang et al., 2026)。这就留下一个开放问题：**为开源图像生成器设计一个高效、免偏好标注、即开即用的奖励模型**。

本文提出 **SpectraReward**——一个免训练的奖励函数，通过复用预训练 MLLM 的图文对齐知识，把任意预训练 MLLM 变成图像生成奖励模型，无需偏好标注或额外训练。给定一张生成图像，SpectraReward 以该图像为条件驱动一个冻结的 MLLM，并对提示做一次 teacher-forced 前向传播，得到反映"每个文本要求能从图像中被读出程度"的 token 级似然轮廓（likelihood profile）。我们将其称为该图文对的**语义谱**（semantic spectrum），并用提示 token 的平均对数似然把它聚合为一个标量奖励。如此，SpectraReward 把模型自身的图像条件化语言建模目标所诱导的提示似然谱，用作一个确定性的、即开即用的奖励信号（如图 1(a) 所示），而不是要求 MLLM 去评判图像或回答分解出的问题。

我们进一步将 SpectraReward 特化到统一多模态模型（UMMs）中，得到一个称为 **Self-SpectraReward** 的特例。由于 UMM 将图像理解与图像生成整合于单一框架，其理解分支可以直接为其生成分支提供奖励信号。这样，模型利用自身的理解能力改进自身的生成，不依赖任何外部模型或外部知识，构成策略凭借自身内在能力演化的闭环自我提升框架。除消除外部依赖外，该设计还有一个结构性优势：理解分支与生成分支共享同一 tokenizer、视觉编码器与预训练分布，因此它"读"图像的方式天然与生成分支采样图像的分布对齐。我们由此假设：**这种内在对齐使策略自身的理解分支成为其自身生成的一个格外合适的奖励源**。

> **图 1**：SpectraReward 概览。(a) 预训练 MLLM 天然诱导出一个度量生成图像与提示对齐程度的语义谱；SpectraReward 将其聚合为 T2I RL 的奖励。(b) RL 训练期间，SpectraReward 稳步上升，复杂场景生成也可见改善。(c) 本文研究来自四个 MLLM 家族的九个奖励 MLLM 骨干，其中外部奖励 MLLM 覆盖三个家族、4B 至 235B 参数；扩大奖励 MLLM 骨干带来的收益非单调：Qwen3-VL-30B-A3B 在外部 MLLM 中表现最佳，而使用 BAGEL 自身理解分支作奖励模型的 Self-SpectraReward 优于所有外部 MLLM。(d) 与基线相比，SpectraReward 与 Self-SpectraReward 在全部六个下游基准上均带来显著且一致的提升。

我们通过一项大规模图像生成 RL 研究验证 SpectraReward，覆盖两个扩散模型、三种 RL 训练算法、来自四个 MLLM 家族（参数量 4B 至 235B）的九个奖励 MLLM 骨干，并在五个分布外下游文生图基准上评估真实的泛化改进。在 SD3.5-M (Esser et al., 2024) 与 BAGEL (Deng et al., 2025) 上，SpectraReward 均在所有下游基准上带来显著提升，展示了所提提示似然奖励的泛化能力。以 BAGEL 为策略模型时，SpectraReward 与 Self-SpectraReward 分别在 TIIF-Bench (Wei et al., 2025) 上较基线提升 +10.0、在 GenEval (Ghosh et al., 2023) 上提升 +4.3/+5.5。SpectraReward 还在 TIIF-Bench 与 GenEval 上分别以 +6.3 与 +2.1 的优势超过强力 RL 基线 AlphaGRPO (Huang et al., 2026)。此外，与采用相似规模 MLLM 骨干的 SpectraReward 相比，Self-SpectraReward 展现出更佳的改进；与 4 倍或 30 倍规模的 MLLM 相比也取得相当甚至更好的性能——例如相对采用最佳 MLLM 骨干的 SpectraReward，在 GenEval 上 +1.2、GenEval2 (Kamath et al., 2025) 上 +2.1、WISE (Niu et al., 2025) 上 +2。这些结果支持我们的假设：**奖励模型与策略之间的分布对齐，其重要性可与奖励模型的规模比肩**。本文贡献总结如下：

1. 提出 SpectraReward——一个免训练、即开即用的奖励函数，将任意预训练 MLLM 变为文生图奖励模型。通过计算图像条件化的提示 token 似然，SpectraReward 复用预训练图文对齐能力，无需偏好标注、奖励模型微调、标量评判或问题分解。
2. 提出 Self-SpectraReward——统一多模态模型中策略自身理解分支充当奖励模型的特例。它构成无需外部奖励模型与外部知识的闭环自我提升框架，模型利用自身理解能力改进生成性能。
3. 开展一项覆盖两个生成器骨干、三种 RL 算法、来自四个 MLLM 家族（参数量 4B–235B）的九个奖励 MLLM 骨干、五个下游基准的大规模图像生成 RL 研究。结果证明了 SpectraReward 与 Self-SpectraReward 的有效性，并揭示扩大奖励模型的收益并非单调，而良好对齐的自奖励模型可以匹敌甚至超越规模大得多的外部模型。

---

## 2 相关工作

### 2.1 图像生成的奖励模型

基于人类反馈的强化学习（RLHF）(Ouyang et al., 2022) 是对齐大语言模型的标准范式 (Gao et al., 2024; Shao et al., 2024)，并已通过在大规模人类偏好或提示-图像对齐数据上训练的奖励模型适配到文生图模型 (Black et al., 2023) (Xu et al., 2023; Xu et al., 2024; Kirstain et al., 2023; Ma et al., 2025)。这些奖励模型提供了关键的对齐信号，但其有效性从根本上受制于一种对与文本无关的美学细节的固有偏好偏差 (Ba et al., 2025)。为突破这些容量上限，第二类研究把奖励架构升级为微调的 MLLM，成功抬高了性能上限 (Xu et al., 2024; Wu et al., 2025; Gong et al., 2025)。但这需要大量标注数据与可观的训练开销，且不可避免地使模型偏离 MLLM 的预训练知识分布。最近，一种新兴范式同时摒弃了偏好标注与奖励模型微调：把预训练 MLLM 用作零样本评判器，直接评估图像质量与美学推理 (Jiang and Chen, 2025; Schoepp et al., 2025)。这类方法通常要么输出直接的标量分数，要么把生成提示分解为由 MLLM 回答的可验证原子问题 (Huang et al., 2026)。并行工作 PromptEcho (Liu et al., 2026) 探索了这一概念的交叉熵形式，但其评估局限于特定领域的文生图骨干。

### 2.2 统一多模态模型中的自我奖励

自我奖励（self-rewarding）机制——模型利用自身评估来监督自身训练——已在 LLM 对齐中展现出显著成效 (Yuan et al., 2024; Wu et al., 2025; Sun et al., 2023)。该范式自然延伸到天然兼具理解与生成能力的统一多模态模型（UMMs）(Chameleon Team, 2024; Xie et al., 2024; Wu et al., 2024; Zhao et al., 2025)。在以自生成偏好数据与迭代自进化对齐多模态模型的更广泛努力之外 (Tan et al., 2025; Jiang et al., 2024)，近期并行研究探索了自我奖励与 UMM 的具体交汇：SRUM (Jin et al., 2025) 引入对 UMM 理解分支全局与局部信号的聚合；GvU (Pan et al., 2026) 利用理解分支的 token 级 logits 作为密集奖励信号。尽管这些研究与"策略的内在理解能有效指导其自身生成"的直觉一致，它们都局限于 UMM 内部的评估，未考察这一奖励与外部奖励模型相比孰优孰劣。

---
## 3 方法

本节首先在 3.1 节介绍 SpectraReward 奖励函数，随后在 3.2 节将其特化到统一多模态模型得到 Self-SpectraReward，最后在 3.3 节分析该奖励信号的判别性质。图 2 将 SpectraReward 与此前基于 MLLM 的奖励函数进行了比较。

> **图 2**：基于 MLLM 的奖励函数比较。(a) 标量打分（scalar scoring）直接让 MLLM 给出一个离散的图文对齐分数，使奖励对评判校准与打分噪声敏感。(b) VQA 分解（VQA decomposition）把提示转换为原子问题并聚合验证器的置信度，但引入两阶段流水线，且依赖问题分解的质量。(c) SpectraReward 通过一次 teacher-forced 前向传播计算图像条件化的提示似然；所得 token 级似然构成语义谱，其平均值用作标量奖励。(d) Self-SpectraReward 在统一多模态模型内部实例化同一提示似然奖励，使用策略自身的理解分支，免去外部奖励 MLLM 并改进奖励-策略对齐。

### 3.1 SpectraReward

设 $G_\theta$ 为文生图策略，从提示 $x=(x_1,\ldots,x_T)$ 采样图像 $y\sim G_\theta(\cdot\mid x)$；设 $\mathcal{M}$ 为冻结的预训练 MLLM。由于预训练 MLLM 已从大规模预训练中学得强大的图文对齐能力，我们的目标是：从预训练 MLLM 中提取一个标量奖励 $R_{\mathcal{M}}(x,y)$，用以度量生成图像 $y$ 对提示 $x$ 的实现程度，同时不需要偏好标注、奖励模型微调或其他辅助流水线。

SpectraReward 的核心思想是**通过一个简单的逆向问题来评估生成图像：从 $\mathcal{M}$ 的视角看，生成图像 $y$ 能被多好地"翻译回"提示 $x$**。具体计算上，我们把图像 $y$ 作为视觉条件输入 $\mathcal{M}$，然后对提示 token $x$ 做一次 teacher-forced 前向传播。最终奖励函数如下：

$$
R_{\mathcal{M}}(x,y)=\frac{1}{T-1}\sum_{t=1}^{T-1}\log p_{\mathcal{M}}\!\left(x_{t+1}\mid x_{\le t},y\right). \tag{1}
$$

该奖励即 $\mathcal{M}$ 下**图像条件化的提示平均对数似然**。逐 token 的似然构成该图文对的*语义谱*，描述 $y$ 中的视觉证据如何在各提示 token 上支撑 $x$ 的语义。SpectraReward 将这一语义谱聚合为标量奖励：更高的奖励 $R(x,y)$ 表示提示从图像出发更可预测、生成图像更好地支撑了提示的语义内容。

其关键优势在于：该奖励函数直接调用预训练 MLLM 最根本的能力——理解图像并将其与相应语言描述相关联。因此，SpectraReward 既不训练单独的奖励模型，也不提示 MLLM 产出抽象的判断分数，而是**复用 MLLM 预训练期间已经学得的视觉-语言扎根（visual-language grounding）能力**。

### 3.2 Self-SpectraReward

当 $G_\theta$ 是统一多模态模型时，它同时包含生成分支 $G_\theta^{\mathrm{gen}}$ 与理解分支 $G_\theta^{\mathrm{und}}$。Self-SpectraReward 用 $G_\theta^{\mathrm{und}}$ 作为 $G_\theta^{\mathrm{gen}}$ 所采样图像的奖励 MLLM 来实例化 SpectraReward。换言之，**模型通过度量"原始提示在其自身图像条件化理解分支下的似然"来给自己生成的图像打分**。这构成一个闭环自我提升框架：模型利用自身的理解能力改进自身的生成质量。此外，这种自我奖励设计消除了对外部奖励模型的需要、避免了为服务大规模 MLLM 而付出的额外计算资源，并简化了 RL 训练流水线。

**奖励-策略对齐**。Self-SpectraReward 在同一个统一模型中耦合了两种对偶能力：生成分支把文本映射为图像，而奖励函数用理解分支度量生成图像能被多好地翻译回原始文本。由于两个分支共享同一 tokenizer、视觉编码器与预训练分布，奖励信号与策略自身的生成知识天然兼容。这并不要求理解分支在一般视觉理解上强于外部 MLLM；相反，它提供的是一个对**策略自身生成图像分布**校准更好的奖励信号。用这一自奖励优化生成分支，会促使文生图策略与其自身的图生文理解能力更加一致。我们将在 4.3 节进一步分析奖励-策略对齐效应。

> **图 3**：SpectraReward 的 token 级语义敏感性。正例与负例图像用同一正例提示评估。对属性失配（实例化为计数错误），负例图像主要拉低 "Two" 的似然；对物体身份失配（吉他被替换为椅子），"guitar" 的似然骤降。柱状图为图像条件化提示 token 对数似然（误差棒基于四对样本计算），虚线为对应的序列级奖励值（即 SpectraReward）。

### 3.3 SpectraReward 的分析

本节分析三个问题：(1) 语言先验是否影响奖励信号的准确性；(2) token 级似然是否对视觉失配敏感；(3) 得到的标量奖励能否可靠区分同一提示组内的样本。

**语言先默契消（language-prior cancellation）**。一个自然的顾虑是式 (1) 包含 MLLM 对提示的语言先验，因为某些词无论图像如何都更容易预测。一种可能的校正是减去一个纯文本似然项，得到 PMI 风格的奖励 $\log p(x\mid y)-\log p(x)$。但由于本文聚焦于组相对（group-relative）强化学习方法，这一校正并不必要：对固定提示，纯文本似然被同组所有生成图像共享，因此在计算组相对优势时会相互抵消。故 PMI 归一化可能影响跨提示的绝对奖励校准，但不会在每个提示组内部提供额外的优化信号。

**token 级语义敏感性**。我们进一步检验语义谱是否对局部视觉错误作出响应。针对两类常见的提示遵循失败——属性失配（attribute mismatch）与物体身份失配（object identity mismatch）——我们构造正/负图像对。属性失配实例化为计数错误：提示要求两只猫，而负例图像包含五只。对每一对，我们都用原始的正例提示评估两张图像。图 3 显示，似然的下降集中在失配的语义词上：计数失配主要拉低 "Two" 的似然，而把吉他换成椅子则骤降 "guitar" 的似然；虚线所示的序列级奖励对正例图像也保持更高。这表明 token 级提示似然对定向的语义错误敏感，而其平均值（即 SpectraReward 所用的计算方式）仍然提供可靠的标量奖励。

**奖励排序可靠性**。对组相对 RL 而言，重要的是奖励能否正确排序同一提示的不同 rollout。如图 4 所示，SpectraReward 给更好满足提示的图像更高奖励，给存在物体缺失、属性错误或空间关系错误的图像更低奖励。这表明图像条件化的提示似然为文生图 RL 提供了可靠的组内排序信号。

> **图 4**：SpectraReward 的可视化解读。奖励排序与视觉质量排序一致。

---

## 4 实验

### 4.1 实验设置

我们采用 AWM 作为默认强化学习算法（依据 4.4 节的优化器消融，AWM 在所测 RL 算法中总体表现最佳）。策略模型为 BAGEL (Deng et al., 2025)——一个原生 AR-Diffusion 的统一多模态模型，同时针对图像理解与图像生成进行了预训练。除非特别说明，SpectraReward 使用 Qwen3-VL-30B-A3B (Bai et al., 2025) 作为奖励 MLLM 骨干，而 Self-SpectraReward 使用 BAGEL 自身的理解分支计算奖励。

**训练设置**。训练时使用 $T=16$ 个采样步，并从前 10 个去噪步中随机采 6 步用于训练。每个训练步使用 32 个提示，组规模 $G=16$。训练图像分辨率为 $512\times 512$。分类器自由引导（classifier-free guidance）尺度为 4，timestep shift 为 3。学习率为 1e-4。我们在 32 张 A100 GPU 上训练，batch size 为 2、梯度累积为 8。总训练步数为 380。其余超参数遵循 AWM 官方实现。训练数据为 AlphaGRPO20k (Huang et al., 2026)。

### 4.2 主结果

我们在训练期间将 SpectraReward 用作代理奖励（proxy reward），并在五个分布外基准上评估生成性能的真实提升。所有下游基准均使用不同的评测协议与分布外测试集：GenEval (Ghosh et al., 2023)、TIIF-Bench (Wei et al., 2025)、DPG-Bench (Hu et al., 2024)、GenEval2 (Kamath et al., 2025) 与 WISE (Niu et al., 2025)。我们对比的最先进模型包括：生成专用模型 SD3 Medium (Esser et al., 2024) 与 FLUX.1 dev (Labs, 2024)，统一多模态模型 Show-o (Xie et al., 2024)、JanusPro (Chen et al., 2025) 以及我们的基线 BAGEL (Deng et al., 2025)；还与 AlphaGRPO (Huang et al., 2026) 比较——它同样在 BAGEL 上应用 RL 训练，使用其所提出的 MLLM 派生 DVReward。

表 1 将 SpectraReward 与 Self-SpectraReward 同代表性文生图生成器及此前基于 MLLM 派生奖励的训练方法进行了比较。总体而言，SpectraReward 与 Self-SpectraReward 取得相当的性能，并在所有被评估基准上一致超越 BAGEL 基线与 AlphaGRPO。具体而言，在 512 分辨率下，SpectraReward 在 TIIF-Bench 整体短/长提示上较 BAGEL 提升 +10.0/+6.2，Self-SpectraReward 取得相当的 85.1/84.3，并在 GenEval 上进一步带来 +5.5 的增益。与 AlphaGRPO 相比，SpectraReward 在 TIIF-Bench 短/长提示上提升 +6.3/+5.3、在 GenEval 上提升 +3.3；Self-SpectraReward 则分别提升 +6.2/+4.8 与 +4.5。

与 AlphaGRPO (Huang et al., 2026) 一致的是，尽管训练在 512 分辨率下进行，改进同样迁移到 1024 分辨率推理。具体而言，Self-SpectraReward 在 GenEval 上达到 89.8、在 GenEval2 上达到 34.3，均超过 BAGEL 与 AlphaGRPO。更重要的是，Self-SpectraReward 在知识接地基准 WISE 上取得 0.76。这些结果表明，SpectraReward 为文生图 RL 提供了比分解式验证（decompositional verification）更有效的监督信号。

**表 1：文生图基准上的结果。** S 与 L 分别表示短提示与长提示。对 WISE，第二个分数表示使用 CoT 的评估。AlphaGRPO 训练于推理式文生图生成任务。加粗表示最佳性能。BAGEL 的结果为复现。

| 模型 | Basic S | Basic L | Adv. S | Adv. L | Designer S | Designer L | 整体 S | 整体 L | WISE | DPG-Bench | GenEval | GenEval2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| *生成专用模型* | | | | | | | | | | | | |
| SD3 Medium | 78.3 | 77.8 | 61.5 | 59.6 | 63.2 | 67.3 | 64.8 | 64.8 | 0.4 | 84.1 | 74.0 | 21.3 |
| FLUX.1 dev | 83.1 | 78.7 | 65.8 | 68.5 | 70.7 | 71.5 | 71.1 | 71.8 | 0.5 | 83.8 | 82.0 | 21.1 |
| *统一多模态模型* | | | | | | | | | | | | |
| Show-o | 73.1 | 75.8 | 55.0 | 50.9 | 53.7 | 50.4 | 59.7 | 58.9 | 0.35 | - | 69.0 | - |
| JanusPro | 79.3 | 78.3 | 59.7 | 58.8 | 65.8 | 60.3 | 66.5 | 65.0 | 0.35 | 84.2 | 80.0 | - |
| *512 分辨率推理* | | | | | | | | | | | | |
| BAGEL | 81.7 | 86.1 | 73.7 | 77.6 | 84.7 | 82.1 | 75.2 | 78.6 | - | 85.0 | 84.0 | 23.6 |
| AlphaGRPO | 85.5 | 84.2 | 77.4 | 78.9 | 84.3 | 86.6 | 78.9 | 79.5 | - | 86.0 | 85.0 | - |
| SpectraReward | 87.7 | 87.4 | **84.9** | 85.4 | 88.1 | **90.3** | **85.2** | **84.8** | - | **88.1** | 88.3 | 31.8 |
| Self-SpectraReward | **89.7** | **87.6** | 83.7 | **85.7** | **90.3** | 86.2 | 85.1 | 84.3 | - | 87.7 | **89.5** | **33.9** |
| *1024 分辨率推理* | | | | | | | | | | | | |
| BAGEL | 83.4 | 83.7 | 75.2 | 76.7 | 79.8 | 73.5 | 76.4 | 76.2 | 0.53/0.70 | 85.1 | 86.6 | 23.5 |
| AlphaGRPO | 84.8 | 85.9 | 79.9 | 78.4 | 80.2 | 80.2 | 78.7 | 79.5 | 0.50/0.69 | 85.9 | 87.4 | - |
| SpectraReward | **88.4** | 87.3 | **83.2** | 81.8 | 82.5 | 81.0 | **83.3** | 81.7 | **0.53**/0.74 | 86.5 | 88.3 | 32.6 |
| Self-SpectraReward | 87.4 | **87.6** | 81.5 | **83.2** | **87.7** | **84.7** | 82.7 | **82.5** | 0.52/**0.76** | **87.0** | **89.8** | **34.3** |

（译注：表中 Basic/Advanced/Designer/整体为 TIIF-Bench 的四个子项，各分短（S）/长（L）提示；GenEval2 列为 Soft TIFAGM 指标；加粗为各推理分辨率块内最佳，512 块内 WISE 未评估。）

---
### 4.3 奖励 MLLM 骨干的影响

我们接下来研究 SpectraReward 中不同奖励 MLLM 骨干的效果，覆盖不同家族与规模。表 2 报告了实验结果。

**SpectraReward 在扩散模型与奖励 MLLM 家族间的一致提升**。在 SD3.5-M (Esser et al., 2024) 与 BAGEL 上，应用 SpectraReward 均能改进基线，凸显了这一似然奖励的架构无关性（architecture-agnostic）。在 BAGEL 上，来自不同家族的不同奖励 MLLM——包括 Gemma3 (Gemma Team, 2025)、InternVL3.5 (Wang et al., 2025) 与 Qwen3-VL (Bai et al., 2025)——均在 TIIF-Bench 上较基线带来提升。这表明 SpectraReward 能够从多样的 MLLM 骨干中提取有用的奖励信号。

**表 2：不同奖励 MLLM 骨干的效果。** 除非特别说明，奖励 MLLM 均使用 Instruct 版本。加粗表示最佳性能，次佳以下划线标出（本表以斜体表示）。

| 基座模型 | 奖励 MLLM | GenEval↑ | TIIF-Short↑ | TIIF-Long↑ |
| --- | --- | --- | --- | --- |
| SD3.5-M | - | 79.8 | 74.0 | 73.2 |
| SD3.5-M | Qwen3-VL-8B | 85.7 | 84.4 | *84.6* |
| BAGEL | - | 84.0 | 75.2 | 78.6 |
| BAGEL | Gemma3-4B | 83.6 | 83.1 | 83.0 |
| BAGEL | Gemma3-12B-Pretrain | *88.7* | 84.3 | 82.6 |
| BAGEL | Gemma3-12B | 87.5 | 82.0 | 82.6 |
| BAGEL | InternVL3.5-8B | 84.5 | 80.4 | 80.0 |
| BAGEL | InternVL3.5-14B | 84.9 | 78.5 | 79.9 |
| BAGEL | Qwen3-VL-8B | 87.8 | 85.0 | 83.8 |
| BAGEL | Qwen3-VL-30B-A3B | 88.3 | **85.2** | **84.8** |
| BAGEL | Qwen3-VL-235B-A22B | 86.8 | 84.3 | 83.4 |
| BAGEL | BAGEL | **89.5** | *85.1* | *84.3* |

**Self-SpectraReward 表明奖励-策略对齐可以匹敌外部规模**。以 BAGEL 自身理解分支作为奖励 MLLM，Self-SpectraReward 在相似规模的奖励模型中取得最佳结果，匹敌最强的 30B 级外部奖励，并超过 30 倍规模的 MLLM，即 Qwen3-VL-235B-A22B (Bai et al., 2025)。这说明奖励质量并非仅由规模决定。我们将 Self-SpectraReward 的成功归因于其与策略的分布对齐：它与生成器共享 tokenizer、视觉编码器与预训练分布，因此其"阅读"生成图像的方式天然契合策略采样图像的分布。

**奖励 MLLM 的规模有帮助，但并非单调**。在 Qwen3-VL 家族内，从 8B 扩展到 30B 改进了所有报告指标，但进一步扩展到 235B 反而明显下降。在 Gemma3 家族内，从 4B 扩展到 12B 改进了 GenEval，却未能一致改进 TIIF-Bench。我们假设：更大的 MLLM 可能把更多容量投入高级视觉-语言推理与指令遵循任务，而 SpectraReward 所需的更根本的图像条件化描述（captioning）能力可能已在中等规模上饱和。这些结果表明，一个 30B 的 MLLM 就足以解锁 SpectraReward 的大部分收益。

**预训练阶段的 MLLM 可以是更好的 SpectraReward 奖励骨干**。Gemma3-12B-Pretrain 在三个被评估基准上一致优于 Gemma3-12B-Instruct。我们假设：预训练所学习的大规模图像描述与图文对齐目标同我们的奖励函数紧密匹配，而指令微调强调的是高级视觉任务与多任务指令遵循——这些并非 SpectraReward 所直接需要的能力。

### 4.4 消融研究

**RL 算法的影响**。我们在 Self-SpectraReward 设置下比较三种图像生成强化学习方法：FlowGRPO (Liu et al., 2025)、AWM (Xue et al., 2025) 与 DiffusionNFT (Zheng et al., 2025)。表 3 显示 AWM 取得最佳的总体下游性能，优于基于 SDE 的 FlowGRPO 与 DiffusionNFT，因此我们在主实验中采用 AWM。

**表 3：不同 RL 算法与奖励模型的比较。** 以 Self-SpectraReward 训练时，AWM 取得最佳下游性能。在可比的 RL 训练下，Self-SpectraReward 进一步优于此前的奖励模型。加粗表示最佳性能。

| RL 算法 | 奖励模型 | GenEval↑ | TIIF-Short↑ | TIIF-Long↑ |
| --- | --- | --- | --- | --- |
| - | - | 84.0 | 75.2 | 78.6 |
| AlphaGRPO | HPSv3 | 83.4 | 78.5 | 77.1 |
| AlphaGRPO | UnifiedReward | 83.7 | 79.2 | 77.3 |
| AlphaGRPO | VIEScore | 81.7 | 79.1 | 77.9 |
| AlphaGRPO | DVReward | 85.0 | 78.9 | 79.5 |
| AWM | DVReward | 88.0 | 83.3 | 83.7 |
| FlowGRPO | Self-SpectraReward | 85.3 | 82.1 | 79.8 |
| DiffusionNFT | Self-SpectraReward | 89.1 | 84.8 | 83.5 |
| AWM | Self-SpectraReward | **89.5** | **85.1** | **84.3** |

**奖励函数的讨论**。我们将 SpectraReward 与两种替代的 MLLM 图文对齐奖励函数进行比较（如图 5 所示）。第一种是**标量打分**：直接让 MLLM 用 1–5 分评价图文对齐。第二种是 **VQA-Score**：提出一个是非问题如 "Does this figure show {text}?"，用 "yes" token 的概率作为奖励。表 4 显示奖励函数对 RL 性能影响巨大：标量打分使所有基准显著退化（GenEval 较基线下降 6.3），VQA-Score 在 TIIF-Bench 上带来小幅增益但无法改进 GenEval。相比之下，SpectraReward 使用的图像条件化提示似然在所有指标上取得最佳性能，表明**基于似然的奖励是利用 MLLM 进行图像生成 RL 的更有效方式**。

**表 4：奖励函数的消融。** 标量打分让 MLLM 给图像打 1–5 分；VQA-Score 询问 "Does this figure show {caption}?" 并以 $P(\text{yes})$ 作为奖励。

| 奖励函数 | GenEval↑ | TIIF-S↑ | TIIF-L↑ |
| --- | --- | --- | --- |
| - | 84.0 | 75.2 | 78.6 |
| 标量打分（1–5） | 77.7 | 67.5 | 76.0 |
| VQA-Score（$P(\text{yes})$） | 83.1 | 77.4 | 78.9 |
| 提示似然（SpectraReward） | **89.5** | **85.1** | **84.3** |

**序列级 vs. token 级优势**。我们比较默认的序列级优势（sequence-level advantage）与 token 级优势（token-level advantage）。前者先把 token 对数似然平均为一个序列级奖励，再计算组相对优势；后者把每个 token 位置视为独立的奖励信号，计算逐 token 的组优势，再将其平均为序列优势。我们测试了三种 token 级标准差的归一化范围：全局 std、组内全部文本 token 的组 std、以及逐 token std。尽管 token 级优势可以强调 rollout 间差异更大的视觉扎根 token，表 5 显示其相对序列级优势并无明显提升。我们默认使用序列级奖励以获得更稳定的训练。

**表 5：奖励粒度的消融。** token 级奖励未能一致超越序列级奖励，因此我们默认使用序列级奖励。**

| 方法 | GenEval↑ | TIIF-S↑ | TIIF-L↑ |
| --- | --- | --- | --- |
| 序列级优势 | **89.5** | **85.1** | 84.3 |
| token 级优势 | 88.4 | 84.9 | **85.1** |
| + 组 std | 88.5 | 84.0 | 84.8 |
| + 逐 token std | 88.8 | 84.1 | 82.1 |

> **图 5**：定性比较。

---

## 5 结论

我们提出了 SpectraReward——一个免训练的奖励函数，通过度量图像条件化的提示似然，把预训练 MLLM 变成图像生成强化学习即开即用的奖励模型。SpectraReward 不依赖海量偏好标注与奖励模型微调，而是把 MLLM 诱导的 token 级语义谱聚合为一个简单而有效的奖励。我们进一步为统一多模态模型提出 Self-SpectraReward：策略自身的理解分支充当其生成分支的奖励模型，构成无需外部奖励模型的闭环自我提升框架。跨生成器骨干、RL 算法、奖励 MLLM 骨干与分布外基准，SpectraReward 显著且一致地改进文生图生成，并优于此前基于 MLLM 派生奖励的训练方法。结果还表明：更大的奖励 MLLM 并不总是更好，而良好对齐的自奖励可以匹敌甚至超越规模大得多的外部奖励模型。这些发现说明，**奖励函数与奖励-策略对齐是统一多模态生成强化学习的关键因素**。

---

## 附录要点（压缩概括）

1. **局限性（附录 A）**：SpectraReward 依赖预训练 MLLM 的图像条件化提示似然，奖励质量受所选骨干视觉理解与推理能力的上界约束；似然仅在输入提示上计算，主要捕捉图文间的**显式**语义对齐，可能弱化未直接写入文本的隐式视觉蕴含（如 "hot coffee" 经物理与常识推理蕴含蒸汽，但这在原提示 token 的似然中反映微弱）。缓解方向：使用具备丰富世界知识的更强 MLLM 骨干，或把 SpectraReward 扩展到推理式文生图生成（中间推理可将隐式要求显式化）。此外它不直接优化美学与安全等准则，可与互补的专用奖励模型组合使用。
2. **奖励提示格式（附录 B）**：默认不为奖励 MLLM 添加任何"请描述图像"类通用前缀——前缀虽会改变绝对似然值，但不实质影响每个提示组内的相对排序，而组相对 RL 使用的正是后者。唯一例外是 InternVL3.5 家族：经验上添加前缀 "Describe the image" 可使奖励值更稳定。
3. **EOS 掩码（附录 B、C，表 6）**：[EOS] token 的似然常由序列终止行为而非图文对齐主导，且在短提示中占据平均似然的很大比重、会显著偏移最终奖励，因此计算 SpectraReward 时剔除 EOS。消融显示：剔除 EOS 使 GenEval 从 88.5 升至 89.5，TIIF-Bench 基本持平（85.5→85.1 / 84.3→84.3）。
4. **VAE 特征（附录 B、C，表 7）**：BAGEL 的理解分支可接收两种视觉输入——语义编码器的 ViT 特征与生成编码器的 VAE 特征。Self-SpectraReward 默认包含 VAE 特征，因消融显示其平均性能更佳：GenEval 87.8→89.5、TIIF-Short 83.6→85.1（TIIF-Long 84.7→84.3 略降）。（译注：附录 C 正文称"改进在 TIIF-Long 上更显著"，与表 7 数值不符，疑为 TIIF-Short 之笔误。）
5. **GenEval 细分（附录 D，表 8）**：512 与 1024 两个推理分辨率下，SpectraReward 与 Self-SpectraReward 的总分均超过 BAGEL 与 AlphaGRPO；最大改进集中在组合性更强的类别——计数（BAGEL 74.1→Self-SR 84.4/88.8）、空间位置（71.0→81.5/82.3）与颜色-属性绑定（74.8→81.5/78.3）。Self-SpectraReward 在两个分辨率均取得最佳总分（89.5/89.8），说明策略自身理解分支可为结构化提示遵循提供对齐格外好的信号。
6. **TIIF-Bench 细分（附录 D，表 9）**：在基础属性/关系遵循、属性+关系/推理的高级组合以及设计导向的真实场景分割上均大幅超越基线；文本（Text）子类绝对值仍具挑战性，但两方法较 BAGEL 均大幅提升（512 下短提示文本子类 33.03→66.97/60.63），表明图像条件化提示似然对细粒度指令细节同样提供有效监督。
7. **DPG-Bench 细分（附录 D，表 10）**：因 BAGEL 本就是该基准的强基线，绝对边际较 TIIF-Bench 小；SpectraReward 仍取得 512 分辨率最佳总分（88.08）并改进全局、关系与其他类别，Self-SpectraReward 给出最强属性分（92.12）；1024 分辨率下 Self-SpectraReward 在被比较的 BAGEL 系方法中总分最佳（86.97）。
8. **WISE 知识接地（附录 D，表 11）**：不使用 CoT 时各方法得分接近（BAGEL 0.52、SpectraReward 0.53、Self-SR 0.52）；结合 self-CoT 后，两方法在相同推理策略下明显超越 BAGEL 与 AlphaGRPO（0.70/0.71），Self-SpectraReward w/ Self-CoT 取得最佳总分 0.76，并在空间（0.82）、生物（0.73）、物理（0.84）、化学（0.69）等多个知识域改进——说明 SpectraReward 在提升提示遵循的同时，也帮助模型更好地响应 CoT 文本。
9. **更多可视化（附录 E，图 6/7）**：同一 rollout 组内，SpectraReward 奖励更高的样本更好地满足物体、属性与空间约束；与 BAGEL 基线的定性比较中（红字标注关键差异），Self-SpectraReward 更常保持所要求的物体间关系（如把玩具车放到橡皮泥上、把钢笔与织物毯分开），覆盖空间关系、计数、属性绑定、相对大小与长提示组合等情形，说明学得的奖励改进的是细粒度提示扎根，而非仅全局图像质量。
10. **更广泛影响（附录 F）**：复用冻结的预训练 MLLM 免去偏好标注与奖励微调，降低了图像生成 RL 的门槛，使标注/训练资源有限的研究者也可开展提示遵循、组合生成与统一多模态自我提升研究；但同时，更强的提示遵循可能被滥用于生成误导性视觉内容、模仿受保护风格或产出不安全图像，奖励也可能继承骨干 MLLM 的偏差与视觉误判，闭环自奖励更促使模型向自身理解分支对齐而未必符合人类偏好与安全要求——建议部署时配合标准安全过滤、偏差审计与人工评估。

---

## 创新点提炼

1. **"读回"（read it back）式零样本奖励：图像条件化提示似然（式 1）**。把奖励设计从"让 MLLM 当裁判"翻转为"让 MLLM 当读者"——冻结的 MLLM 以生成图像为条件、对原提示做一次 teacher-forced 前向传播，取平均 token 对数似然为奖励。该信号确定性、免训练、免偏好标注、免流水线，直接复用 MLLM 预训练中最根本的视觉-语言扎根能力，从机制上规避了标量打分对评判校准与打分噪声的敏感性，以及 VQA 分解式验证的工程复杂度（消融中较标量打分在 GenEval 上高出 11.8 分）。

2. **语义谱（semantic spectrum）概念及其可解释性**。逐 token 的图像条件化似然构成图文对的"语义谱"，精细刻画视觉证据对各提示要求的支撑程度：计数错误集中拉低数量词（"Two"）、物体替换骤降对应名词（"guitar"），序列级均值仍保持可靠的组内排序。这一视角同时回答了"为什么简单平均就足够"的机理问题，并为奖励提供了 token 级可解释性。

3. **Self-SpectraReward：统一多模态模型的闭环自我提升**。策略自身的理解分支为自身生成分支打分，无需外部奖励模型与外部知识；由于共享 tokenizer、视觉编码器与预训练分布，奖励天然与策略的生成分布对齐。实验证明这种"奖励-策略对齐"使 BAGEL 的自奖励匹敌 4 倍规模（30B 级）、超越 30 倍规模（235B）的外部奖励模型——奖励质量并非仅由规模决定。

4. **对奖励模型规模的系统性反直觉发现**。跨四个家族、九个骨干（4B–235B）的受控对照表明：奖励 MLLM 的收益非单调（Qwen3-VL 8B→30B 全面提升、30B→235B 明显回落），且预训练版骨干（Gemma3-12B-Pretrain）一致优于指令微调版——提示"图像条件化描述能力"在中等规模即已饱和，指令微调反而使模型偏离这一基础能力；约 30B 的 MLLM 即可解锁 SpectraReward 的大部分收益。

5. **组相对 RL 下无需 PMI 归一化的理论分析与干净的实现细节**。证明 PMI 式修正 $\log p(x\mid y)-\log p(x)$ 中的纯文本似然被同组图像共享、在组相对优势计算中自动抵消，故无需额外归一化；辅以 EOS 剔除（终止信号污染平均似然，尤其伤害短提示）、默认不加描述前缀（仅影响绝对校准而不影响组内排序）等设计，保证了奖励信号的干净与稳定。

6. **大规模、多维度的实证验证与强基准超越**。覆盖两种生成器（SD3.5-M、BAGEL）、三种 RL 算法（FlowGRPO/AWM/DiffusionNFT）、九个奖励骨干与五个 OOD 基准（GenEval/GenEval2/TIIF/DPG/WISE），并与 HPSv3、UnifiedReward、VIEScore、DVReward 等奖励模型同台比较。在 BAGEL 上，SpectraReward/Self-SpectraReward 将 TIIF-Bench 整体短提示提升 +10.0、GenEval 提升 +4.3/+5.5，全面超越 AlphaGRPO；且 512 分辨率训练的增益可迁移到 1024 推理（Self-SR GenEval 89.8、WISE w/CoT 0.76）。

---

*注：本翻译基于 ar5iv（arXiv HTML / LaTeXML）转换的 Markdown 版本，已忽略站点导航残留与 LaTeX 转义符；参考文献列表从略，正文引用保留 (作者, 年份) 格式。原文贡献列表中"44B–235235B"系排版错误，应为"4B–235B"（译注已按正确数值翻译）；附录 C 中"TIIF-Long"一句与表 7 数值不符，已在附录要点第 4 条以译注标明。*


