# 《Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation》中文精读翻译

## 论文信息

- **英文标题**：Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation
- **中文标题**：Trust Your Critic：面向忠实图像编辑与生成的鲁棒奖励建模与强化学习
- **作者**：Xiangyu Zhao、Peiyuan Zhang、Junming Lin、Tianhao Liang、Yuchen Duan、Changyao Tian、Shengyuan Ding、Yuhang Zang、Junchi Yan、Xue Yang（含同等贡献与通讯作者标注）
- **机构**：上海交通大学（Shanghai Jiao Tong University）、武汉大学（Wuhan University）、北京邮电大学（BUPT）、香港中文大学（CUHK）、复旦大学（Fudan University）、上海人工智能实验室（Shanghai AI Laboratory）
- **arXiv 链接**：<https://arxiv.org/abs/2603.12247>
- **项目主页**：<https://firm-reward.github.io/>；代码：<https://github.com/VisionXLab/FIRM-Reward>；模型：<https://huggingface.co/collections/VisionXLab/firm-reward>
- **提出的框架**：FIRM（Faithful Image Reward Modeling，忠实图像奖励建模），包含数据流水线 FIRM-Edit/FIRM-Gen、数据集 FIRM-Edit-370K/FIRM-Gen-293K、奖励模型 FIRM-Edit-8B/FIRM-Gen-8B、基准 FIRM-Bench、奖励融合策略 CME/QMA，以及经 RL 微调的下游模型 FIRM-Qwen-Edit/FIRM-SD3.5

## 一句话核心总结

FIRM 通过"差异优先（difference-first）的编辑数据构造流水线 + 清单式（checklist）生成数据构造流水线 + 乘性耦合的 Base-and-Bonus 奖励融合（CME/QMA）"这一整套面向批评者（critic）的生态建设，解决了图像编辑与文生图强化学习中奖励模型幻觉频发、分数噪声大的核心瓶颈：仅 8B 规模的专门化奖励模型即可在与人类判断的对齐度上超越 GPT-5 等通用巨模型，并引导下游编辑/生成模型以极少量训练样本取得 SOTA 级性能提升。

---

## 摘要

强化学习（Reinforcement Learning, RL）已成为增强图像编辑与文生图（text-to-image, T2I）生成的颇具前景的范式。然而，当前在 RL 过程中充当批评者（critic）的奖励模型常常遭受幻觉（hallucination）困扰并给出带噪声的分数，从根本上误导优化过程。本文提出 **FIRM（Faithful Image Reward Modeling，忠实图像奖励建模）**——一个训练鲁棒奖励模型、为忠实图像生成与编辑提供准确可靠指导的综合框架。首先，本文设计了定制化的数据治理（data curation）流水线来构建高质量打分数据集：图像编辑同时从执行（execution）与一致性（consistency）两方面评估，而生成则主要通过指令遵循（instruction following）评估。借助这些流水线，本文收集了 FIRM-Edit-370K 与 FIRM-Gen-293K 两个数据集，并训练了准确反映上述标准的专门化奖励模型（FIRM-Edit-8B 与 FIRM-Gen-8B）。其次，本文引入 FIRM-Bench——一个专为编辑与生成批评者设计的综合基准。评测表明，与现有方法相比，本文模型实现了与人类判断的更优对齐。此外，为将这些批评者无缝集成进 RL 流水线，本文提出了一种新颖的"Base-and-Bonus"（基础加成）奖励策略，以平衡相互竞争的目标：编辑任务采用一致性调制执行（Consistency-Modulated Execution, CME），生成任务采用质量调制对齐（Quality-Modulated Alignment, QMA）。在该框架的赋能下，最终得到的 FIRM-Qwen-Edit 与 FIRM-SD3.5 模型取得了实质性的性能突破。大量实验表明，FIRM 缓解了幻觉问题，在忠实性与指令遵循方面对现有通用模型确立了新标准。所有数据集、模型与代码将全部公开。

---

## 1 引言

扩散模型（diffusion model）与自回归模型（autoregressive model）的快速进展已经彻底改变了文生图生成 (Crowson et al., 2022; Balaji et al., 2022; Saharia et al., 2022; Zhang et al., 2023) 与图像编辑 (Liu et al., 2025; Wu et al., 2025; Zhang et al., 2025) 两大领域。近来，强化学习（RL）已成为一种主流范式，其严重依赖奖励模型（或称"批评者"）来提供优化信号。在这一 RL 驱动的对齐过程中，生成或编辑图像的质量从根本上受限于批评者的准确性与可靠性。尽管 RL 已被广泛采用，当前流水线仍面临一个关键瓶颈：**批评者的不可靠性**。近期的多模态大语言模型（Multimodal Large Language Model, MLLM）(Bai et al., 2025; OpenAI, 2025; Google, 2025; Wang et al., 2025; Kimi Team, 2026) 虽已展现出令人印象深刻的通用能力，但当被用作细粒度图像编辑与生成任务的零样本（zero-shot）奖励模型时却屡屡受挫。这些模型天然存在严重的幻觉、物体忽略（object neglect）与精确空间推理缺失等问题，导致不合理且带噪声的奖励分数。

> **图 1**：不同方法的图像编辑结果对比。"w. FIRM-Edit-8B"表示在 RL 过程中采用 FIRM-Edit-8B 作为奖励模型。

为解决这些关键局限，本文提出 **FIRM（Faithful Image Reward Modeling）**——一个综合框架，旨在训练鲁棒的、任务专属的奖励模型，作为图像编辑与生成均可信赖的批评者。FIRM 引入定制化的数据构造流水线来合成高质量奖励数据。对于图像编辑，本文观察到一个关键的反直觉现象：虽然 MLLM 难以直接判断一张编辑后的图像是否在保持一致性的同时完美遵循了指令，它们却**极其擅长识别两张图像之间的差异**。因此，本文的编辑流水线采用"差异优先"（difference-first）方法：先利用一个 MLLM 为源图像与编辑后图像之间的确切视觉差异生成描述，随后将这份文本差异提供给一个 MLLM，可靠地推导出执行（execution）与一致性（consistency）两项最终分数。对于 T2I 生成，本文提出基于清单（checklist-based）的提示策略：先由一个 LLM 从用户提示中提取关键检查点，再将其附加到 MLLM 的打分提示中，从而显式引导 MLLM 验证细粒度细节，显著减少幻觉。通过这些流水线与既有开源数据集，本文构建了两个高质量奖励数据集：FIRM-Edit-370K 与 FIRM-Gen-293K。随后，本文用这些数据集训练奖励模型 FIRM-Edit-8B 与 FIRM-Gen-8B（均从 Qwen3-VL-8B-Instruct 模型初始化）。

> **图 2**：不同方法的 T2I 生成结果对比。"w. FIRM-Gen-8B"表示在 RL 过程中采用 FIRM-Gen-8B 作为奖励模型。

为严格验证本文的批评者，本文还构建了一个覆盖生成与编辑任务的人工标注基准 FIRM-Bench。通过从多样化的既有基准中获取提示、从各种模型中获取图像，同时严格控制真值（ground-truth）分数分布，本文证明所训练的奖励模型与人类偏好的对齐度显著优于现有开源 MLLM。

在这些可信赖批评者的基础上，本文进一步执行 RL 来优化生成模型。RL 中一个众所周知的挑战是：朴素地最大化多个常常相互竞争的奖励，往往引发优化崩溃（optimization collapse）或严重的奖励作弊（reward hacking）。为缓解此问题，本文广泛探索了"Base-and-Bonus"（基础加成）奖励加权策略，并识别出一种被称为一致性调制执行（Consistency-Modulated Execution, CME）的协同奖励表述。该策略成功平衡了编辑任务中的权衡并防止了奖励作弊。类似地，针对 T2I 生成，本文提出质量调制对齐（Quality-Modulated Alignment, QMA）来平衡指令遵循与图像质量。在鲁棒批评者与上述奖励表述的指导下，本文成功训练了 FIRM-Qwen-Edit 与 FIRM-SD3.5 模型。大量实验表明，本文模型在生成与编辑两种范式下均取得了实质性的性能提升，如图 1 与图 2 所示。

概括而言，本文的主要贡献有四：

1. 本文提出 FIRM 框架，由两条专门化的数据构造流水线（FIRM-Gen 与 FIRM-Edit）组成，用于训练鲁棒奖励模型。本文为图像编辑设计了新颖的"差异优先"（MLLM-to-LLM）流水线，为 T2I 生成设计了清单引导方法。这些流水线产出了高质量奖励数据集 FIRM-Edit-370K 与 FIRM-Gen-293K，及对应的奖励模型 FIRM-Edit-8B 与 FIRM-Gen-8B。
2. 本文构建了一个全面、完全人工标注的基准 FIRM-Bench，同时覆盖图像编辑与生成。评测确认本文奖励模型相比现有模型实现了与人类的更优对齐。
3. 本文在 RL 过程中提出新颖的"Base-and-Bonus"奖励融合策略，包括编辑任务的一致性调制执行（CME）与生成任务的质量调制对齐（QMA）。该表述有效缓解了奖励作弊，并确保在各种场景下的均衡表现。
4. 通过产出 FIRM-Qwen-Edit 与 FIRM-SD3.5 模型的大量 RL 实验，本文证明 FIRM 奖励模型与奖励策略在忠实图像编辑与精确图像生成两方面均带来实质性且一致的性能提升，验证了整体方法的有效性。

---

## 2 相关工作

### 2.1 图像编辑与生成的近期发展

文生图（T2I）生成经历了根本性的范式转变：从早期的对抗式方法 GAN (Goodfellow et al., 2014)、变分方法 VAE (Kingma et al., 2013)，到基于扩散的架构 (Ho et al., 2020; Nichol et al., 2021; Rombach et al., 2022)。随后，基于流（flow-based）的模型 (Esser et al., 2024; Labs, 2024) 作为一种高效替代方案兴起，在不牺牲合成质量的前提下提供加速采样。自回归模型 (Yu et al., 2022; Tian et al., 2024; Xie et al., 2024) 将图像合成视为序列建模任务，也获得了广泛关注。

在生成能力的基础上，图像编辑迅速演进。早期的扩散模型 (Hertz et al., 2022; Wallace et al., 2023) 严重依赖双提示（dual-prompt）形式。自 InstructPix2Pix (Brooks et al., 2023) 开创这一路线以来，近期工作 (Zhang et al., 2023; Sheynin et al., 2024) 通过策划大规模、高质量数据集对齐进行了大幅精炼。最近，流匹配（flow-matching）模型 (Labs et al., 2025) 正在提升训练与采样效率，而序列化自回归方法 (Yu et al., 2022; Tian et al., 2024) 则在从根本上改进组合推理。处于这一演进最前沿的是混合多模态架构，如 BAGEL (Deng et al., 2025) 与 Qwen-Image (Wu et al., 2025)。

### 2.2 图像编辑与生成中的强化学习

传统扩散模型 (Rombach et al., 2022; Podell et al., 2023) 主要通过最大似然估计（Maximum Likelihood Estimation, MLE）优化以匹配底层数据分布。近来，强化学习（RL）已成为关键范式。早期里程碑式方法如 DDPO (Black et al., 2023) 与 DPOK (Fan et al., 2023) 将扩散模型的迭代去噪过程形式化为多步马尔可夫决策过程（Markov Decision Process, MDP），从而可通过近端策略优化（Proximal Policy Optimization, PPO）进行直接策略优化。

近来，图像生成领域将思维链（Chain-of-Thought, CoT）推理与 RL 相结合。T2I-R1 (Jiang et al., 2025) 提出经由分组相对策略优化（Group Relative Policy Optimization, GRPO）优化的双层 CoT 推理框架。EDIT-R1 (Li et al., 2025) 则明确应对通用编辑奖励缺失的问题：利用一个 MLLM 作为统一的、免训练的奖励模型，借助其输出 logits 提供细粒度反馈。然而，在图像生成与编辑中，通用 MLLM 常常无法提供可靠奖励。EditScore (Luo et al., 2025) 引入了严格的基准套件与一族高保真奖励模型，显著超越开源 MLLM。EditReward (Wu et al., 2025) 使用人工标注数据集微调 MLLM 用于数据过滤。上述方法均缺乏对图像生成与编辑两个方向上可扩展数据治理（data curation）的综合研究。

---

## 3 方法

### 3.1 FIRM-Edit 流水线

在实践中，本文观察到一个显著现象：**模型作为"问题解决者"的表现通常优于作为"评估者"的表现**。具体而言，当被要求评判一张编辑后的图像时，模型常常无法捕捉那些在"解决"（例如描述）语境下本可识别的细粒度细节。这一差异导致最终评估分数出现整体性失配。

> **图 3**：FIRM 数据治理流水线总览。（上）FIRM-Edit 流水线遵循新颖的"差异优先"设计。（下）FIRM-Gen 流水线采用"先规划后评分"（plan-then-score）范式以显著提升打分准确性。

为解决此问题，本文引入 FIRM-Edit 流水线——一种"差异优先"方法，旨在为编辑评估生成大规模、高质量的 VQA 数据集（如图 3 所示）。首先，给定原始图像与编辑后图像对，本文提示一个最先进（state-of-the-art, SOTA）的 MLLM 执行**双层次差异分析（dual-level difference analysis）**：该模型被指示同时识别**明显修改**（obvious modifications）与**细节修改**（detailed modifications），并将其综合为一份统一的差异报告。所生成的对比描述能够有效捕捉编辑过程中的细微改动，从而得到全面的观察。随后，该描述与图像对及原始编辑指令一起，被输入一个充当评估者的鲁棒 MLLM。通过显式地以这份变更描述为条件，MLLM 评估者给出的最终评估与人类专家判断的对齐程度显著提升。

在奖励表述方面，受 RISEBench (Zhao et al., 2025) 启发，本文从两个主要维度评估编辑表现：**执行**（execution）与**一致性**（consistency），两项分数均在 1 到 5 之间。执行量化模型执行用户提示的准确性，而一致性衡量未被修改的物体与区域的保持程度。一次最优的编辑必须在满足全部编辑约束的同时维持原始语境的完整性。通过将评估过程解耦为这两个不同维度，本文的方法提供了细粒度的奖励信号，从而促成更优的模型表现。源数据来自公开可用的图像编辑数据集，包括 OpenGPT-4o-Image (Chen et al., 2025)、GPT-Image-Edit (Wang et al., 2025)、ShareGPT-4o-Image (Chen et al., 2025) 与 ImgEdit (Ye et al., 2025)。由于训练数据集中低分样本（如 1 分或 2 分）极其稀缺，本文有意改写了一部分指令来合成生成低质量匹配样本，以确保奖励分数的均衡分布。

### 3.2 FIRM-Gen 流水线

与图像编辑中观察到的挑战类似，直接提示一个 MLLM 为生成图像打一个整体分数往往导致粗粒度评估。在评估 T2I 生成、尤其是含大量需求的复杂指令时，当前开源模型难以同时权衡多个复杂约束（如实体数量、空间关系与风格属性），导致失准或不可解释的奖励信号。正如专业摄影评委先全面理解评审目标、识别关键视觉维度、逐一分析后再综合总分，本文主张：一个鲁棒的 T2I 奖励模型需要对评估准则的**显式结构化分解**。因此，本文引入 FIRM-Gen 流水线（如图 3 所示）——一种"先规划后评分"（plan-then-score）方法，旨在引出精确且可解释的判断。

FIRM-Gen 流水线将评估过程分解为两个不同阶段。**第一阶段（显式准则规划，Explicit Criteria Planning）**：本文使用一个强大的 LLM（Qwen3-32B (Bai et al., 2025)）充当"规划者"（planner）。给定原始文生图提示，规划者负责制定一份详细的、定制化的打分清单（checklist）。该计划将提示动态分解为细粒度的评估维度，例如主要主体/元素准确性、风格/构图对齐以及潜在的负向约束。**第二阶段（结构化分析评分，Structured Analytical Scoring）**：一个强大的 MLLM（Qwen3-VL-235B-A22B (Bai et al., 2025)）承担"评估者"（evaluator）角色。它接收生成图像、原始生成提示，以及第一阶段得出的定制化评分计划。通过让视觉分析显式地以该结构化计划为条件，MLLM 被迫对每个预定义维度进行逐步检查，然后再将其聚合为最终分数。这种"分而解之"（decouple-and-conquer）策略有效缓解了 MLLM 中的"注意力稀释"（attention dilution）问题，使奖励建模更鲁棒、准确、可解释，并贴近人类认知过程。

为确保训练出的奖励模型在多样场景下良好泛化，本文构建了高度全面的源数据集。生成提示采样自主流的高质量数据集，包括 OpenGPT-4o-Image (Chen et al., 2025)、ShareGPT-4o-Image (Chen et al., 2025) 与 BLIP3o-60k (Chen et al., 2025)，覆盖了用户意图与复杂度的广泛谱系。此外，为确保训练数据覆盖广泛的图像质量与生成风格，本文使用一组多样化模型从这些提示生成图像。该模型池纳入了架构与容量各异的具体模型：Ovis-image (Wang et al., 2025)、Z-image-turbo (Team, 2025)、Flux.1-dev (Labs, 2024)、SDXL (Podell et al., 2023) 与 SD1.5 (Rombach et al., 2022)。这种刻意的多样性防止了奖励模型过拟合于单一生成器的伪影（artifacts），从而为 T2I 生成确立普适且可靠的奖励信号。

### 3.3 FIRM-Bench 的构建

为严格验证本文的奖励模型，本文引入 FIRM-Bench——一个包含 807 个精心筛选样本的综合基准。该基准划分为两个子集：FIRM-Bench-Edit（301 条用于执行评估、256 条用于一致性评估）与 FIRM-Bench-Gen（250 条用于指令遵循评估）。

**数据收集与标注**。为规避数据污染，本文将基准与既有训练数据集严格隔离。从标准基准中采样提示，并从当前流行模型的开源输出中收集相应结果图像。随后，雇佣人类专家按照任务为每个"提示-图像"对打分。

**数据集均衡与评测协议**。为确保评测稳健且无偏，本文仔细控制采样过程，使人类标注真值分数（范围 1 到 5）在所有指标上保持均匀分布。此外，本文依据指令复杂度将 FIRM-Bench-Gen 划分为"简单"（easy）与"困难"（hard）两个子集，从而支持对模型能力的细粒度分析。最终，一个给定奖励模型的性能以模型预测分数与人类标注真值之间的**平均绝对误差**（Mean Absolute Error, MAE）来量化。

### 3.4 RL 中的奖励设计

**在线 RL 算法**。本文使用 DiffusionNFT (Zheng et al., 2025) 优化模型——这是一种定义在**前向**（forward）扩散过程上的在线 RL 范式，经由流匹配（flow matching）实现。记前向加噪过程为：

$$
x_{t}=\alpha_{t}x_{0}+\sigma_{t}\epsilon,\quad\epsilon\sim\mathcal{N}(0,I) \tag{1}
$$

速度参数化（velocity parameterization）由流匹配训练：

$$
\mathbb{E}_{c,\,x_{0}\sim\pi_{\text{old}}(\cdot|c),\,t,\,\epsilon}\big[w(t)\lVert v_{\theta}(x_{t},c,t)-v\rVert_{2}^{2}\big], \tag{2}
$$

其中目标速度由噪声调度的导数决定：

$$
v=\dot{\alpha}_{t}x_{0}+\dot{\sigma}_{t}\epsilon \tag{3}
$$

DiffusionNFT 在速度场中定义一个强化引导的目标策略：

$$
v^{\ast}(x_{t},c,t)=v_{\text{old}}(x_{t},c,t)+\frac{1}{\beta}\Delta(x_{t},c,t), \tag{4}
$$

其中 $\Delta(x_{t},c,t)$ 是由 $\pi_{\text{old}}$ 的正负样本划分（positive and negative splits）之间的对比导出的**强化引导**（reinforcement guidance）方向。

给定每个采样得到干净图像 $x_{0}$ 的**最优性概率**（optimism probability）$r\in[0,1]$，DiffusionNFT 优化负感知（negative-aware）目标：

$$
\mathcal{L}(\theta)=\mathbb{E}_{c,\,x_{0}\sim\pi_{\text{old}}(\cdot|c),\,t,\,\epsilon}\Big[r\lVert v_{\theta}^{+}(x_{t},c,t)-v\rVert_{2}^{2}+(1-r)\lVert v_{\theta}^{-}(x_{t},c,t)-v\rVert_{2}^{2}\Big] \tag{5}
$$

其隐式正/负策略为：

$$
v_{\theta}^{+}(x_{t},c,t)=(1-\beta)v_{\text{old}}(x_{t},c,t)+\beta v_{\theta}(x_{t},c,t) \tag{6}
$$

$$
v_{\theta}^{-}(x_{t},c,t)=(1+\beta)v_{\text{old}}(x_{t},c,t)-\beta v_{\theta}(x_{t},c,t) \tag{7}
$$

在数据与模型容量无限的条件下，上述目标的最优解满足：

$$
v_{\theta}^{\ast}(x_{t},c,t)=v_{\text{old}}(x_{t},c,t)+\frac{2}{\beta}\Delta(x_{t},c,t) \tag{8}
$$

**图像编辑的奖励表述**。在 RL 框架内，本文设计了一个由两个互补信号组成的奖励方法：**执行**与**一致性**，各自以 1–5 分制打分并在训练期间归一化到 $[0,1]$。然而，一个关键挑战在于这两个信号之间的合理功劳分配（credit assignment）。本文的初始设计采用简单线性组合：

$$
R=w_{1}\cdot\text{Consistency}+w_{2}\cdot\text{Execution},
$$

其中 $w_{1}=w_{2}=0.5$。该表述引发了严重的奖励作弊。实践中，模型发现最大化一致性远比提升执行容易。结果，它收敛到一种退化的策略：输出与输入几乎相同的图像，获得高一致性分数却无法执行有意义的编辑。这种行为严重阻碍了 RL 下编辑能力的提升。

本文首先尝试通过重新平衡权重来缓解该问题（例如将执行系数提高到 0.6）。虽然这一调整略微缓解了问题，但并未从根本上阻止这种捷径策略。为更有原则地解决此问题，本文采用了一种更合理的奖励函数——**一致性调制执行**（Consistency-Modulated Execution, CME）：

$$
R_{CME}=\text{Execution}\cdot\left(w_{1}+w_{2}\cdot\text{Consistency}\right),
$$

其中 $w_{1}=0.6$、$w_{2}=0.4$。该表述将执行设为获得高奖励的**必要条件**：若执行分数低，则无论一致性多高，整体奖励都保持受抑。与此同时，一致性仍作为塑形信号（shaping signal），在有意义编辑被执行之后精化结构保真度。经验上，该设计带来显著更优的功劳分配，有效防止奖励作弊，并大幅提升编辑性能。

**图像生成的奖励表述**。在本文最初的 RL 流水线中，仅使用指令遵循作为唯一奖励信号。然而，这一朴素设置暴露出一类有趣的奖励作弊。对于综合性强、高度详细的提示，策略表现得符合预期；但对于仅含裸物体类别的短提示，模型发现了平凡解（trivial solution）：倾向于合成所请求物体的一个**黑色剪影**——完美满足文本条件却完全缺乏内在视觉保真度。为解决此问题，本文主张生成质量必须充当约束。借鉴编辑奖励设计的思路，本文提出**质量调制对齐**（Quality-Modulated Alignment, QMA）：

$$
R_{QMA}=\text{InsFollowing}\cdot\left(w_{1}+w_{2}\cdot\text{Quality}\right),
$$

其中 $w_{1}=0.4$、$w_{2}=0.6$。该奖励塑形策略在指令遵循分数达到高水平后更加重视图像质量，有效缓解了上述作弊行为。

---

## 4 实验结果

### 4.1 实验设置

在奖励模型的监督微调（Supervised Fine-Tuning, SFT）阶段，本文从 Qwen3-VL-8B-Instruct 模型初始化 FIRM-Edit-8B 与 FIRM-Gen-8B，分别在 FIRM-Edit-370K 与 FIRM-Gen-293K 数据集上训练。SFT 过程在 8×H200 GPU 上基于 LLaMA-Factory (Zheng et al., 2024) 代码库完成。

为进一步验证 FIRM-Edit-8B 与 FIRM-Gen-8B 的有效性，本文将它们作为奖励模型集成进图像编辑与生成任务基座模型的强化学习（RL）过程。在 RL 阶段，图像编辑采用 Edit-R1 (Li et al., 2025) 框架，图像生成采用 Diffusion-NFT (Zheng et al., 2025) 框架。每个 RL 阶段均在 16×H200 GPU 上扩展。超参数方面，每次 rollout 的采样数设为 $N=16$，总批量大小为 16（编辑）与 48（生成）。编辑与生成 RL 过程的训练提示均来自 ShareGPT-4o-Image (Chen et al., 2025) 数据集。RL 阶段中，编辑模型训练 150 步、生成模型训练 600 步。本文将 RL 后的性能与当前最先进（SOTA）方法进行对比。此外，为兼顾效率与能力，本文还以 Qwen3-VL-8B 与 Qwen3-VL-32B 替换本文的奖励模型进行消融研究。

> **图 4**：FIRM-Bench 的示例，显示本文的奖励模型比 Qwen3-VL-8B 与 Qwen3-VL-32B 更贴合人类判断。

**表 1**：FIRM-Bench-Edit 上的结果（MAE，越低越好）。

| 模型 | Exec. | Cons. | Overall |
| --- | --- | --- | --- |
| GPT-4o | 0.73 | 1.16 | 0.93 |
| GPT-4.1 | 0.74 | 1.04 | 0.88 |
| GPT-5 | 0.62 | 0.73 | 0.67 |
| Gemini-3-pro | 0.54 | 0.57 | 0.55 |
| InternVL3.5-8B | 0.90 | 1.13 | 1.00 |
| InternVL3.5-38B | 0.85 | 1.22 | 1.02 |
| InternVL3.5-241B-A28 | 0.69 | 1.05 | 0.86 |
| Qwen3-VL-8B | 0.66 | 1.12 | 0.87 |
| Qwen3-VL-32B | 0.69 | 1.14 | 0.90 |
| Qwen3-VL-235B-A22 | 0.72 | 0.91 | 0.81 |
| FIRM-Edit-8B（本文） | **0.53** | **0.73** | **0.62** |

**表 2**：FIRM-Bench-Gen 上的结果（MAE，越低越好）。

| 模型 | Easy | Hard | Overall |
| --- | --- | --- | --- |
| GPT-4o | 0.63 | 0.66 | 0.65 |
| GPT-4.1 | 0.58 | 0.69 | 0.64 |
| GPT-5 | 0.50 | 0.53 | 0.52 |
| Gemini-3-pro | **0.38** | **0.42** | **0.40** |
| InternVL3.5-8B | 0.61 | 0.55 | 0.58 |
| InternVL3.5-38B | 0.79 | 0.58 | 0.68 |
| InternVL3.5-241B-A28 | 0.60 | 0.61 | 0.60 |
| Qwen3-VL-8B | 0.63 | 0.63 | 0.63 |
| Qwen3-VL-32B | 0.51 | 0.56 | 0.54 |
| Qwen3-VL-235B-A22 | 0.60 | 0.52 | 0.56 |
| FIRM-Gen-8B（本文） | 0.45 | 0.57 | 0.51 |

### 4.2 FIRM-Bench 上的评测结果

本文在 FIRM-Bench 上对若干当前 SOTA 模型进行评测实验，包括专有模型（GPT 系列与 Gemini）和开源模型（Qwen3-VL 与 InternVL3.5 系列）。结果见表 1 与表 2，更多案例见图 4。

在 FIRM-Bench-Edit 上，专有模型与开源模型之间的性能差距进一步拉大。Gemini-3-Pro 取得第二低的执行 MAE 0.54 与最低的一致性 MAE 0.57，整体 MAE 为 0.55。相比之下，最大的开源模型 Qwen3-VL-235B 在执行 MAE 上挣扎于 0.72、一致性 MAE 为 0.91（整体 0.81）。这表明当前开源模型的图像编辑能力仍有巨大提升空间。然而，在 FIRM-Edit-370K 上训练后，本文的 FIRM-Edit-8B 取得了最低的执行 MAE 0.53 与一致性 MAE 0.73，整体 MAE 达到 0.62，成功超越 GPT-5 与所有其他开源基线。

在 FIRM-Bench-Gen 上，Gemini-3-Pro 取得最低的 MAE 0.40，GPT-5 取得 0.52，而最佳开源模型 Qwen3-VL-32B 仅达到 0.54。值得注意的是，尽管参数量少得多，本文的 FIRM-Gen-8B 取得了高度竞争力的 MAE 0.51，同时超越 GPT-5 与所有被评测的开源模型。FIRM-Bench 上的这些结果直接证明了本文所提框架的有效性。

**表 3**：GEdit-Bench 与 ImgEdit 上的性能对比。在 RL 过程中以 FIRM-Edit-8B 为引导，所得 FIRM-Qwen-Edit 大幅超越基座模型以及以 Qwen3-VL 模型训练的对照版本。

| 模型 | G_SC | G_PQ | G_Overall（GEdit-Bench） | ImgEdit Overall |
| --- | --- | --- | --- | --- |
| InstructPix2Pix (Brooks et al., 2023) | 3.58 | 5.49 | 3.68 | 1.88 |
| AnyEdit (Yu et al., 2025) | 3.18 | 5.82 | 3.21 | 2.45 |
| Step1X-Edit (Liu et al., 2025) | 7.66 | 7.35 | 6.97 | 3.06 |
| FLUX.1-Kontext[Dev] (Labs et al., 2025) | 6.52 | 7.38 | 6.00 | 3.71 |
| OmniGen2 (Wu et al., 2025) | 7.16 | 6.77 | 6.41 | 3.44 |
| FLUX.1-Kontext[Pro] (Labs et al., 2025) | 7.02 | 7.60 | 6.56 | 4.00 |
| UniWorld-FLUX.1-Kontext (Li et al., 2025) | 7.28 | 7.49 | 6.74 | 4.02 |
| GPT-Image (Singh et al., 2025) | 7.85 | 7.62 | 7.53 | 4.20 |
| UniWorld-Qwen-Image-Edit (Li et al., 2025) | **8.36** | **7.87** | 7.76 | **4.48** |
| Qwen-Image-Edit-2509 (Wu et al., 2025) | 8.15 | 7.86 | 7.54 | 4.35 |
| - RL with Qwen3VL-8B | 8.04 | **8.22** | 7.69(+0.15) | 4.36(+0.01) |
| - RL with Qwen3VL-32B | 7.94 | 8.16 | 7.65(+0.11) | 4.28(-0.07) |
| - FIRM-Qwen-Edit（本文） | 8.25 | 8.20 | **7.84(+0.30)** | 4.42(+0.07) |

### 4.3 经 RL 过程的图像编辑性能

在图像编辑任务上，本文以 Qwen-Image-Edit-2509 (Wu et al., 2025) 为实验基座模型，并在表 3 中汇报 GEditBench (Liu et al., 2025) 与 ImgEdit (Ye et al., 2025) 两个基准上的 RL 后评测结果。如表所示，由 FIRM-Edit-8B 引导的策略优化带来了卓越的性能增益。仅通过单一一个 RL 阶段，本文的 FIRM-Qwen-Edit 便在 GEditBench 上确立了 7.84 的新 SOTA 分数，同时在 ImgEdit 上取得第二高的 4.42 分。值得注意的是，本文模型仅需 $150\times 16=2,400$ 个训练样本即可取得与 UniWorld-Qwen-Image-Edit（使用多达 27K 样本）相当的性能。这一鲜明对比进一步凸显了本文方法的高效性。

尤其值得注意的是，与采用通用型 Qwen3-VL-8B 和 Qwen3-VL-32B 作为奖励模型相比，FIRM-Edit-8B 在两个基准上都带来了稳健的性能提升（GEditBench +0.30、ImgEdit +0.07），这可归因于其高度准确的奖励信号。相比之下，用 Qwen3-VL-8B 做 RL 仅带来微小改进（GEditBench +0.15、ImgEdit +0.01）。此外，Qwen3-VL-32B 虽在 GEditBench 上带来 +0.11 的小幅增益，却在 ImgEdit 上意外造成 -0.07 的性能下降。这一现象有力地表明：**单纯扩大通用 VLM 的规模并不能保证更好的奖励建模**；相反，它凸显了 FIRM-Edit-8B 所提供的精确且任务对齐的奖励信号的不可或缺性——这类信号对防止负优化、引导生成器实现更优编辑忠实性至关重要。本文在图 5 中展示了 RL 奖励曲线的对比（译注：原文此处写作 figure 7 且将奖励模型写作 FIRM-Gen-8B，结合上下文编辑场景及图 5/6 的图题，应为 figure 5 与 FIRM-Edit-8B）。FIRM-Edit-8B 给出的奖励持续低于 Qwen3-VL 系列。这表明在评估图像编辑场景时，通用型 Qwen3-VL 模型经常忽视细微变化，导致人为偏高的分数。

**表 4**：GenEval、DPG、TIIF 与 UniGenBench++ 上的性能对比。借助 FIRM-Gen-8B 进行 RL，本文的 FIRM-SD3.5 取得实质性性能增益，超越基座模型以及以 Qwen3-VL 模型训练的对照版本。

| 模型 | GenEval | DPG-Bench | TIIF | UniGenBench++ Short | UniGenBench++ Long |
| --- | --- | --- | --- | --- | --- |
| SD-XL (Podell et al., 2023) | 0.55 | 74.65 | 54.96 | 40.22 | 41.48 |
| Show-o (Xie et al., 2024) | 0.53 | - | 59.72 | - | - |
| EMU3-Gen (Wang et al., 2024) | 0.54 | 80.60 | - | 45.42 | 50.59 |
| JanusFlow (Ma et al., 2025) | 0.63 | 79.68 | - | 47.10 | 54.80 |
| FLUX.1-Dev (Labs, 2024) | 0.66 | 83.84 | - | 60.97 | 69.42 |
| DALLE-3 (Hurst et al., 2024) | 0.67 | 83.50 | 74.96 | 68.85 | 70.82 |
| BLIP3o-4B (Chen et al., 2025) | 0.81 | 79.36 | - | 59.57 | 61.01 |
| Janus-Pro-7B (Chen et al., 2025) | 0.80 | 84.19 | 66.50 | 61.36 | 71.11 |
| Show-o2 (Xie et al., 2025) | 0.76 | 86.14 | - | 61.90 | 70.33 |
| OmniGen2 (Wu et al., 2025) | 0.80 | 83.57 | - | 63.09 | 71.39 |
| BAGEL (Deng et al., 2025) | **0.82** | 85.07 | 71.50 | 59.91 | 71.26 |
| SD3.5-Medium (Rombach et al., 2022) | 0.52 | 84.08 | 70.17 | 60.71 | 64.67 |
| - RL with Qwen3VL-8B | 0.76(+0.24) | 86.87(+2.79) | 75.99(+5.82) | 67.17(+6.46) | 74.50(+9.83) |
| - RL with Qwen3VL-32B | 0.70(+0.18) | 85.94(+1.86) | 76.43(+6.26) | 67.79(+7.08) | 73.56(+8.89) |
| - FIRM-SD3.5（本文） | 0.77(+0.25) | **87.16(+3.08)** | **77.12(+6.95)** | **69.56(+8.85)** | **76.22(+11.55)** |

### 4.4 经 RL 过程的图像生成性能

为评估生成能力，本文采用 SD3.5-Medium (Rombach et al., 2022) 作为基座模型，并在 GenEval (Ghosh et al., 2023)、DPGBench (Hu et al., 2024)、TIIF (Wei et al., 2025)（test-mini-short）与 UniGenBench++ (Wang et al., 2025) 基准上汇报 RL 后评测结果，如表 4 所示。可以看到，由 FIRM-Gen-8B 引导的 FIRM-SD3.5 在多样且具挑战性的设置下产生了深远的性能增益。最终的 FIRM-SD3.5 模型在 GenEval 上取得 0.77、DPGBench 上取得 87.16、TIIF 上取得 77.12、UniGenBench-Short/Long 上取得 69.56/76.22 的高度竞争力分数。值得注意的是，这些结果超越了 BAGEL 与 OmniGen2 等资源雄厚的模型——后者受益于显著更大的参数规模与训练数据。

此外，在与充当替代奖励模型的 Qwen3-VL-8B 和 Qwen3-VL-32B 的直接对比中，FIRM-Gen-8B 始终驱动更优的生成结果，有力验证了本文专门化奖励表述的有效性。在以短小、无歧义提示为主的 GenEval 上，FIRM-SD3.5 相对 Qwen3-VL-8B 基线仅有边际优势（+0.25 对 +0.24）。然而，随着提示长度与组合复杂度的攀升，FIRM-Gen-8B 的优势变得极为显著：本文模型在 DPGBench（本文 +3.08 对 +2.79）、TIIF（本文 +6.95 对 +5.82）、UniGenBench-Short/Long（本文 +8.85/11.55 对 +6.46/9.83）上确立了大幅领先。这种稳健的扩展趋势清楚地表明，本文的奖励模型尤其擅长引导复杂视觉场景的生成。图像生成的 RL 奖励曲线对比见图 6（译注：原文此处引用 figure 7，结合上下文应为展示生成奖励曲线的 figure 6）。有趣的是，与编辑相反，FIRM-Gen-8B 给出的奖励分数持续**高于** Qwen3-VL 基线。这一现象有力地表明：在复杂生成场景中，通用模型容易出现评估准则纠缠与严重幻觉，二者共同导致无正当理由的低分。

**表 5**：不同奖励表述的性能对比。本文提出的 CME 通过"Base-and-Bonus"机制有效平衡竞争性奖励，取得最佳整体表现。

| 方法 | 表述 | 奖励模型 | G_SC | G_PQ | G_Overall | ImgEdit Overall |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | - | - | 8.15 | 7.86 | 7.54 | 4.35 |
| Edit-R1 | Non-CoT Logits | Qwen2.5VL-32B | 4.08 | 5.05 | 4.06 | 2.75 |
| Weighted Score | 0.5 * Exe. + 0.5 * Cons. | FIRM-Edit-8B | 0.94 | 8.71 | 1.06 | 2.17 |
| Weighted Score | 0.6 * Exe. + 0.4 * Cons. | FIRM-Edit-8B | 6.78 | 8.25 | 6.51 | 3.73 |
| CME（本文） | Exe. * (0.6 + 0.4 * Cons.) | FIRM-Edit-8B | **8.25** | **8.20** | **7.84** | **4.42** |

> **图 5**：不同奖励模型的编辑 RL 奖励曲线。

> **图 6**：不同奖励模型的生成 RL 奖励曲线。

> **图 7**：不同奖励计算指标消融的 RL 奖励曲线。

### 4.5 奖励表述消融

为验证本文一致性调制执行（CME）奖励指标的有效性，本文进行了消融研究，对比若干替代奖励表述。下游评测结果及对应的 RL 奖励轨迹见表 5 与图 7。可以观察到，除本文的 CME 外，所有其他基线指标都遭受了各种形式的性能退化。此外，奖励曲线表明，唯有 CME 在整个优化过程中实现了奖励信号一致且稳健的增长。

对于 Edit-R1，使用 Non-CoT logits 提供的是带噪声且混乱的信号，无法与人类偏好对齐。此外，依赖简单的分数加权和会导致严重的奖励作弊。具体而言，模型可以通过让图像完全不变来获得相对较高的总奖励；虽然指令遵循分数下降，但它通过确保完美的外观一致性分数来利用该指标。这种有缺陷的奖励结构激励编辑模型变得"懒惰"，选择直接输出原始输入图像，最终导致糟糕的编辑表现。而本文的 CME 指标有效缓解了这一问题：它持续激励模型执行给定指令的同时保持任务无关区域，从而带来显著更优的性能。

---

## 5 结论

本文提出了 FIRM——一个面向忠实图像编辑与生成的综合框架。本文的贡献覆盖整个强化学习生态，包括定制化的数据治理流水线、高质量奖励数据集、严格的评测基准、鲁棒的奖励模型以及新颖的奖励表述。通过构建这一端到端流水线，本文成功利用奖励模型引导 RL 过程，在下游生成任务中取得了实质性的性能提升。最终，本文的研究验证了准确批评者在 RL 过程中不可或缺的作用，希望能启发未来关于生成模型奖励引导对齐的研究。

---

## 附录要点（压缩概括）

1. **附录 A：FIRM 数据集与基准的统计分析（表 6）**。FIRM 数据集各分数级样本量如下（数据集列：Edit-Exec / Edit-Cons / Gen-Ins；基准列同序）：5 分——49,995 / 50,823 / 116,333（基准 105 / 73 / 67）；4 分——16,004 / 45,720 / 41,833（57 / 83 / 73）；3 分——52,661 / 30,999 / 56,157（108 / 82 / 50）；2 分——21,695 / 38,486 / 59,010（21 / 14 / 38）；1 分——49,819 / 14,053 / 19,459（10 / 4 / 22）；总计——190,174 / 180,102 / 292,792（基准 301 / 256 / 250）。可见编辑执行与一致性两个维度的样本合并约 37 万条（对应 FIRM-Edit-370K），生成维度约 29.3 万条（对应 FIRM-Gen-293K）；基准侧各指标真值分数接近均匀分布，与 3.3 节的均衡设计一致。

2. **附录 B：奖励函数作弊分析**。初始奖励 $R=0.5\,\text{Consistency}+0.5\,\text{Execution}$ 导致严重奖励作弊：模型过度优化一致性而忽视执行，RL 训练中倾向于复现视觉上接近源图像的输出，一致性得分高但任务执行差。将线性权重调整为 $R=0.4\,\text{Consistency}+0.6\,\text{Execution}$ 后执行表现部分改善，但模型转而弱化与源图像的一致性——优化固定加权和仍会在两个目标间诱发不良权衡（图 8 展示了原始奖励设计下一致性与执行的奖励曲线）。最终采用乘性耦合的 CME：$R=\text{Execution}\times(0.6+0.4\,\text{Consistency})$，同时鼓励强执行与源图像对齐的保持。

3. **附录 C：FIRM-Edit 流水线各阶段提示词**。包括：①"明显编辑差异"提示——要求模型只描述两图间发生了变化的主体及其在两图中的各自特征，此阶段不做直接比较；②"细节编辑差异"提示——要求仔细比较全部差异，涵盖物体数量、形状与形态、纹理与表面、位置与布局等；③"编辑执行评估"提示——输入两图、差异描述与编辑指令，按 5（完美执行）到 1（无执行）的五级量表打分，并明确规定只评估所请求的变更是否出现与正确实现，忽略额外/未请求的修改、渲染质量、真实感与无关一致性问题，以 JSON 格式输出简短推理与分数；④"编辑一致性评估"提示——只评估"是否出现了指令未提及的变更"，给出显著变更/细微变更的定义与对应 5–1 分级标准，并明确遮挡效应（如移除物体后显露的背景）、画质变化（清晰度、噪声、色温、光照、阴影、饱和度等）不计为不一致；⑤"推断编辑执行/一致性判断"提示——推理时不提供差异描述、仅凭图像对与指令直接评估，量表与前者一致。

4. **附录 D：FIRM-Gen 流水线各阶段提示词**。包括：①"分析计划生成"提示——要求以视觉 QA 分析专家身份将文生图提示深度分解为可验证的视觉标准，按"主要主体/元素、属性/动作、风格/构图、负向约束（可选）"的模板输出 Markdown 计划；②"指令遵循评分"提示——要求先完整审视图像，再对照分析计划逐步评估每一项，按 5（全部要求、细节、风格与负向约束均正确）到 1（图像与提示无关）量表打分，以 `\boxed{}` 形式输出最终评级；③"推断指令遵循判断"提示——推理时仅凭原始提示自行识别显式与隐式要求（主体、属性、动作、关系、风格、构图、负向约束）并逐步评估打分。

---

## 创新点提炼

1. **"差异优先"（difference-first）的编辑奖励数据构造流水线**。核心洞察是一个反直觉现象：MLLM 作为"评估者"时难以综合判断编辑质量，但作为"问题解决者"时擅长识别两图差异。FIRM-Edit 先让 SOTA MLLM 对源图/编辑图做双层次（明显+细节）差异分析并综合成统一差异报告，再让评估 MLLM 基于该文本差异推导执行与一致性分数，从根本上规避了直接评判时的细粒度细节丢失，产出了 37 万条规模的 FIRM-Edit-370K 数据集（并通过对低分样本稀缺的指令改写合成来均衡分数分布）。

2. **"先规划后评分"（plan-then-score）的生成奖励数据构造流水线**。针对复杂多约束提示下 MLLM"注意力稀释"导致的评估失准，FIRM-Gen 将评估解耦为两阶段：Qwen3-32B 充当规划者从用户提示生成定制化评分清单（主体/元素、属性/动作、风格/构图、负向约束），Qwen3-VL-235B-A22B 充当评估者对照清单逐步检查后聚合总分。该显式结构化分解模拟了人类评委的认知过程，使奖励更准确、可解释，且用五种异构生成器（Ovis-image、Z-image-turbo、Flux.1-dev、SDXL、SD1.5）合成图像以防止奖励模型过拟合单一生成器伪影。

3. **8B 专门化奖励模型全面超越通用巨模型 + 全人工标注的 FIRM-Bench**。从 Qwen3-VL-8B-Instruct 出发、分别在 370K/293K 数据上 SFT 得到的 FIRM-Edit-8B 与 FIRM-Gen-8B，在 807 样本、真值分数严格均匀分布、与训练数据完全隔离的 FIRM-Bench 上取得编辑侧整体 MAE 0.62（超过 GPT-5 的 0.67 与全部开源基线）与生成侧 MAE 0.51（超过 GPT-5 的 0.52 与全部开源模型，仅次于 Gemini-3-Pro 的 0.40）——以不足 1/30 的参数量证明了"专门化数据 > 通用规模"的奖励建模路线。

4. **Base-and-Bonus 乘性奖励融合（CME/QMA）从机制上消灭奖励作弊**。不同于线性加权和（$0.5\text{Cons}+0.5\text{Exec}$ 诱发"复制输入图像"的退化解，GEditBench 崩溃至 1.06；调整权重至 0.6/0.4 仍只能部分缓解），CME 将执行作为乘性必要条件、一致性作为 $(w_1+w_2\cdot\text{Cons})$ 的调制加成项，使"高一致性"只有在"高执行"基础上才能兑换为奖励；生成侧对偶的 QMA（$\text{InsFollowing}\cdot(0.4+0.6\cdot\text{Quality})$）同样消除了短提示下合成"黑色剪影"的平凡解。消融表中 CME 以 7.84/4.42 显著领先所有替代表述。

5. **以极小样本量实现 SOTA 编辑性能的高效 RL**。在 FIRM-Edit-8B 引导下，Qwen-Image-Edit-2509 仅经 150 步 ×16 rollout（共 2,400 个训练样本）的 RL 即在 GEditBench 上达到 7.84 的新 SOTA、ImgEdit 上 4.42 的第二高分，媲美使用 27K 样本的 UniWorld-Qwen-Image-Edit；而用通用 Qwen3-VL-8B/32B 做奖励仅带来 +0.15/+0.11 的微弱增益，Qwen3-VL-32B 甚至在 ImgEdit 上造成 -0.07 的负优化——直接证明通用 VLM 缩放不保证奖励质量，且奖励信号精度决定 RL 增益方向。

6. **生成侧复杂提示的规模化增益与奖励分数的"方向性证据"**。FIRM-SD3.5（SD3.5-Medium + DiffusionNFT + FIRM-Gen-8B）在 GenEval/DPGBench/TIIF/UniGenBench++ 上全面超越 BAGEL、OmniGen2 等大模型，且提示越复杂领先越大（UniGenBench-Long +11.55）；奖励曲线还揭示一个方向性现象：编辑场景中 FIRM-Edit-8B 打分持续低于 Qwen3-VL（通用模型忽视细微变化、分数虚高），生成场景中 FIRM-Gen-8B 打分持续高于 Qwen3-VL（通用模型评估准则纠缠、幻觉性压分）——从两个方向印证了通用 MLLM 作为奖励模型的系统性失准。

---

*注：本翻译基于 arXiv HTML（LaTeXML）转换的 Markdown 版本，已忽略站点导航残留与 LaTeX 转义符；参考文献列表从略，正文引用保留 (作者, 年份) 格式。原文 4.3/4.4 节两处奖励曲线的图号引用疑有笔误（均写作 figure 7，且 4.3 节将编辑奖励模型误写为 FIRM-Gen-8B），已在相应位置以译注标明。*



