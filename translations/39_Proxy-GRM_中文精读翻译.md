# 《Rationale Matters: Learning Transferable Rubrics via Proxy-Guided Critique for VLM Reward Models》中文精读翻译

## 论文信息

- **英文标题**：Rationale Matters: Learning Transferable Rubrics via Proxy-Guided Critique for VLM Reward Models
- **中文标题**：理由很重要：经由代理引导批评为 VLM 奖励模型学习可迁移量规
- **作者**：Weijie Qiu、Dai Guan、Junxin Wang、Zhihang Li、Yongbo Gai、Mengyu Zhou、Erchao Zhao、Xiaoxi Jiang、Guanjun Jiang
- **机构**：阿里巴巴 Qwen 大模型应用团队（Qwen Large Model Application Team, Alibaba）；其中 Junxin Wang 来自中国科学院自动化研究所（Institute of Automation, Chinese Academy of Sciences）
- **arXiv 链接**：<https://arxiv.org/abs/2603.16600>
- **提出的框架**：Proxy-GRM

## 一句话核心总结

Proxy-GRM 通过训练一个专门"消费"量规（rubric）而非生成量规的轻量代理评测器（Proxy-SFT / Proxy-RL），以其"仅凭量规能否复现正确偏好判断"的预测准确率作为量规质量奖励，注入 GRPO 强化学习闭环，使多模态生成式奖励模型学会生成可迁移、可独立验证的评测量规——仅用约 5 万训练样本（约为可比方法的 1/4）即在 VL-RewardBench、Multimodal Reward Bench、MM-RLHF-Reward Bench 三大基准上取得 SOTA，且学得的量规无需再训练即可迁移给未见过的评测器模型。

---

## 摘要

面向视觉-语言模型（Vision-Language Model, VLM）的生成式奖励模型（Generative Reward Model, GRM）通常通过一条三阶段流水线评估输出：量规（rubric，即结构化评估准则）生成、逐准则打分、最终裁决。然而，中间生成的量规极少被直接优化。既有工作往往要么把量规视为附带产物，要么依赖昂贵的 LLM-as-a-Judge 检查——后者既不提供可微分信号，训练期的指导也十分有限。本文提出 **Proxy-GRM**，将代理引导的量规验证引入强化学习（Reinforcement Learning, RL），以显式提升量规质量。具体而言，本文训练轻量级代理智能体（Proxy-SFT 与 Proxy-RL），其输入为候选量规连同原始查询与偏好对，随后**仅以量规为证据**预测偏好次序。代理的预测准确率即充当量规质量奖励，激励模型产出内部一致且可迁移的量规。仅用约 5 万条数据样本，Proxy-GRM 便在 VL-RewardBench、Multimodal Reward Bench 与 MM-RLHF-Reward Bench 上取得最先进（SOTA）结果，超越以四倍数据量训练的方法。消融研究表明 Proxy-SFT 是比 Proxy-RL 更强的验证器，且隐式奖励聚合表现最佳。至关重要的是，学得的量规可迁移至未见过的评测器，在无需额外训练的情况下提升测试时的奖励准确率。代码开源于 <https://github.com/Qwen-Applications/Proxy-GRM>。

---

## 1 引言

视觉-语言模型（VLM）(Alayrac et al., 2022) 在视觉问答、图像描述与复杂多模态推理等广泛任务上进展迅速 (Liu et al., 2023; Bai et al., 2023)。随着这些模型日益被部署到开放式生成场景，可靠地评估并引导其输出质量已成为核心挑战 (Ouyang et al., 2022; Zheng et al., 2023; Lambert et al., 2025)。早期方法采用标量奖励模型，将响应质量压缩为单一数值分数；这类模型虽然高效，可解释性却很差——它们只揭示一个响应**是否**被偏好，而不揭示**为什么** (Ouyang et al., 2022)。为克服这一缺陷，生成式奖励模型（GRM）在偏好裁决之外同时产出自然语言批评（critique）(Zhang et al., 2024; Ankner et al., 2024; Xiong et al., 2025)，如图 1 所示，其典型结构为：

$$
\texttt{<rubric>}\;\mathcal{R}\;\texttt{</rubric>}\;\texttt{<eval>}\;\mathcal{E}\;\texttt{</eval>}\;\texttt{<answer>}\;\mathcal{A}\;\texttt{</answer>}, \tag{1}
$$

其中 $\mathcal{R}$ 是一组评估准则（即**量规**），$\mathcal{E}$ 是逐准则的评估，$\mathcal{A}\in\{1,2\}$ 是最终偏好裁决 (Kim et al., 2023; Kim et al., 2024)。这一分解赋予 GRM 结构与可解释性。

> **图 1**：经由代理（proxy）的量规质量奖励建模总览。左上：传统 GRM 训练只使用来自最终答案的奖励；量规质量是隐式/潜在的，使监督沦为黑箱，模型可能在没有可靠引导的情况下"猜对"答案。右上：本文框架在最终答案之外显式生成可迁移量规，并用一个代理模型计算量规质量奖励来优化 GRM——该代理在给定问题与所生成量规的条件下预测偏好；代理预测正确即可为改进量规质量提供强引导。底部：在 VL-RewardBench、Multimodal Reward Bench 与 MM-RLHF-Reward Bench 上的数据效率比较表明，本文方法（1× 数据）优于以 ≥4× 数据量训练的基线。

尽管这一范式广受欢迎，整条流水线的可靠性却关键取决于其第一步——量规 $\mathcal{R}$ 的质量。一个含糊、不完整或存在微妙偏置的量规会让后续的批评毫无意义。GRM 的训练方式又进一步加剧了这一问题：现有方法 (Wang et al., 2025a; Shen et al., 2025) **只针对最终结果优化**——对 $\mathcal{A}$ 施加奖励或在裁决 token 上计算损失——中间量规完全未受监督、未被验证。其后果是，模型可能学会生成"为预定答案进行事后辩护（rationalize）"而非"引导有原则评估"的量规，这与思维链推理中的事后合理化（post-hoc rationalization）现象类似 (Turpin et al., 2023; Lanham et al., 2023)。此类量规的泛化性与可迁移性都很差：交给独立评测器或用于不同任务分布时，它们无法引出准确的偏好判断。

最常见的补救手段是 LLM-as-a-Judge：提示一个外部模型为生成的量规打分 (Zheng et al., 2023; Youssef et al., 2024; Chen et al., 2023)。这存在两个根本局限：其一，计算昂贵，在训练规模上不可行；其二，更关键的是它无法闭合训练环路——外部判断不可微，无法充当 GRM 自身强化学习的奖励，量规质量因此永远处于开环状态。近期的纯文本工作探索了从标注中学习量规 (Xie et al., 2025) 或将量规用作结构化奖励信号 (Gunjal et al., 2025; Xu et al., 2026)，但没有一项工作解决多模态场景下的闭环量规验证。

由此本文提出问题：**能否学习生成可被独立智能体验证的量规，并将该信号纳入训练环路？** Proxy-GRM 给出肯定回答——它引入一个专用**代理评测器（proxy evaluator）**来度量**量规可迁移性（rubric transferability）**：即一个量规被交给独立模型后，仍能引出正确偏好判断的程度。本文训练 Proxy-SFT 与 Proxy-RL 两种代理变体，并将其一致性信号用作闭环奖励。一个关键发现是：Proxy-SFT 出人意料地优于 Proxy-RL，揭示出纯结果的 RL 监督可能损害过程级评估的保真度。Proxy-GRM 仅用约 5 万条训练样本（约为可比方法的 1/4）在 VL-RewardBench、Multimodal Reward Bench 与 MM-RLHF-Reward Bench 上取得 SOTA，其量规无需再训练即可有效迁移到未见过的评测器模型。

**本文主要贡献**如下：

1. 识别出量规可迁移性是多模态 GRM 质量中一个关键而未被充分探索的维度，并提出基于代理的验证作为一个有原则、可扩展且与训练环路兼容的解决方案；
2. 提出 Proxy-GRM——一个将独立代理评测器作为量规质量信号集成进 RL 的闭环训练框架，直接解决 LLM-as-a-Judge 方法的开环局限；
3. 训练两种代理变体（Proxy-SFT 与 Proxy-RL）并发现 Proxy-SFT 更优，揭示出结果级 RL 优化与过程级评估保真度之间的张力；
4. 仅用约 5 万训练样本（约为可比方法的 1/4）在 VL-RewardBench、Multimodal Reward Bench 与 MM-RLHF-Reward Bench 上取得 SOTA，且量规无需再训练即可泛化到未见评测器。

---

## 2 相关工作

### 2.1 面向视觉-语言模型的奖励模型

奖励模型是使大语言模型与视觉-语言模型同人类偏好对齐的关键组件 (Christiano et al., 2017; Stiennon et al., 2020; Bai et al., 2022)。早期方法主要采用输出标量分数的判别式奖励模型 (Bradley and Terry, 1952; Zhu et al., 2024)。Bradley–Terry 模型 (Ouyang et al., 2022) 及其变体被广泛用于参数化成对偏好。在多模态领域，IXC-2.5-Reward (Zang et al., 2025)、MM-RLHF-Reward (Zhang et al., 2025b) 与 LLaVA-Critic (Xiong et al., 2025) 已将这些思想扩展到图文输入。

近来，生成式奖励模型（GRM）(Jia et al., 2025; Jia et al., 2026) 因能在偏好判断之外产出可解释批评而备受关注 (Zhang et al., 2024; Ankner et al., 2024; Mahan et al., 2024)。R1-Reward (Zhang et al., 2025a) 引入稳定的强化学习技术，训练带思维链的多模态奖励模型；Unified-Reward (Wang et al., 2025b) 提出经由结构化思考统一处理文本与多模态评测的框架。这些模型通常遵循第 1 节所述的"量规—评估—答案"分解，但其中没有任何一个显式验证或优化所生成量规的质量。

### 2.2 基于量规的评测与奖励建模

结构化评估准则的使用在教育测评中历史悠久 (Brookhart, 2013)，近来被引入 LLM 评测。JudgeLM (Zhu et al., 2023) 与 Prometheus (Kim et al., 2023; Kim et al., 2024) 率先在 LLM-as-judge 评测中使用细粒度量规。Auto-Rubric (Xie et al., 2025) 提出从人类偏好数据学习隐式量规权重，并将其转化为显式、可解释的量规用于奖励建模。该工作凸显了量规质量的重要性，但其焦点在于从既有标注中抽取量规，而非端到端地生成并验证量规。

在强化学习语境下，Rubrics-as-Rewards (Gunjal et al., 2025) 证明量规结构化的奖励能把 RL 从可验证领域（如数学）扩展到创意写作与开放对话等主观任务；其做法是用量规把复杂评估分解为若干可独立打分的子准则。面向基于量规奖励建模的交替强化学习（Alternating RL for Rubric-based Reward Modeling）(Xu et al., 2026) 进一步提出量规精炼与奖励模型训练之间的交替优化方案，服务于不可验证任务的 LLM 后训练。

CritiqueVLM (Wu et al., 2025) 探索了多模态模型的批评式评测，但未涉及量规可迁移性；RubricEval (Bhat, 2023) 提出用量规对语言模型进行系统化评测，但聚焦基准设计而非量规生成质量。本文与既有工作的根本区别在于：(i) 在训练期引入外部代理智能体验证量规可迁移性；(ii) 将该验证信号作为奖励分量集成进 RL 环路，实现量规质量的端到端优化。

### 2.3 面向语言与视觉-语言模型的强化学习

基于人类反馈的强化学习（RLHF）(Christiano et al., 2017; Ziegler et al., 2019) 及其变体——包括直接偏好优化（DPO）(Rafailov et al., 2023)、组相对策略优化（GRPO）(Shao et al., 2024) 与近端策略优化（PPO）(Schulman et al., 2017)——已成为对齐生成模型的标准工具。在 VLM 领域，RLHF-V (Yu et al., 2024a) 与 RLAIF-V (Yu et al., 2024b) 展示了减少幻觉、提升响应质量的有效偏好学习。

近期工作开始探索用 RL 训练奖励模型本身。R1-Reward (Zhang et al., 2025a) 采用精心设计的奖励信号与稳定 RL 训练多模态奖励模型。本文沿此方向推进，引入一种新颖的、基于代理的奖励分量来评估模型中间推理产物（量规）的质量而非只评估最终输出，弥合结果监督与过程监督之间的鸿沟 (Lightman et al., 2023; Uesato et al., 2022)。

### 2.4 过程监督与验证

结果监督与过程监督的区分在数学推理领域已被广泛研究 (Lightman et al., 2023; Wang et al., 2024)。过程奖励模型（PRM）评估中间推理步骤，比只评估最终答案的结果奖励模型（ORM）提供更密集的反馈。本文的代理式量规验证可视为作用于量规生成步骤的一种过程监督：除奖励正确的最终裁决外，还额外奖励生成**可被独立验证**的量规。这一视角将本文与自验证（self-verification）(Weng et al., 2023) 及批评模型 (McAleese et al., 2024; Gou et al., 2023) 的近期进展联系起来，同时引入了经由代理智能体进行外部验证的新元素。

---

## 3 方法

本文提出 **Proxy-GRM**——一个训练具备可迁移量规生成能力的多模态生成式奖励模型的框架，包含两大组件：(1) 面向量规评估的**代理智能体训练**；(2) 面向策略模型的**代理引导强化学习**。图 2 给出完整流水线总览。下文依次介绍问题形式化（3.1 节）、代理智能体训练（3.2 节）、策略模型训练（3.3 节）与推理流程（3.4 节）。

> **图 2**：Proxy-GRM（可迁移量规）训练流水线。(A) 数据与蒸馏。合并多个 VLM 偏好数据集并施加自动过滤，得到约 6 万样本；教师模型将数据划分为 Correct 与 Hard 两个子集，分别用于训练代理智能体与冷启动/RL 训练 Proxy-GRM。(B) 代理智能体训练。从 Qwen2.5-VL-7B-Instruct 出发，先 Proxy-SFT、后 Proxy-RL，训练出一个提供量规质量评估信号的评测器。(C) Proxy-GRM 训练。先用 SFT 冷启动 GRM 策略，随后以准确率、代理（量规质量）与格式三类奖励进行强化学习，并以 GRPO 更新。(D) 可迁移量规。若一个独立智能体能依循某量规选出正确答案，则该量规可迁移；否则不可迁移——尽管看似合理，却可能导致错误判断。

### 3.1 问题形式化

给定多模态查询 $q$（由文本问题与关联图像 $\mathcal{I}$ 组成）以及候选响应对 $(r_{1},r_{2})$（依人类判断其中一个更优），生成式奖励模型 $\pi_{\theta}$ 产出结构化输出：

$$
y=\pi_{\theta}(\mathcal{I},q,r_{1},r_{2})=\underbrace{\texttt{<rubric>}\,\mathcal{R}\,\texttt{</rubric>}}_{\text{评测量规}}\underbrace{\texttt{<eval>}\,\mathcal{E}\,\texttt{</eval>}}_{\text{详细评估}}\underbrace{\texttt{<answer>}\,\mathcal{A}\,\texttt{</answer>}}_{\text{偏好响应}}, \tag{2}
$$

其中 $\mathcal{R}$ 为评估准则集合（量规），$\mathcal{E}$ 为逐准则评估，$\mathcal{A}\in\{1,2\}$ 为更优响应的索引。模型的训练目标是最大化产出正确偏好 $\mathcal{A}^{*}$ 的概率。

本文定义**量规可迁移性**为如下性质：生成的量规 $\mathcal{R}$ 足够信息丰富且无偏，使得一个独立评测器在**仅给定** $(\mathcal{I},q,r_{1},r_{2},\mathcal{R})$ 的情况下也能得出正确偏好 $\mathcal{A}^{*}$。形式化地，对代理智能体 $\phi$：

$$
\text{Transferability}(\mathcal{R})=\mathbf{1}\!\left[\phi(q,\mathcal{I},r_{1},r_{2},\mathcal{R})=\mathcal{A}^{*}\right]. \tag{3}
$$

该定义的直觉是：好量规应当**可迁移**——它应能引导任何胜任的评测器做出正确判断，而不仅仅是生成它的那个模型。

### 3.2 代理智能体训练

为使量规可迁移性可操作化，本文引入一个专用代理智能体，其职责被严格限定为**消费（consume）量规而非生成量规**。代理的输入为原始查询 $q$、图像 $\mathcal{I}$、候选响应 $(r_{1},r_{2})$ 与外部提供的量规 $\mathcal{R}$，遵循 $\texttt{<think>}\ldots\texttt{</think>}\texttt{<answer>}\ldots\texttt{</answer>}$ 的格式产出逐准则评估 $\mathcal{E}$ 与最终裁决 $\mathcal{A}\in\{1,2\}$，而不自行生成量规。代理的准确率因而直接度量了量规可迁移性：成功意味着 $\mathcal{R}$ 编码了足够的判别性信息；失败则说明 $\mathcal{R}$ 欠规格或具有误导性。

**Proxy-SFT**。在 5k 条正确蒸馏样本上对 Qwen2.5-VL-7B-Instruct (Bai et al., 2025b) 做监督微调（译注：原文此处因 HTML 数学渲染重复而写作 "55k"，据附录 B.3 的数据分配表应为 5k）。每个训练实例由完整输入元组 $(q,\mathcal{I},r_{1},r_{2},\mathcal{R})$ 与真值评估及裁决配对构成，其中量规 $\mathcal{R}$ 取自高质量标注。训练目标为标准交叉熵损失。Proxy-SFT 旨在内化"忠实遵循给定量规、产出有据偏好判断"的技能，不存在任何偏离或重新诠释所给准则的动机。

**Proxy-RL**。从 Proxy-SFT 检查点出发，在 10k 样本上继续以强化学习训练代理（译注：原文 "1010k" 同为渲染重复，应为 10k），唯一奖励信号为二元准确率——预测裁决 $\mathcal{A}$ 是否匹配真值标签 $\mathcal{A}^{*}$。该阶段旨在使代理在更多样的量规措辞与质量水平下更鲁棒地充当量规消费者。两种配置共享一个重要设计原则：**代理智能体只被训练去遵循量规，而不生成量规**。这一严格专化确保在策略模型训练期间，代理是量规质量忠实且独立的裁判，其裁决不会被自身的量规生成倾向所混淆。

### 3.3 策略模型训练

图 2 展示了 Proxy-GRM 的整体架构，框架由两个顺序训练阶段构成。

#### 阶段 1：冷启动监督微调

在 10k 条正确蒸馏样本上微调基础 VLM（Qwen2.5-VL-7B-Instruct），得到 Proxy-GRM-SFT。该阶段教会模型结构化输出格式，并为 RL 提供合理的初始化：

$$
\mathcal{L}_{\mathrm{cold}}=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{cold}}}\left[\log\pi_{\theta}(y\mid x)\right]. \tag{4}
$$

#### 阶段 2：代理引导强化学习

从 Proxy-GRM-SFT 出发，用 GRPO (Shao et al., 2024) 训练策略模型，复合奖励函数包含三个分量：

**准确率奖励 $r_{\text{acc}}$**：评估策略模型的最终裁决是否正确。

$$
r_{\text{acc}}=\begin{cases}+1&\text{若 }\mathcal{A}=\mathcal{A}^{*}\\-1&\text{否则}\end{cases} \tag{5}
$$

**格式奖励 $r_{\text{format}}$**：评估输出是否遵循带正确 XML 标签的规定结构化格式。

$$
r_{\text{format}}=\begin{cases}+1&\text{若输出符合格式}\\0&\text{否则}\end{cases} \tag{6}
$$

**代理奖励 $r_{\text{proxy}}$**：把生成的量规 $\mathcal{R}$ 传给冻结的代理智能体 $\phi$，检验 $\phi$ 能否仅凭 $\mathcal{R}$ 正确识别更优响应，以此评估量规可迁移性。

$$
r_{\text{proxy}}=\begin{cases}+1&\text{若 }\phi(q,\mathcal{I},r_{1},r_{2},\mathcal{R})=\mathcal{A}^{*}\\-1&\text{否则}\end{cases} \tag{7}
$$

复合奖励为：

$$
r=r_{\text{acc}}+r_{\text{proxy}}+0.5\cdot r_{\text{format}}. \tag{8}
$$

这一设计形成了协同的训练动力学：$r_{\text{acc}}$ 确保策略模型学会做出正确判断，$r_{\text{proxy}}$ 确保生成的量规鲁棒且可迁移，$r_{\text{format}}$ 维持结构合规。代理智能体在策略训练全程保持冻结，纯粹充当评测器，不与策略共同适应（co-adapt）。

### 3.4 带代理验证的推理

推理时，Proxy-GRM 有两种运行模式：

- **标准模式**：策略模型生成结构化输出，最终答案直接用作偏好裁决。该模式不需要代理智能体。
- **代理验证模式**：策略模型生成含量规 $\mathcal{R}$ 的结构化输出；量规连同原始输入被传给代理智能体。若代理裁决与策略模型一致，则以高置信度接受预测；若不一致，仍采用策略模型的答案（因为代理也可能出错），但可将分歧记录下来供下游分析。

---

## 4 实验

本文从三个维度对 Proxy-GRM 进行全面评估：(1) 与最先进方法相比的奖励建模整体性能；(2) 关于代理智能体选择与多代理配置的分析研究；(3) 量规向未见评测器模型的可迁移性。

### 4.1 实现细节

**基础模型**。策略模型（Proxy-GRM）与代理智能体（Proxy-SFT、Proxy-RL）均以 Qwen2.5-VL-7B (Bai et al., 2025b) 为基础模型；数据蒸馏的教师模型为 Qwen3-VL-235B-A22B (Bai et al., 2025a)。

**训练数据**。从 LLaVA-Critic-113k (Xiong et al., 2025)、RLAIF-V (Yu et al., 2024b)、RLHF-V (Yu et al., 2024a) 与 MMIF-23k (Zhang et al., 2025b) 中整理出 6 万条偏好样本。经蒸馏与过滤后，2.5 万条正确样本用于代理智能体与策略冷启动训练，其余 3.5 万条充当 RL 训练数据。

**训练细节**。Proxy-SFT 与 Proxy-GRM 的 SFT 阶段学习率为 $1\times 10^{-5}$，余弦调度，训练 1 轮；Proxy-RL 与 Proxy-GRM 的 RL 阶段使用 GRPO，学习率 $5\times 10^{-6}$，组大小 $G=7$。

**评测基准**。在三个成熟的多模态奖励基准上评测：

- **VL-RewardBench** (Li et al., 2025)：覆盖通用多模态理解、幻觉检测与视觉推理的综合基准，含 1,247 个偏好对；
- **Multimodal Reward Bench** (Yasunaga et al., 2025)：覆盖正确性、偏好、知识、数学、编程、安全与 VQA 的多样基准，含 5,000 个样本；
- **MM-RLHF-Reward Bench** (Zhang et al., 2025b)：包含多选题、长文本、短文本、安全与视频评测任务的高难度基准。

**基线**。比较对象包括：(1) 闭源模型 GPT-4o、Claude-3.5/3.7-Sonnet 与 Gemini-1.5-Pro/Flash；(2) 开源奖励模型 IXC-2.5-Reward (Zang et al., 2025)、R1-Reward (Zhang et al., 2025a)、MM-RLHF-Reward (Zhang et al., 2025b) 与 Unified-Reward-Think (Wang et al., 2025a)；(3) 自身消融基线 Proxy-GRM-SFT（仅冷启动）与 Proxy-GRM-RL（无代理奖励的 RL）。

**表 1**：VL-RewardBench、Multimodal Reward Bench 与 MM-RLHF-Reward Bench 的综合结果（最优以绿色底标出，次优以黄色底标出）。

| 模型 | 代理智能体 | 数据规模 | VL-RewardBench OverallAcc | VL-RewardBench MacroAcc | Multimodal Reward Bench | MM-RLHF-Reward Bench Acc | MM-RLHF-Reward Bench Acc+ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4o (2024-08-06) | 无 | – | 65.80 | 62.40 | 70.80 | 58.23 | 26.01 |
| Claude-3.5-Sonnet (2024-06-22) | 无 | – | 55.30 | 53.60 | 71.50 | 62.94 | 26.11 |
| Claude-3.7-Sonnet | 无 | – | 66.31 | 66.53 | 71.90 | 82.35 | 65.22 |
| VITA-1.5 | 无 | – | 16.48 | 16.53 | 53.60 | 20.58 | 2.78 |
| SliME | 无 | – | 19.04 | 17.64 | 42.00 | 17.10 | 1.76 |
| NVLM-D-72B | 无 | – | 40.10 | 44.10 | – | 34.70 | 6.52 |
| Llama-3.2-90B | 无 | – | 56.20 | 53.90 | – | 35.29 | 10.86 |
| Qwen2-VL-72B | 无 | – | 39.50 | 43.00 | 70.90 | 48.23 | 13.04 |
| IXC-2.5-Reward | 无 | >200k | 65.80 | 70.00 | 66.60 | 71.18 | 50.00 |
| R1-Reward | 无 | >200k | 71.92 | 71.44 | 82.20 | 80.59 | 54.35 |
| Unified-Reward-SFT | 无 | >200k | 66.10 | 66.50 | – | – | – |
| Unified-Reward-Think | 无 | >200k | 73.80 | 72.30 | – | – | – |
| Proxy-GRM-SFT | 无 | 10k | 69.53 | 69.18 | 85.18 | 79.41 | 52.17 |
| Proxy-GRM-RL | 无 | 45k | 72.17 | 71.18 | 85.28 | 82.94 | 56.52 |
| Proxy-GRM-RL | Proxy-SFT | 50k | **75.22** | **73.93** | **85.62** | **82.94** | 56.52 |
| Proxy-GRM-RL | Proxy-RL | 60k | 73.38 | 72.38 | 84.78 | 81.76 | **58.70** |

（译注：表中"无代理"一行的 MM-RLHF-Reward Bench Acc 为 82.94，与附录表 8 及附录 D.3 正文给出的同一配置数值 80.59 不一致，疑为原文笔误；另 4.2 节正文的"渐进提升"叙述亦应理解为 79.41% → 80.59% → 82.94%，见附录 D.3。）

### 4.2 主要结果

本文在三个基准上呈现主要比较结果，证明 Proxy-GRM 以显著更少的训练数据取得最先进性能。综合结果见表 1，各基准的详细结果见附录 D。

**VL-RewardBench 结果**。如表 1 所示，以 Proxy-SFT 为代理智能体的 Proxy-GRM-RL 取得最高的总体准确率 75.22% 与宏平均准确率 73.93%，超越所有既有方法。相比此前最佳开源模型 Unified-Reward-Think（总体 73.80%），本文方法带来 1.42 个百分点的提升；相对 R1-Reward（71.92%），优势扩大到 3.30 个百分点。此外，Proxy-SFT 作为代理始终优于 Proxy-RL，这一出人意料的发现将在 4.3 节详细剖析。

**Multimodal Reward Bench 结果**。在该基准上，以 Proxy-SFT 为代理的 Proxy-GRM-RL 取得最佳总体分 85.62%，比 R1-Reward（82.2%）高 3.42 个百分点，并以可观差距超越包括 Claude-3.7-Sonnet（71.9%）与 GPT-4o（70.8%）在内的全部闭源模型。在包括编程与数学等困难维度上的一致提升表明：代理引导的量规生成交给模型的是**本质上更可靠的评估策略**，而非对特定任务模式的过拟合。

**MM-RLHF-Reward Bench 结果**。在该基准上，以 Proxy-SFT 为代理的 Proxy-GRM-RL 取得最高 Acc 82.94%，超过 R1-Reward（80.59%）；除 Acc+ 指标上的 Claude-3.7-Sonnet 外，超越所有闭源模型。从 Proxy-GRM-SFT（79.41%）到 Proxy-GRM-RL 的渐进提升清晰展示了强化学习与代理引导量规验证的互补收益。

**数据效率**。Proxy-GRM 的一大优势是数据效率。R1-Reward 与 Unified-Reward-RL 在超过 20 万条样本上训练，而 Proxy-GRM 仅使用 5 万条样本——约少 4 倍。尽管数据量大幅缩减，Proxy-GRM 仍持续超越这些基线，表明代理引导的量规验证比单纯扩大训练数据提供了更高效的学习信号。

> **图 3**：不同代理智能体配置下所生成量规的定性比较。给定同一多模态输入与成对偏好（$r_1$ 更优），比较四种变体：RL（无代理）产出准则冗余且裁决错误；RL+Qwen2.5-VL-3B 未能修正重复并扭曲权重分配，答案同样错误；RL+Qwen2.5-VL-32B 引入更具信息量的准则（*Contextual Awareness*，语境觉察）并得到正确裁决；RL+Proxy-SFT 生成判别性最强的量规，含 *Contextual Relevance*（语境相关性）准则、权重均衡、预测正确。更强的代理智能体引导策略产出更具体、更不冗余的量规。

**表 2**：以 Proxy-GRM-RL 为基础模型的代理智能体选择结果（最优以绿色底标出）。

| 代理智能体 | 平均分 | VL-Reward Bench | Multimodal Reward Bench | MM-RLHF-Reward Bench |
| --- | --- | --- | --- | --- |
| **Proxy-SFT** | **81.26** | **75.22** | **85.62** | **82.94** |
| Proxy-RL | 79.97 | 73.38 | 84.78 | 81.76 |
| Unified-Reward-SFT | 81.18 | 74.98 | **85.62** | **82.94** |
| Unified-Reward-Think | 79.77 | 73.14 | 85.00 | 81.18 |
| R1-Reward | 80.59 | 73.54 | 85.28 | **82.94** |
| Qwen2.5-VL-3B-Instruct | 77.87 | 72.25 | 84.88 | 76.48 |
| Qwen2.5-VL-7B-Instruct | 79.76 | 73.70 | 84.98 | 80.60 |
| Qwen2.5-VL-32B-Instruct | 80.25 | 74.02 | 84.98 | 81.76 |

### 4.3 分析研究

本文进行了大量分析研究以理解 Proxy-GRM 背后的设计选择。

**代理智能体选择**。一个自然的问题是什么构成好的代理智能体。本文系统比较三类代理：(1) 不同规模的通用指令遵循模型（Qwen2.5-VL-3B/7B/32B-Instruct）；(2) 既有奖励模型（R1-Reward、Unified-Reward-SFT、Unified-Reward-Think）；(3) 专用的量规评测代理（Proxy-SFT、Proxy-RL）。表 2 汇总了结果。

**SFT 与 RL 代理之辨**。跨自有代理与外部奖励模型的一致发现是：**SFT 型模型作为代理优于 RL 型模型**。Proxy-SFT 在全部三个基准上超过 Proxy-RL（VL-RewardBench 上 75.22% 对 73.38%）；类似地，Unified-Reward-SFT 超过 Unified-Reward-Think（74.98% 对 73.14%）。

**表 3**：奖励反馈配置。每行给出准确率奖励（$r_{\text{acc}}$）与代理奖励（$r_{\text{proxy}}$）四种组合结果下的复合奖励值；最右列表示 $r_{\text{proxy}}$ 在最终奖励中是否被置零。

| 配置 | $r_{\text{acc}}=1$, $r_{\text{proxy}}=1$ | $r_{\text{acc}}=1$, $r_{\text{proxy}}\neq 1$ | $r_{\text{acc}}\neq 1$, $r_{\text{proxy}}=1$ | $r_{\text{acc}}\neq 1$, $r_{\text{proxy}}\neq 1$ | 置零 $r_{\text{proxy}}$ |
| --- | --- | --- | --- | --- | --- |
| fb1 | 1.5 | 0.5 | −1.0 | −1.0 | ✗ |
| fb2 | 1.5 | 0.5 | −1.5 | −1.0 | ✗ |
| fb3 | 1.5 | 0.5 | −1.5 | −2.0 | ✗ |
| fb4 | 1.0 | 0.5 | −1.0 | −1.0 | ✗ |
| fb5 | 1.5 | 0.5 | −1.0 | −1.0 | ✓ |
| fb6 | 1.5 | 0.5 | −1.5 | −1.0 | ✓ |
| fb7 | 1.5 | 0.5 | −1.5 | −2.0 | ✓ |

本文将其归因于结果监督 RL 中的一种根本张力：只以最终答案奖励训练的模型可能发展出**内部不一致的评估过程**——通过有缺陷的推理得出正确裁决，或生成看似合理却并不忠实反映其实际决策过程的评估。当此类模型被用作量规评测器时，其不一致的评估过程会带来嘈杂且不可靠的量规质量信号。相比之下，SFT 模型被训练去忠实复现教师的评估过程，因而量规遵循行为更可靠。值得注意的是，R1-Reward 这一在训练中纳入了部分过程级监督的 RL 模型，取得了高于 Proxy-RL 与 Unified-Reward-Think 的代理性能（73.54%），为"过程级一致性是有效量规评估的关键"提供了进一步证据。

**模型规模效应**。在通用指令模型中，更大模型带来更好的代理性能：Qwen2.5-VL-32B-Instruct（VL-RewardBench 74.02%）优于 7B（73.70%），后者又优于 3B（72.25%）。然而，3B 模型在 MM-RLHF-Reward Bench 上反而使性能降到无代理基线之下（76.48% 对 80.59%），表明能力不足的代理智能体会给训练信号注入有害噪声。如图 3 所示，3B 模型在 RL 早期无法区分高质量与低质量量规，提供本质上是随机的反馈，误导策略模型对"什么是好量规"的认知。

**专用与通用代理之辨**。Proxy-SFT 尽管基于 7B 模型却取得最佳总体性能，在 VL-RewardBench 上甚至超过大得多的 Qwen2.5-VL-32B-Instruct（75.22% 对 74.02%）。这证明了面向量规评估的专门化训练的价值——5k 条定向训练样本注入了通用指令微调无法提供的量规遵循能力。如图 3 所示，尽管 32B 模型引入了一条新准则，它对量规关键方面的理解仍弱于 Proxy-SFT。

**奖励配置分析**。默认配置使用隐式奖励聚合（即式 8 的简单相加）。本文研究了基于代理反馈显式修改准确率奖励是否可能比简单叠加两信号更有效。表 3 定义了七种备选反馈配置，它们依据代理结果修改 $r_{\text{acc}}$，对量规不一致预测施加不同程度的惩罚。

> **图 4**：三种基准上不同代理奖励反馈配置的平均分比较。每个子图报告各变体的平均分（柱上标注数值），三角标记指示每个基准内表现最佳与最差的变体。总体而言，基础（相加）配置在 VL-RewardBench 上得分最高（译注：原文此处提及 "Instructability"，疑指 Multimodal Reward Bench，系笔误），而 MM-RLHF-RewardBench 上各变体波动较小、性能保持高位。

图 4 呈现的结果显示：**所有显式修改方案都劣于隐式相加基线**。最激进的惩罚方案（fb3，准确率与代理双双失败时复合奖励为 −2.0）导致训练不稳定；温和方案（fb1、fb4）比相加基线低 1–3 个百分点。在修改 $r_{\text{acc}}$ 后置零 $r_{\text{proxy}}$（配置 fb5–fb7）与保留它的效果相当，但两种做法都不及简单的相加形式。

本文假设：显式奖励修改造成功劳分配（credit assignment）混乱——当 $r_{\text{acc}}$ 依 $r_{\text{proxy}}$ 被修改后，策略无法清晰分辨奖励变化究竟源自其最终裁决还是其量规质量，从而妨碍二者的学习。相加形式通过保持两信号独立，保留了干净的功劳分配。

### 4.4 量规可迁移性分析

本文方法的核心主张是：代理引导训练产出的量规不仅**自洽**，而且**真正可迁移**——即这些量规对并未生成它们的外部模型同样有用。本文通过一个量规迁移实验加以验证。

**协议**。对评测基准中的每个样本，提取由各种量规源生成的量规 $\mathcal{R}$。这些量规随后被提供给三个评测器模型：Qwen2.5-VL-7B-Instruct、Qwen2.5-VL-32B-Instruct 与 Unified-Reward-SFT。本文度量外部提供的量规是否提升它们的偏好准确率。

**表 4**：VL-RewardBench、Multimodal Reward Bench、MM-RLHF-Reward Bench 上的量规可迁移性结果。

| 评测器 | 量规来源 | VL-Reward Bench | Multimodal Reward Bench | MM-RLHF-Reward Bench |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL-7B-Instruct | Qwen2.5-VL-7B-Instruct | 36.89 | 70.98 | 40.00 |
| | Qwen2.5-VL-32B-Instruct | 41.06 | 71.08 | 41.18 |
| | UnifiedReward-SFT | 42.02 | 72.36 | 41.76 |
| | Proxy-GRM-RL | 52.33 | 79.46 | 71.03 |
| | Proxy-GRM-RL + UnifiedReward-SFT | 53.87 | 79.82 | 73.79 |
| | Proxy-GRM-RL + Proxy-SFT | **54.13** | **80.00** | **73.79** |
| Qwen2.5-VL-32B-Instruct | Qwen2.5-VL-7B-Instruct | 40.02 | 62.46 | 40.59 |
| | Qwen2.5-VL-32B-Instruct | 43.30 | 72.48 | 42.35 |
| | UnifiedReward-SFT | 44.91 | 73.84 | 43.53 |
| | Proxy-GRM-RL | 66.37 | 80.02 | 70.83 |
| | Proxy-GRM-RL + UnifiedReward-SFT | 66.63 | 81.02 | 73.10 |
| | Proxy-GRM-RL + Proxy-SFT | **66.80** | **81.14** | **75.17** |
| UnifiedReward-SFT | Qwen2.5-VL-7B-Instruct | 49.00 | 63.08 | 58.82 |
| | Qwen2.5-VL-32B-Instruct | 51.32 | 65.16 | 60.00 |
| | UnifiedReward-SFT | 63.56 | 72.88 | 67.65 |
| | Proxy-GRM-RL | 67.84 | 82.16 | 75.18 |
| | Proxy-GRM-RL + UnifiedReward-SFT | 68.60 | 82.92 | 77.06 |
| | Proxy-GRM-RL + Proxy-SFT | **69.86** | **83.14** | **77.65** |

**结果**。表 4 给出量规迁移结果。在全部三个评测器模型与全部三个基准上，由 Proxy-GRM（以 Proxy-SFT 为代理训练）生成的量规持续提供最佳准确率（译注：原文简写作"由 Proxy-SFT 生成的量规"）。例如，当 Unified-Reward-SFT 使用 Proxy-GRM-RL + Proxy-SFT 的量规时，其 VL-RewardBench 准确率从 66.10%（不使用生成量规）提升到 69.86%，增益达 3.76 个百分点。

迁移分析呈现出若干规律。**其一**，Proxy-SFT 变体的量规全面超越其他所有来源的量规——包括比之大 4.5 倍的 Qwen2.5-VL-32B-Instruct——确认了代理引导训练**真正提升了量规质量**，而非教策略模型去利用特定的评估捷径。**其二**，改进在不同架构与规模的评测器模型之间泛化，说明量规捕捉的是**普遍的评估原则**而非模型特定的模式。**其三**，MM-RLHF-Reward Bench 上的增益尤为可观，表明可迁移量规在评测器本会挣扎的高难评测场景中价值最大。

这些结果验证了本文的核心论点：通过代理智能体面向**外部可验证性**优化量规，产出的是真正更具信息量、更可用的评估准则，而非自我服务的合理化辩护。

---

## 5 结论

本文提出了 Proxy-GRM——一个训练多模态生成式奖励模型以产出鲁棒、可迁移评测量规的框架。通过在 RL 训练期引入验证量规质量的代理智能体，本文解决了既有方法的一个根本局限：中间量规生成步骤缺乏任何质量信号。主要发现包括：(1) 代理引导的量规验证在三个基准上一致提升奖励模型性能，以少于可比方法 4 倍的训练数据取得 SOTA；(2) SFT 型代理智能体在量规评估上优于 RL 型代理，凸显验证任务中过程保真度的重要性；(3) 简单的隐式奖励聚合比基于代理反馈显式修改准确率奖励更有效；(4) Proxy-GRM 生成的量规能有效迁移到未见过的评测器模型，验证了本方法确实提升了量规可迁移性。作者相信，"外部可验证性"这一原则——将中间推理产物优化为对独立智能体有用的东西——代表着提升生成式奖励模型可靠性与可解释性的一个有前景方向。

---

## 附录要点（压缩概括）

1. **提示词（附录 A）**：图 5 给出蒸馏与 RL 共用的系统提示，要求模型产出细粒度评估准则、系统化应用准则并输出最终裁决；图 6 给出量规评估（代理）提示，代理以 $\texttt{<think>}/\texttt{<answer>}$ 格式在给定外部量规下做出判断。
2. **数据来源与整理（附录 B.1）**：从 LLaVA-Critic-113k、RLAIF-V、RLHF-V、MMIF-23k 四个多模态偏好数据集整理约 6 万样本，并施加三项自动过滤准则——质量（剔除退化或平凡可分的响应对）、难度（保留偏好区分非平凡的样本以提供有意义的学习信号）、相似度（按语义相似度去重避免冗余）。
3. **数据蒸馏（附录 B.2）**：以 Qwen3-VL-235B-A22B 为教师，对全部 6 万样本按 $\texttt{<rubric>}\mathcal{R}\texttt{</rubric>}\texttt{<eval>}\mathcal{E}\texttt{</eval>}\texttt{<answer>}\mathcal{A}\texttt{</answer>}$ 格式生成结构化批评。
4. **数据分配（附录 B.3）**：从蒸馏输出中选出教师判对的约 2.5 万条 Correct 样本；其余 3.5 万条（含教师判错者）作为 RL 的困难负样本。Correct 集分配为：Proxy-SFT 训练 5k、Proxy-RL 训练 10k、Proxy-GRM-SFT 10k。加上 3.5 万困难样本构成约 4.5 万的 RL 训练池，算上 5k 代理训练样本，完整 Proxy-GRM-RL 流程共消耗约 5 万样本——约为 R1-Reward、Unified-Reward-Think（各 >20 万样本）的 1/4。
5. **训练细节（附录 C）**：Proxy-GRM-SFT 用 ms-swift 实现，1 轮，学习率 $1\times 10^{-5}$，余弦衰减，训练样本 10k；Proxy-GRM-RL 用 verl + GRPO 实现，1 轮，学习率 $5\times 10^{-6}$，rollout 组大小 $G=7$，训练批大小 256，mini-batch 128，代理智能体全程冻结。代理两阶段（Proxy-SFT、Proxy-RL）采用与策略模型对应阶段类似的配置。
6. **分基准详细结果（附录 D）**：D.1 VL-RewardBench 上幻觉子类提升最显著（93.08% 对 R1-Reward 的 85.71%），说明代理引导促使策略产出对事实落地尤其敏感的准则，而推理子类增益较小；D.2 Multimodal Reward Bench 上 SFT 与 RL 变体均已大幅领先既有方法（最强配置 85.6%，超 R1-Reward 3.4 个点、超闭源模型 13–15 个点），各代理变体间差距也小于 VL-RewardBench；D.3 MM-RLHF-Reward Bench 上 Acc+ 最优为 63.04%（R1-Reward 作代理，追平 Claude-3.7-Sonnet 的量级并大幅超过 R1-Reward 自身的 54.35%），而 Qwen2.5-VL-3B 代理使性能退化到 76.47%，甚至低于仅 SFT 基线（79.41%）。
7. **多代理集成消融（附录 E.1）**：将四个最佳单体代理（Proxy-SFT、Unified-Reward-SFT、Qwen2.5-VL-7B/32B-Instruct）做两两与三三组合，代理奖励按 $r_{\text{proxy}}=\frac{1}{K}\sum_{k=1}^{K}r_{\text{proxy}}^{(k)}$（式 9）取平均。结果一致大幅劣化：最佳组合（PSFT + Q32B）在 VL-RewardBench 上仅 62.71%，比 PSFT 单独使用（75.22%）剧降 12.51 个百分点。原因是异质代理对"何为好量规"存在系统性分歧（PSFT 给高奖励的量规 Q7B 可能给低奖励），平均后的奖励信号互相抵消、退化为近随机噪声。这对"朴素融合多个奖励信号更稳"的直觉是一个有力反例：单一、校准良好的代理优于评估倾向错位的异质代理集成。
8. **代理训练数据量消融（附录 E.2）**：分别用 2.5k/5k/7.5k/10k 蒸馏样本训练 Proxy-SFT，5k 配置几乎全面最佳（VL-RB 75.22、MM-RB 85.62、Acc 82.94）；数据更多反而退化（7.5k 时 VL-RB 降至 72.81）。推测原因是样本边际质量递减：纳入更多样本必然引入量规遵循任务定义不清的困难/歧义样本，使代理过拟合于伪模式，输出校准不良的代理奖励。由于缺乏直接度量代理判断质量的专用基准，最优数据量只能通过完整 Proxy-GRM-RL 流水线的下游评测确定。
9. **更多案例分析（附录 F，图 8–10）**：在物体/属性描述、动作理解、博弈状态推理三类任务上定性比较：无代理 RL 倾向产出泛化或结构不良的量规（准则重复、判别力弱、权重错配），常导致错误裁决；加 3B 小代理收益有限，甚至引入无关准则或把权重移离关键失败模式（如幻觉控制）；32B 强代理持续以更贴合任务、更有信息量的准则（描述精度、避免虚构、博弈状态解读）替换冗余项，得到正确偏好；RL+Proxy-SFT 产出最鲁棒的量规——不冗余、权重均衡、显式针对任务特定错误模式，偏好判断最可靠。

---

## 创新点提炼

1. **把"量规可迁移性"确立为多模态 GRM 的关键质量维度并给出可操作定义**。以"独立代理仅凭量规能否复现正确偏好判断"（式 3）来形式化量规质量，将批评（rationale）从不可观测的潜变量转变为可度量、可优化的显式训练目标，直指 GRM 中"答案对但理由错"的事后合理化风险。

2. **代理引导的闭环量规验证**。训练严格专用于"消费"量规（而非生成量规）的轻量代理评测器，将其预测准确率作为量规质量奖励 $r_{\text{proxy}}$ 注入 GRPO 复合奖励（式 7–8），首次在多模态设置中闭合了量规质量监督回路——这是对 LLM-as-a-Judge 不可微、计算昂贵、无法进训练环路等开环局限的直接回应。

3. **"过程保真度优于纯结果优化"的重要实证发现**。SFT 型代理（Proxy-SFT、Unified-Reward-SFT）一致优于 RL 型代理（Proxy-RL、Unified-Reward-Think），揭示纯结果奖励的 RL 会让模型的评估过程内部不一致（凭 flawed 推理蒙对答案），因此验证类任务更需 SFT 式的过程级保真；R1-Reward（含过程级监督的 RL 模型）优于其他 RL 代理亦为佐证。

4. **极高的数据效率**。仅约 5 万样本（约为可比 SOTA 的 1/4）即在 VL-RewardBench（75.22%）、Multimodal Reward Bench（85.62%）、MM-RLHF-Reward Bench（82.94%）上取得 SOTA，证明显式的量规质量信号比单纯堆叠训练数据提供更高效的学习信号。

5. **量规真正可迁移、可复用**。学得的量规使未见过的评测器（7B/32B 通用模型、Unified-Reward-SFT）偏好准确率大幅提升（如 32B 评测器在 VL-RewardBench 上从 43.30 升至 66.80），无需任何再训练——量规本身成为可独立交付、跨模型复用的评估资产，也支撑了"代理验证式推理时校准"的推理模式。

6. **反直觉负结果的方法论启示**。多代理集成平均奖励一致大幅劣化（最佳组合比单一代理低 12.51 个百分点），说明评测器之间存在系统性分歧时信号互相抵消，单一、校准良好的代理优于异质集成；显式修改奖励配置（fb1–fb7）全部劣于简单相加，凸显保持功劳分配干净的重要性。这两条消融结论对多信号奖励设计具有普遍参考价值。

---

*注：本翻译基于 arXiv HTML（LaTeXML）转换的 Markdown 版本，已忽略站点导航残留与 LaTeX 转义符；参考文献列表从略，正文引用保留 (作者, 年份) 格式。原文中因 HTML 数学渲染产生的数字重复（如 3.2 节 "55k"/"1010k" 应为 5k/10k）、表 1 与附录表 8 在无代理配置上的数值不一致（82.94 对 80.59）、图 4 注中的 "Instructability" 等疑似笔误，已在相应位置以译注形式标明。*
