# 《Can We Perform Online RL for Image Editing without Editing Rewards?》中文精读翻译

## 论文信息

- **英文标题**：Can We Perform Online RL for Image Editing without Editing Rewards?
- **中文标题**：我们能否在没有编辑奖励的情况下进行图像编辑的在线强化学习？
- **作者**：Qichao Ma、Jikang Cheng（项目负责人）、Ling Liang、Zhaofei Yu、Tiejun Huang、Renye Yan（通讯作者）
- **机构**：北京大学（Peking University）
- **arXiv 链接**：<https://arxiv.org/abs/2608.22780>
- **提出的框架**：Lever-Edit

## 一句话核心总结

Lever-Edit 通过"学习一个与冻结的 T2I 奖励对齐的轻量描述器（前缀微调的连续查询嵌入），把相对编辑指令转写为自包含的反事实目标描述"这一杠杆机制，使成熟的文生图（T2I）奖励生态无需任何图像编辑专用奖励模型，即可为图像编辑的在线强化学习提供有效监督信号。

---

## 摘要

强化学习（Reinforcement Learning, RL）可以借助编辑专用奖励（editing-specific reward）对图像编辑进行直接的偏好优化，但这类奖励因昂贵的三元组监督（triplet supervision）与复杂的任务依赖校准而发展相对滞后。与之形成鲜明对比的是，文生图（text-to-image, T2I）生成拥有一个成熟且多样的奖励生态，覆盖语义对齐、美学、真实感、字形（glyph shape）以及其他视觉偏好。若能将这一生态扩展到图像编辑，将大幅拓宽基于 RL 优化所能触及的视觉偏好范围，由此引出本文的核心问题：**我们能否在不使用编辑奖励的情况下进行图像编辑强化学习？**

本文论证：标准的图像编辑评价维度具有映射到 T2I 奖励空间的潜力——图像质量可以直接迁移；提示遵循（prompt following）可以通过对"期望视觉状态"的描述来对齐；参考一致性（reference consistency）则可通过将需保留的源内容编码进该描述，完成一次粗粒度的语义转换。然而，编辑指令指定的是**相对变化**，而 T2I 奖励需要**自包含的目标描述**；此外，通用视觉-语言模型（Vision-Language Model, VLM）生成的语义上合法的描述，也可能与被冻结的奖励模型不兼容。为此，本文进一步提出 **Lever-Edit**：一个两阶段框架——先学习一个面向反事实目标描述（counterfactual target description）的奖励对齐描述器（captioner）并将其冻结，随后仅用迁移而来的 T2I 奖励优化编辑策略。实验表明，该方法在编辑对齐与源内容保持上可与基于编辑奖励的微调相竞争，并显著优于直觉式迁移基线。

---

## 1 引言

指令引导的图像编辑（instruction-guided image editing）允许用户通过自然语言指令修改一张现有图像，同时保留与所请求变化无关的内容（Huang et al., 2025）。尽管近期的扩散与流匹配（flow-based）编辑器已取得长足进展（Nichol et al., 2021; Liu et al., 2025b; Yan et al., 2026b），监督微调（Supervised Fine-Tuning, SFT）优化的是精选编辑样本对上的似然，而非任务层面的人类偏好。强化学习则转而对采样结果直接优化标量或带推理依据的反馈（Black et al., 2023; Fan et al., 2023; Guo et al., 2026; Liu et al., 2026）。

编辑 RL 的核心在于奖励。然而，构建一个可靠的编辑奖励需要联合推理参考图像（reference image）、编辑指令（editing instruction）与编辑输出，且需要昂贵的三元组级监督；"执行指令"与"保持源内容"之间所期望的平衡还随编辑类型而变化，使奖励的校准与泛化更加复杂（Wu et al., 2025; Luo et al., 2026; Zhao et al., 2026; Long et al., 2026）。

编辑质量通常沿三个维度刻画：**纯图像质量**（pure image quality）、**编辑执行**（edit execution）与**参考一致性**（reference consistency）（Luo et al., 2026; Zhao et al., 2026）。纯图像质量关注编辑输出的内在属性；编辑执行衡量所请求的变换是否被实现；参考一致性衡量与编辑无关的源内容是否得到保持。令人意外的是，本文发现这三个维度都表现出向 T2I 奖励空间映射的潜力：图像质量可直接评估；编辑执行可被重新表述为与"期望视觉状态描述"的对齐；参考一致性则可通过把应保留的源语义纳入该描述，实现粗粒度投影（如图 1 所示）。

> **图 1**：T2I 到编辑的奖励迁移动机。Lever-Edit 充当一个在线"杠杆"，撬动广阔 T2I 奖励社区的集体力量，把其成熟多样的视觉偏好迁移为图像编辑 RL 的有效监督。

这就引出一个核心问题：**我们能否在不使用编辑奖励、而只用 T2I 奖励的情况下进行图像编辑 RL？** 这一表述带来两点好处。其一，它使编辑 RL 可以直接接入成熟多样的图像生成奖励生态，包括语义对齐、美学、真实感、OCR、安全以及其他视觉偏好的奖励（Xu et al., 2023; Wu et al., 2023b; Kirstain et al., 2023; Ye et al., 2025; Zhu et al., 2026; Wang et al., 2025; Huang et al., 2026; Lamba et al., 2025）；这些奖励的后续改进也因此无需重新设计对齐目标即可迁移到编辑任务。其二，它免除了对昂贵的编辑评测器开发与部署的依赖。

然而，将 T2I 奖励直接用于图像编辑颇具挑战。**第一，两种任务的条件语义存在本质差异**：编辑指令定义的是相对于参考图像的变换，而 T2I 奖励期望的是对目标图像的自包含描述；直接使用指令虽然保留了所请求的变化，却遗漏了应保持不变的源语义，因此需要一种参考感知的转换，得到反事实目标描述。**第二，语义合法的目标描述未必与冻结的 T2I 奖励兼容**：现成的 VLM 可能生成看似合理的描述，其表示却与奖励模型的评估空间校准不佳，从而削弱优化信号。因此，有效的奖励迁移需要同时完成"编辑目标转换"与"显式奖励对齐"，这仍是一个有挑战的问题。

本文提出 **Lever-Edit**——一个奖励迁移框架，其中一个轻量的奖励对齐描述器充当**杠杆（lever）**：它使成熟的 T2I 奖励能够为图像编辑 RL 提供有效监督，而无需构建或调用任何编辑奖励。具体而言，Lever-Edit 采用两阶段设计：

- **第一阶段**：通过前缀微调（prefix tuning）适配一个 VLM，将参考图像与编辑指令映射为反事实目标描述。该描述联合刻画预期的修改与应保留的源语义；同时将描述器的表示与冻结的 T2I 奖励对齐，产出"奖励兼容的条件"，而非泛泛的图像描述。
- **第二阶段**：冻结该描述器，仅用迁移的 T2I 奖励优化扩散编辑策略；策略微调全程不使用任何编辑奖励。

该设计将奖励迁移与策略优化解耦，可作为插件应用于不同的 VLM 与图像编辑主干网络。

**本文贡献**如下：

1. 从奖励迁移的视角重新审视编辑质量的标准维度，识别其与通用图像评价的潜在对应关系，提出无需编辑专用奖励的图像编辑 RL；
2. 提出 Lever-Edit——一个两阶段框架，将参考条件化的编辑目标转换为奖励对齐的目标描述，并仅用冻结的 T2I 奖励微调编辑策略；
3. 通过实验证明：在策略微调阶段不使用图-图（image-to-image）奖励模型的情况下，迁移的 T2I 奖励即可提供有竞争力的编辑对齐与源内容保持能力。

---

## 2 相关工作

### 2.1 图像编辑的强化学习

强化学习无需密集的任务专属标注即可对视觉生成进行直接的偏好优化。DDPO（Black et al., 2023）与 DPOK（Fan et al., 2023）开创了文生图扩散模型的 RL 微调；DanceGRPO（Xue et al., 2025）与 Flow-GRPO（Liu et al., 2025a）随后将这一范式拓展到当代扩散与流匹配主干，确立了在线 RL 作为视觉生成通用后训练（post-training）框架的地位。后续工作通过前向过程策略更新、与预训练对齐的目标、熵自适应探索以及沿去噪轨迹的选择性干预等手段改进优化效率与稳定性（Zheng et al., 2026; Xue et al., 2026; Yan et al., 2025; Yan et al., 2026a）。

为 T2I 生成发展的 RL 范式随后被成功适配到指令引导图像编辑的偏好优化。在该 setting 下，RL 旨在进一步提升提示遵循、参考一致性与整体视觉质量，近期方法在这些维度上展现出可观的收益（Luo et al., 2026; Zhao et al., 2026; Guo et al., 2026）。

### 2.2 图像强化学习中的奖励建模

**图像生成**。文生图的奖励建模已演化为一个多样生态，涵盖通用人类偏好与任务专属标准。ImageReward（Xu et al., 2023）、HPS（Wu et al., 2023b; Wu et al., 2023a）与 PickScore（Kirstain et al., 2023）捕捉语义对齐与视觉吸引力等广泛判断；专门化奖励包括面向真实感的 RealGen（Ye et al., 2025）、面向视觉文本保真度的 TextPecker（Zhu et al., 2026）；UnifiedReward（Wang et al., 2025）支持逐点与成对的多模态评测，SigLIP（Zhai et al., 2023）提供通用的文本-图像对应性分数。此外，SpectraReward（Huang et al., 2026）将预训练 MLLM 以免训练的零样本方式改造为奖励模型（通过图像条件化的提示似然）。这种广度使 T2I 奖励生态成为更广泛视觉对齐任务的一个极具吸引力的监督来源。

**图像编辑**。编辑奖励针对三元组 $(I_{\mathrm{ref}}, q_{\mathrm{edit}}, I_{\mathrm{edit}})$，从输出质量、编辑执行与参考一致性三方面评估。EditReward（Wu et al., 2025）构建了超过 20 万条专家标注的偏好对并训练多维不确定性感知排序器；EditScore（Luo et al., 2026）遵循"基准构建→奖励数据制备→奖励模型训练→自集成→在线 RL"的完整流水线；FIRM Reward（Zhao et al., 2026）采用差异优先（difference-first）的 MLLM 流水线、专门化奖励模型训练与一致性调制奖励融合；SpatialReward（Long et al., 2026）进一步引入多阶段空间定位与验证流水线，辅以渐进式 SFT 与 GRPO 训练。这类分阶段构建是"评估相对编辑"内在困难的必然结果，伴随可观的设计与实现成本。而且，这些奖励大多仍聚焦于编辑执行与参考一致性，对 T2I 奖励所支持的更广泛对齐偏好覆盖有限——受限的奖励空间约束了图像编辑 RL 的优化潜力。

---

## 3 问题形式化

### 3.1 奖励迁移鸿沟（The Reward-Transfer Gap）

记 $I_{\mathrm{ref}}$ 为参考图像，$q_{\mathrm{edit}}$ 为编辑指令，$I_{\mathrm{edit}}$ 为从编辑策略采样的输出：

$$
I_{\mathrm{edit}} \sim \pi_\theta(\cdot \mid I_{\mathrm{ref}}, q_{\mathrm{edit}}). \tag{1}
$$

常规图像编辑 RL 优化定义在完整编辑三元组上的任务专属奖励：

$$
\max_\theta \; \mathbb{E}\!\left[R_{\mathrm{edit}}(I_{\mathrm{ref}}, q_{\mathrm{edit}}, I_{\mathrm{edit}})\right]. \tag{2}
$$

参考图像使 $R_{\mathrm{edit}}$ 得以同时判定"什么应当改变"与"什么应当保持不变"。相比之下，T2I 奖励遵循图-文接口：

$$
R_{\mathrm{T2I}}(I, c), \tag{3}
$$

其中 $c$ 被期望是对图像 $I$ 中视觉内容的**自包含描述**。因此，直接用 $q_{\mathrm{edit}}$ 替换 $c$ 会造成根本性的条件失配（conditioning mismatch）：编辑指令指定的是相对变换（例如"把汽车改成红色"），其目标状态依赖于具体参考图像；它既不描述完整的目标图像，也不暴露应被保留的源语义。其后果是，$R_{\mathrm{T2I}}(I_{\mathrm{edit}}, q_{\mathrm{edit}})$ 可能奖励了所请求的属性，却对参考特有内容的丢失不敏感。

因此，奖励迁移需要一个条件转换器（condition converter）$g:\mathcal{I}\times\mathcal{Q}\rightarrow\mathcal{C}$，将参考图像与相对编辑指令映射为自包含的目标描述：

$$
c_{\mathrm{tgt}} = g(I_{\mathrm{ref}}, q_{\mathrm{edit}}). \tag{4}
$$

编辑输出随后由 $R_{\mathrm{T2I}}(I_{\mathrm{edit}}, c_{\mathrm{tgt}})$ 评估。核心问题在于：$g$ 能否在保留编辑评估所需信息的同时，产出对冻结 T2I 奖励仍然有效的条件？

### 3.2 编辑质量三维度到 T2I 奖励空间的映射

编辑质量通常沿三个维度评估：纯图像质量、编辑执行与参考一致性（Luo et al., 2026; Zhao et al., 2026）。我们将其在概念上表示为：

$$
\mathbf{r}_{\mathrm{edit}}=
\begin{bmatrix}
r_{\mathrm{qual}}(I_{\mathrm{edit}})\\
r_{\mathrm{exec}}(I_{\mathrm{ref}}, q_{\mathrm{edit}}, I_{\mathrm{edit}})\\
r_{\mathrm{cons}}(I_{\mathrm{ref}}, q_{\mathrm{edit}}, I_{\mathrm{edit}})
\end{bmatrix}, \tag{5}
$$

且不假设三者的某种特定标量化（scalarization）方式。尽管其原生形式是参考条件化的，每一维度都与 T2I 奖励空间存在对应关系。

在三个维度中，**纯图像质量**可即时迁移：它是 $I_{\mathrm{edit}}$ 的内在属性，无需参照源图像或编辑指令即可评估。表面上的困难在于编辑执行与参考一致性——二者常规上都相对于 $I_{\mathrm{ref}}$ 定义。但对**编辑执行**而言，这种依赖主要源于条件的形式：指令 $q_{\mathrm{edit}}$ 描述的是相对变化，而评估目标关心的是"对应视觉状态是否出现在结果中"。一旦把预期结果表达为绝对的编辑后描述（absolute post-edit description），编辑执行就变成了 T2I 奖励已然解决的图-文对齐问题。**参考一致性**看似障碍最大，因为它显式关注 $I_{\mathrm{ref}}$ 与 $I_{\mathrm{edit}}$ 之间的关系。尽管如此，应保持不变的源语义可以与所请求的修改一起被纳入目标描述。目标条件因此可视为如下复合：

$$
c_{\mathrm{tgt}} = c_{\mathrm{change}} \oplus c_{\mathrm{preserve}}, \tag{6}
$$

其中 $c_{\mathrm{change}}$ 指明预期的修改，$c_{\mathrm{preserve}}$ 编码应保留的源语义。该表述把编辑执行与参考一致性重新塑造为 $I_{\mathrm{edit}}$ 与一个统一目标描述之间的对齐，使其可由 T2I 奖励联合评估。

需要强调的是，这一映射应被理解为一种**语义投影（semantic projection）**，而非 T2I 评估与图对评估之间的等价关系。文本条件无法完全编码低层身份信息或像素级精确保持，但它能够表示图像质量、编辑执行与参考一致性背后的主要语义。因此，尽管直接奖励迁移在接口层面是不适定的（ill-posed），核心编辑目标在原则上是可迁移的。实践挑战在于构建能忠实实现这一对应关系的目标条件。

### 3.3 三种直觉奖励迁移协议

式 (4) 提示了若干直接的迁移协议。本文介绍三种代表性变体，其差异在于如何实例化 $g$。

#### （1）直接编辑提示（Direct Edit Prompt）

最简单的协议直接令 $c_{\mathrm{tgt}} = q_{\mathrm{edit}}$。它没有转换成本并保留了所请求的变换，但把一条相对指令当作了绝对的图像描述：得到的奖励条件遗漏了参考特有语境，也没有对"应保留内容"的显式说明。

#### （2）在线 VLM 改写（Online VLM Rewriting）

也可以用一个通用 VLM 按需生成目标条件：

$$
c_{\mathrm{tgt}} = g_{\mathrm{VLM}}(I_{\mathrm{ref}}, q_{\mathrm{edit}}). \tag{7}
$$

该协议引入了参考感知，能更完整地表达预期的编辑后状态。但在奖励计算过程中反复调用 VLM 会带来推理开销；且其泛用描述是为一般语义描述而生，并未针对所选 T2I 奖励的评估行为进行校准。

#### （3）离线专有 MLLM 改写（Offline Proprietary-MLLM Rewriting）

可以使用强大的专有 MLLM 离线预计算目标描述。凭借其指令遵循与视觉推理能力，所得描述在人类语义视角下可以准确表达预期的编辑与保留内容。但离线改写使这些条件独立于下游 T2I 奖励而被固定，对奖励模型的选择或后续优化需求缺乏弹性，还引入了对闭源模型的依赖。更根本的是，面向人类读者的语义质量并不保证奖励兼容性：一段准确的描述可能恰好强调该奖励表示薄弱的属性，或落在其偏好的条件分布之外。

综合来看，这些协议指出了有效奖励迁移的两项要求：**目标描述必须忠实编码所请求的修改与应保留的源语义，并且必须与冻结的 T2I 奖励保持兼容**。直接使用编辑指令无法指定完整目标状态，而泛用的 VLM/MLLM 改写无法保证针对特定奖励的对齐。这些观察动机了下文提出的奖励对齐转换机制。

---

## 4 方法（Lever-Edit）

本节介绍 Lever-Edit 的整体流水线与方法设计。Lever-Edit 包含两个阶段：

- **阶段 1** 解决引入 T2I 奖励时的文本表示失配问题：联合优化一个可学习前缀（learnable prefix）与一个固定系统提示（system prompt），在**嵌入层**与**文本层**同时对齐多个 T2I 奖励信号——以显式文本保证执行与一致性信号，以隐藏嵌入保证质量信号。
- **阶段 2** 将学得的对齐融入图像编辑优化：一方面冻结阶段 1 学得的参数，使该模块可作为插件无缝嵌入现有图像编辑微调框架，而不修改其原有训练目标；另一方面，将嵌入级的对齐信号作为辅助监督注入图像编辑轨迹，为平衡提示遵循与编辑一致性提供互补引导。

> **图 2**：Lever-Edit 的整体两阶段 T2I 奖励迁移流水线。

### 4.1 T2I 查询对齐（T2I Query Alignment）

如 图 2 右侧所示，阶段 1 的目标是：**在不更新图像编辑主干、也不更新 VLM 本体的前提下**，使 VLM 生成的目标描述与下游 T2I 奖励模型的偏好对齐。为此，两个模型均被冻结，仅通过前缀微调优化一组可学习的连续查询嵌入（continuous query embeddings）$\mathcal{E}_{\mathrm{query}}$。同时，一个固定的系统提示与原始编辑指令显式拼接，在语义层面约束生成任务。在参考图像 $I_{\mathrm{ref}}$、显式文本指令 $p_{\mathrm{sys}} \oplus q_{\mathrm{edit}}$ 与连续编辑查询 $\mathcal{E}_{\mathrm{query}}$ 的条件下，VLM 自回归地生成目标描述 $c_{\mathrm{tgt}}$。随后由多个互补的 T2I 奖励模型评估该描述，其聚合奖励作为策略梯度信号来优化 $\mathcal{E}_{\mathrm{query}}$。由此，阶段 1 在保持原模型参数不变的同时，将文本生成策略适配向下游 T2I 奖励偏好。

#### 4.1.1 语义保持的嵌入级前缀微调（Semantic-Preserving Embedding-level Prefix Tuning）

奖励驱动的适配应当改善下游 T2I 兼容性，而不改变底层生成任务。给定参考图像 $I_{\mathrm{ref}}$ 及其编辑指令，将固定系统提示 $p_{\mathrm{sys}}$ 与编辑指令显式拼接：

$$
c_{\mathrm{in}} = [p_{\mathrm{sys}} \oplus q_{\mathrm{edit}}], \tag{8}
$$

其中 $\oplus$ 表示沿 token 维度的拼接。系统提示显式指示 VLM 依据 $I_{\mathrm{ref}}$ 与 $q_{\mathrm{edit}}$ 描述"假想的编辑后图像"，因此在整个奖励优化过程中为目标描述任务提供了固定、离散的规约。

除显式文本指令外，引入一个可学习的连续前缀：

$$
\mathcal{E}_{\mathrm{query}} = [\mathcal{E}_1, \ldots, \mathcal{E}_{N_q}] \in \mathbb{R}^{N_q \times d}, \tag{9}
$$

其中 $N_q$ 为可学习查询 token 的数量，$d$ 为 VLM 的隐藏维度。记 $\mathcal{E}_{\mathrm{VLM}}^{text}(\cdot)$ 为 VLM 最终的文本域输入嵌入，则输入构造为：

$$
\mathcal{E}_{VLM}^{text} = \Big[\mathcal{E}_{\mathrm{query}} \,;\, \mathcal{E}_{\mathrm{text}}\big([p_{\mathrm{sys}} \oplus q_{\mathrm{edit}}]\big)\Big]. \tag{10}
$$

于是，$p_{\mathrm{sys}} \oplus q_{\mathrm{edit}}$ 在离散文本空间显式规约生成任务，而 $\mathcal{E}_{\mathrm{query}}$ 在连续嵌入空间提供可训练的自由度。（译注：原文此处两处写作 $q_{\mathrm{edit}}$，结合上下文"仅优化查询嵌入、VLM 参数全程冻结"的表述，应为 $\mathcal{E}_{\mathrm{query}}$，疑系笔误。）在整个阶段 1 中，VLM 的全部原始参数保持冻结，仅优化 $\mathcal{E}_{\mathrm{query}}$。与直接微调 VLM 相比，该设计把奖励驱动的适配限制在一个轻量的前缀空间中，从而限制了对预训练语言与语义能力的干扰，同时仍提供足够灵活性，以捕捉超出固定手工提示之外的、T2I 奖励所偏好的生成模式。

对每个采样的目标描述，采用 $K$ 个互补的 T2I 奖励模型评估其与下游图像生成目标的兼容性，其输出聚合为：

$$
R_{\mathrm{T2I}} = \sum_{k=1}^{K} \omega_k R_k\left(c_{\mathrm{tgt}}; I_{\mathrm{ref}}\right), \tag{11}
$$

其中 $\omega_k$ 为第 $k$ 个奖励模型的权重。优化 $\mathcal{E}_{\mathrm{query}}$ 以最大化期望下游 T2I 奖励：

$$
J_{\mathrm{T2I}}\left(\mathcal{E}_{\mathrm{query}}\right) = \mathbb{E}_{\substack{(I_{\mathrm{ref}}, q_{\mathrm{edit}}) \sim \mathcal{D}, \\ c_{\mathrm{tgt}} \sim \pi_{\mathcal{E}_{\mathrm{query}}}}} R_{\mathrm{T2I}}, \tag{12}
$$

由于 $c_{\mathrm{tgt}}$ 经由离散 token 采样获得，下游奖励无法对生成序列直接求导，因此采用策略梯度（policy gradient）优化 $\mathcal{E}_{\mathrm{query}}$。

#### 4.1.2 离线 token 级分布对齐（Offline Token-level Distribution Alignment）

尽管固定系统提示显式规约了目标描述任务，**纯奖励优化仍可能引发语义漂移（semantic drift）与 token 级失配**。为提供额外的语义锚点，本文使用一个强大的离线 MLLM（如 GPT 或 Gemini）为一小部分训练样本生成高质量参考描述 $c_{\mathrm{tgt}}^{*}$。

具体地，构建离线监督集：

$$
\mathcal{D}_{\mathrm{gold}} = \left\{\left(I_{\mathrm{ref}}^{(i)}, q_{\mathrm{edit}}^{(i)}, c_{\mathrm{tgt}}^{*(i)}\right)\right\}_{i=1}^{N}, \tag{13}
$$

其中 $c_{\mathrm{tgt}}^{*} = (c_1^*, \ldots, c_{L^*}^*)$ 表示由离线 MLLM 生成的参考描述。在解码步 $t$，用经验性 one-hot 目标分布表示参考 token $c_t^*$：

$$
\mathcal{E}_{\mathrm{gold},t}(v) = \mathbb{I}\left[v = c_t^*\right], \qquad v \in \mathcal{V}, \tag{14}
$$

其中 $\mathcal{V}$ 表示 VLM 词表。为紧凑起见，用 $\mathbb{E}_{\mathcal{D}_{\mathrm{gold}}}[\cdot]$ 表示对 $\mathcal{D}_{\mathrm{gold}}$ 中样本的期望。

通过下式将查询条件化的策略正则到该经验语义目标上：

$$
\mathcal{L}_{\mathrm{SFT}} = \mathbb{E}_{\mathcal{D}_{\mathrm{gold}}}\Bigg[\frac{1}{L^*}\sum_{t=1}^{L^*} D_{\mathrm{KL}}\Big(\mathcal{E}_{\mathrm{gold},t} \,\Big\|\, \pi_{\mathcal{E}_{\mathrm{query}}}\big(\cdot \mid I_{\mathrm{ref}},\, p_{\mathrm{sys}} \oplus q_{\mathrm{edit}},\, c_{\mathrm{tgt},<t}^*\big)\Big)\Bigg]. \tag{15}
$$

由于 $\mathcal{E}_{\mathrm{gold},t}$ 是集中于 $c_t^*$ 的 one-hot 分布，该 KL 目标可精确退化为标准 SFT 中使用的 token 级负对数似然（negative log-likelihood）：

$$
\mathcal{L}_{\mathrm{SFT}} = -\,\mathbb{E}_{\mathcal{D}_{\mathrm{gold}}}\Bigg[\frac{1}{L^*}\sum_{t=1}^{L^*} \log \pi_{\mathcal{E}_{\mathrm{query}}}\Big(c_t^* \mid I_{\mathrm{ref}},\, p_{\mathrm{sys}} \oplus q_{\mathrm{edit}},\, c_{\mathrm{tgt},<t}^*\Big)\Bigg]. \tag{16}
$$

两个目标提供互补监督。将下游奖励对齐与离线语义监督结合，阶段 1 的总目标为：

$$
\boxed{\mathcal{L}_{\mathrm{Stage1}} = \mathcal{L}_{\mathrm{PG}} + \lambda_{\mathrm{SFT}} \mathcal{L}_{\mathrm{SFT}}} \tag{17}
$$

其中策略梯度项 $\mathcal{L}_{\mathrm{PG}}$ 在嵌入层提高"获得高下游 T2I 奖励的描述对齐"的可能性；$\mathcal{L}_{\mathrm{SFT}}$ 则将策略正则向离线 MLLM 生成的语义可靠描述靠拢，从而缓解奖励驱动优化带来的语义漂移与潜在的奖励作弊（reward hacking）。

### 4.2 奖励对齐微调（Reward-aligned Finetuning）

阶段 1 冻结图像编辑主干，并在 VLM 生成的目标描述 $c_{\mathrm{tgt}}$ 上进行 RL rollout，产出一组与多个 T2I 奖励直接对齐的可学习查询嵌入 $\mathcal{E}_{\mathrm{query}}$。在阶段 2 中，所得查询嵌入作为一个**即插即用（plug-and-play）的奖励对齐模块**，可跨不同图像编辑主干与微调范式提供有效奖励信号。此外，$\mathcal{E}_{\mathrm{query}}$ 对应的 VLM prefill 嵌入已经同时被视觉与文本输入上下文化，因此可以把这一额外的奖励对齐多模态条件引入图像编辑微调过程，作为辅助监督在编辑轨迹上平衡提示遵循与编辑一致性。

---

## 5 实验

### 5.1 实验设置

**数据集**。评估同时覆盖一个为 T2I 奖励迁移专门构建的、聚焦失败案例的基准，以及现有公开图像编辑基准。具体地，本文构建 **Lever-Bench** 来评估 T2I 奖励向图像编辑的迁移：并非均匀采样常见编辑案例，而是从多个来源精选 **1,880 对**图像-指令对 $(I_{\mathrm{in}}, q_{\mathrm{edit}})$——来源包括 Arena（Jiang et al., 2024）、Counting Edit（Ghosh et al., 2023）与 EditScore-RL-Data（Luo et al., 2026）——并聚焦现有图像编辑模型的失败案例。Lever-Bench 覆盖四大类编辑场景：**局部编辑**（local edits，含物体移除、替换、材质或属性修改）；**以人为中心的编辑**（human-centric edits，含肖像、姿态、动作与面部表情变化）；**全局编辑**（global edits，如背景与风格修改）；以及涉及物体计数与关系推理的**组合编辑**（compositional edits）。为补充这些针对性失败案例并评估超出该基准的泛化能力，另在开源基准 GEdit（Liu et al., 2025b）与 EditReward-Bench（Luo et al., 2026）上汇报结果。

**奖励与指标**。训练奖励与评估指标均选取覆盖图像编辑质量的三个互补维度：感知质量、编辑执行与参考一致性。阶段 1 与阶段 2 均采用 **HPSv3**（Ma et al., 2025）与 **RealGen**（Ye et al., 2025）作为 T2I 奖励：HPSv3 为编辑执行、一致性与美学提供奖励信号，RealGen 强调真实感与一致性。评估层面对主要质量维度分别测评以获得更全面的视图：感知质量从美学与真实感两方面衡量；语义一致性用 **SigLIP**（Zhai et al., 2023）度量编辑图像与其文本条件之间的对应性；另用 **EditScore**（Luo et al., 2026）评估指令执行以及输入图与编辑图之间的一致性。这些指标共同刻画感知质量、指令遵循与源一致性。

**模型与基线**。Lever-Edit 在 **FLUX.1-Kontext-dev**（Labs et al., 2025）上实例化，并应用 **DiffusionNFT**（Zheng et al., 2026）作为编辑微调算法。基线设计旨在区分"奖励对齐的目标描述生成"与"仅仅提供另一种文本条件"的收益。除未对齐的原始图像编辑模型（w/o Alignment）外，比较四种提示构造方式：

1. 直接使用原始编辑指令 $q_{\mathrm{edit}}$（**Lever Pure Edit**）；
2. 由在线 VLM 在固定 $p_{\mathrm{sys}}$ 条件下生成、但不做奖励驱动查询优化的描述（**Lever Pure VLM**）；
3. 由强大闭源 MLLM 离线生成的高质量描述（**Lever Offline SOTA**）；
4. 由 **Lever-Edit** 生成的奖励对齐描述。

除非特别说明，对比各变体使用相同的图像编辑主干、编辑指令集与训练预算。

**主要结果（表 1）**：

**表 1**：不同方法在 Lever-Bench 与 GEdit（Liu et al., 2025b）上的图像编辑评估指标比较（加粗为最优）。

| 方法 | SigLIP ↑（一致性） | HPSv3 ↑（美学） | EditScore 一致性 ↑ | EditScore 自然度 ↑ | EditScore 执行度 ↑ | RealGen v1 ↑（T2I 自然度） | GEdit S ↑ | GEdit Q ↑ | GEdit O ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w/o Alignment | 0.0769 | 5.2742 | 0.8726 | 0.8811 | 0.6311 | 1.1958 | 6.181 | 6.330 | 5.763 |
| Lever Pure Edit | 0.0772 | 5.9823 | 0.9044 | **0.8974** | **0.7307** | 1.2119 | 6.345 | 6.330 | 5.755 |
| Lever Pure VLM | 0.0771 | 7.3152 | 0.9244 | 0.8837 | 0.6249 | 1.1946 | 6.058 | **6.661** | 5.706 |
| Lever Offline SOTA | 0.0763 | 8.2645 | 0.9252 | 0.8896 | 0.7191 | 1.1995 | 6.042 | 6.389 | 5.517 |
| Lever-Edit | **0.0788** | **14.9762** | **0.9263** | 0.8822 | 0.7040 | **1.2532** | **6.364** | 6.362 | **5.788** |

### 5.2 引入 T2I 奖励的益处（The Merits of Introducing T2I Rewards）

为验证核心假设——"编辑质量的标准维度可以映射到 T2I 奖励空间"，本文在两个互补层面对迁移奖励进行评估。**奖励层面**：成功的迁移应能正确区分"更优编辑"与"更差编辑"，即使不存在一个联合处理参考图像、指令与编辑输出的编辑专用评测器。**优化层面**：它应提供稳定的学习信号，其收益能转化为更好的指令执行、语义与源一致性以及图像质量。为此，本文在带真值标注的偏好对上评估奖励准确率、分析训练中的奖励轨迹，并在 Lever-Bench、EditReward-Bench（Luo et al., 2026）与 GEdit（Liu et al., 2025b）上比较下游编辑性能。这些评估共同检验：迁移的 T2I 奖励不仅能给出有意义的偏好分数，其偏好知识还能充当图像编辑 RL 的有效监督。

本文的奖励对齐目标描述使 T2I 奖励能比"直接应用于原始编辑指令"提供更强的编辑偏好信号。在奖励层面，在带真值标注的偏好对上比较三种变体：以 Lever-Edit 生成的奖励对齐目标描述为条件的 HPSv3（**HPSv3+Ours**）、直接以原始编辑指令为条件的 HPSv3（**HPSv3+Edit**）、以及编辑专用奖励 EditScore。如 图 3(a) 所示（在 EditReward-Bench 上评估），HPSv3+Ours 在提示遵循准确率上比 EditScore 高出约 **9 个百分点**；相反，直接以编辑指令为条件的 HPSv3 在一致性准确率上显著下降。原因在于：相对编辑指令只指明"什么该变"，却未显式描述应保留的源内容。在优化层面，训练过程中的奖励曲线（原文称 "Forensic-Chat reward curves"；Forensic-Chat 为 RealGen（Ye et al., 2025）所用的 MLLM 图像检测器，此处即指基于该检测器的 RealGen 真实感奖励曲线）显示 Lever-Edit 在整个训练过程中始终获得高于其他方法的奖励（图 3(b)）。这些结果验证了目标描述接口的核心作用：把相对编辑请求转换为自包含的目标状态，使成熟 T2I 奖励更适合评估图像编辑偏好，进而改进图像编辑模型的微调。

> **图 3**：T2I 奖励迁移的有效性。(a) 在 EditReward-Bench 上评估 T2I 奖励与 I2I 奖励（EditScore）；(b) 不同方法的 T2I 奖励对比，Lever-Edit 展现稳定且最优的优化表现。

### 5.3 执行与一致性的有效性（Effectiveness of Execution and Consistency）

除提升被优化的 T2I 奖励数值外，纯 T2I 监督在编辑感知的核心维度——指令执行与参考一致性——上也带来明确收益。如 表 1 所示，Lever 系列方法在 EditScore 执行度（提示遵循）指标上一致优于未对齐的基础编辑模型，表明迁移奖励为执行所请求编辑提供了有效学习信号；同时 Lever 系列方法在 EditScore 的一致性、自然度与执行度上整体更高。但需要承认，低质量的在线目标提示可能导致更低的执行度分数（如 Lever Pure VLM 的 EditScore 执行度仅 0.6249）。在 GEdit 上，Lever-Edit 取得最佳语义分（S）与最高总分（O）。由于部分奖励模型也参与了优化，本文采用独立的 SigLIP 一致性与 GEdit 作为一致性评估的额外证据：如表 1 所示，Lever-Edit 在所有竞争者中取得最佳的 SigLIP 与 GEdit S 分（一致性）。因此，Lever-Edit 生成的奖励对齐目标描述被证明比原始或泛用文本条件能更有效地表达期望编辑状态的语义。

### 5.4 图像质量的有效性（Effectiveness of Image Quality）

Lever-Edit 在 Lever-Bench 上与迁移的 T2I 质量偏好实现最强对齐。如表 1 所示，它以明显优势取得最高 HPSv3 分数（14.9762，次优为 8.2645），并取得最佳 RealGen 分数（1.2532）。HPSv3 的大幅增益与阶段 1 的目标一致：学得的目标描述显式地架起了编辑任务与 T2I 奖励所表示偏好空间之间的桥梁；RealGen 的相应提升进一步表明所得编辑图像与"真实感+一致性"导向的互补奖励信号对齐良好。

### 5.5 消融实验（Ablation Study）

如 图 5 所示，在不同训练步数上，Lever-Edit 在 HPSv3、SigLIP、EditScore 一致性与 RealGen v1 等指标上一致优于对比方法。为把"描述质量"的影响与"编辑图像质量"的影响解耦，本文使用四种替代文本条件重新计算 SigLIP 相似度：原始编辑指令、原始 VLM 生成的描述（Pure VLM）、离线强模型生成的描述（Offline SOTA）、以及 Lever-Edit 生成的奖励对齐目标描述。重算的 SigLIP 分数列于 表 2，揭示两点明确观察：**其一**，对同一张编辑图像，Lever-Edit 生成的目标描述始终取得最高 SigLIP 分数，表明本文描述与预期编辑内容的语义对齐更好；**其二**，在同一文本描述下，Lever-Edit 生成的图像也取得最高 SigLIP 分数，表明本文编辑图像与预期目标语义更匹配。二者共同证明了目标描述质量与图-文语义对齐的双重提升。

> **图 5**：不同训练步数下的消融研究。Lever-Edit 在绝大多数训练步数上取得最佳的美学、语义对齐、编辑一致性与真实感。

**表 2**：不同 T2I 奖励提示对 SigLIP 语义一致性影响的消融研究（加粗为最优）。

| 方法 \ 评估所用提示 | Pure Edit 提示 | Offline SOTA 提示 | Lever-Edit 提示 |
| --- | --- | --- | --- |
| w/o Alignment | 0.0769 | 0.0768 | 0.0784 |
| Lever Pure Edit | 0.0772 | 0.0766 | 0.0785 |
| Lever Pure VLM | 0.0770 | 0.0765 | 0.0784 |
| Lever Offline SOTA | 0.0767 | 0.0763 | 0.0782 |
| Lever-Edit | **0.0777** | **0.0770** | **0.0788** |

### 5.6 定性分析（Qualitative Analysis）

图 4 从四个互补视角进行定性比较：执行（提示遵循）、美学、一致性与自然度。在提示遵循方面，Lever-Edit 成功地只把狗转换为带大理石纹理的雕像，而基于离线描述的方法失去了与狗原外观的一致性，Pure VLM 方法则错误地保留了与大理石属性相矛盾的原始羽毛纹理。此外，Lever-Edit 将红色跑车放置在正确位置，且渲染质量具有更高的美学与自然感。Lever-Edit 在全局/局部/绘画类编辑中均展现出色的美学表现，在将"冲浪板"与正确物体对齐的同时保持人物头部姿态，体现出高语义与源一致性。最后，在物体移除任务中，Lever-Edit 以更高的自然度质量恢复了被遮挡的树叶。

> **图 4**：不同方法的定性比较。Lever-Edit 在保持语义与源一致性及自然度的同时，以强提示遵循与高美学成功完成图像编辑（放大查看效果更佳）。

---

## 6 结论

本文表明：成熟的 T2I 奖励生态可以被有效迁移到图像编辑 RL，而无需排他性地依赖编辑专用奖励。本文提出 Lever-Edit——一个两阶段框架，通过先学习奖励对齐的目标描述策略、再将其冻结以指导下游图像编辑优化，在相对编辑指令与 T2I 奖励所期望的自包含目标描述之间架起桥梁。在 Lever-Bench 与既有基准上的实验表明：其编辑对齐与源保持能力具有竞争力，同时优于基于原始编辑指令、泛用 VLM 描述与离线高质量提示的直觉迁移基线。这些结果表明：有效的图像编辑 RL 并不一定需要为每个编辑目标单独设计奖励；通过恰当的目标描述接口，T2I 奖励中多样的视觉偏好即可被复用。

---

## 创新点提炼

1. **提出并系统论证"无需编辑奖励的图像编辑在线 RL"这一问题路径**。本文首次从奖励迁移视角重新审视编辑质量三维度（纯图像质量、编辑执行、参考一致性），论证其均可映射到 T2I 奖励空间，从而把"为编辑任务单独构建奖励模型"这一默认前提转变为一个可被绕过的选项。

2. **反事实目标描述接口：$c_{\mathrm{tgt}} = c_{\mathrm{change}} \oplus c_{\mathrm{preserve}}$**。将"改什么"与"保什么"统一编码进一段自包含目标文本，使原本需要图-图对比的编辑执行与参考一致性评估，转化为 T2I 奖励已擅长求解的图-文对齐问题。这是全文最核心的概念贡献，也是"杠杆"得以撬动 T2I 奖励生态的支点。

3. **语义保持的嵌入级前缀微调**。冻结 VLM 与编辑主干，仅训练连续查询前缀 $\mathcal{E}_{\mathrm{query}}$：固定系统提示在离散文本空间锁定任务语义，可学习前缀在连续嵌入空间提供奖励适配自由度。这既避免了全参数微调对预训练语言/语义能力的破坏，又超越了固定手工提示的表达上限，并显式解决了"语义合法的描述未必与冻结奖励兼容"这一被直觉协议忽视的问题。

4. **奖励对齐与离线 token 级分布对齐的双重监督（式 17）**。策略梯度项 $\mathcal{L}_{\mathrm{PG}}$ 最大化多奖励聚合的 T2I 奖励，KL 退化为 token 级 NLL 的 $\mathcal{L}_{\mathrm{SFT}}$ 项则以离线强 MLLM 的金标准描述为锚，有效抑制了纯奖励优化固有的语义漂移与奖励作弊风险。

5. **两阶段解耦、即插即用的模块化架构**。阶段 1 产出的查询嵌入（及其已上下文化的 VLM prefill 嵌入）可作为独立插件，适配不同 VLM、不同图像编辑主干与微调范式（文中实例为 FLUX.1-Kontext-dev + DiffusionNFT）；阶段 2 仅用迁移的 T2I 奖励优化策略，全程无图-图编辑奖励模型，还额外获得嵌入级多模态条件作为平衡提示遵循与编辑一致性的辅助监督。

6. **面向失败案例的 Lever-Bench 基准**。从 Arena、Counting Edit 与 EditScore-RL-Data 精选 1,880 对现有编辑模型的失败样本，覆盖局部、以人为中心、全局与组合四大类编辑场景，为"奖励迁移是否真正有效"提供了针对性、高难度的测试床；配合双层面评估（奖励层面的偏好对准确率 + 优化层面的下游编辑性能），实验设计本身也具有方法论参考价值。

---

*注：本翻译基于 ar5iv（LaTeXML）转换的 Markdown 版本，已忽略站点导航残留与 LaTeX 转义符；参考文献列表从略，正文引用保留 (作者, 年份) 格式。原文中个别疑似笔误（如 $\mathcal{E}_{\mathrm{query}}$ 误作 $q_{\mathrm{edit}}$、"Forensic-Chat reward curves" 等）已在相应位置以译注形式标明。*
