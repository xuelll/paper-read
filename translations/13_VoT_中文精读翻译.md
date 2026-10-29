# 13. VoT: Vision-of-Thought for Unified Multimodal Representation Alignment
# （VoT：面向统一多模态表示对齐的"视觉思维"框架）——中文精读翻译

- **英文标题**: VoT: Vision-of-Thought for Unified Multimodal Representation Alignment
- **中文标题**: VoT：面向统一多模态表示对齐的视觉思维（Vision-of-Thought）
- **作者与机构**: Jingxiang Sun、Chao Liao（共同一作）、Zhengxiong Luo、Chaorui Deng、Chen-lin Zhang、Junke Wang、Ceyuan Yang、Haoqi Fan、Weilin Huang —— ByteDance Seed
- **arXiv**: https://arxiv.org/abs/2609.07815（2026年9月）
- **一句话总结**: 在 VLM 与扩散解码器之间插入一层**离散的"视觉思维"token**，让 VLM 先"想清楚"要画什么（生成对象与布局的显式视觉计划），再交给扩散分支渲染，从而把 VLM 的推理能力迁移到生成中，GenEval 达到 0.91，超过 BAGEL、Mogao 等统一模型。

---

## 摘要

当前的文生图（T2I）系统普遍采用"文本编码器 + 扩散解码器"范式：文本语义直接调制连续的隐噪声。这类方法虽然成功，但缺乏一个**显式、可解释的中间表示**来弥合高层语言语义与低层视觉信号。本文提出 **Vision-of-Thought（VoT）** 框架，在视觉语言模型（VLM）与扩散 Transformer（DiT）之间引入一个**离散的视觉思维层**。与把 VLM 仅当作文本编码器不同，VoT 让 VLM 充当**多模态规划器**：在渲染像素之前，先生成表示高层视觉计划（如物体、布局）的离散 VoT token。我们训练了一个专门的 VoT tokenizer，它在 VLM 语义空间中通过**闭环目标**学习，该目标由三部分组成：VLM 对齐损失、特征重建损失和向量量化损失。这些目标使 token 既能被 VLM"读懂"，又保留了生成所需的视觉信息。实验表明，VoT 提升了语义对齐能力，并为可解释、可控的生成提供了结构化接口。

## 1. 引言

**背景与动机**。能同时做深度理解与高保真生成的统一模型是 AI 的重要方向（GPT-4o 的亮相进一步激发了这一兴趣）。但设计统一框架极具挑战：需要把自回归模型（擅长离散推理与文本生成）的优势与扩散模型（擅长高质量图像合成）的稳健性结合起来。

现有 T2I 范式把它当作直接映射问题：把文本 prompt 编码为静态 embedding 或 KV cache 去条件化扩散过程。这在简单描述上有效，但存在显著的**模态鸿沟**：LLM 中蕴含的丰富、结构化的世界知识，很难仅通过静态 embedding 有效传递给扩散解码器——模型"知道"要生成什么，却"画不准"，形成瓶颈。

**VoT 的核心思想**是作为对齐两种模态的语义桥梁。采用"并行专家"哲学：冻结一个强大的 VLM 以保持其多模态理解能力，同时引入轻量的自回归专家来生成离散的语义视觉 token。与近期的 query 类方法（MetaQuery、BLIP-3o）生成并行查询不同，VoT **自回归地**生成视觉 token。这一选择是刻意为之：query 类方法有并行效率，但缺乏与 VLM 复杂推理所需的结构一致性。把视觉生成当作 next-token prediction 任务，VoT 让模型执行"视觉思维链"——在渲染前一步步规划图像构图，把 VLM 的推理能力延伸到视觉领域。

**关键障碍：视觉 tokenization**。自回归图像生成的核心难题是视觉 tokenizer。标准视觉 tokenizer（VQ-VAE、VQGAN）主要优化重建保真度，把图像压缩成由局部纹理和低层细节主导的"像素编码"。其码本与 VLM 的语义表示空间**根本错位**：token 序列往往过长且语义冗余，更像压缩的像素碎片而非有意义的语义单元；而且这些以重建为中心的 token 的统计特性（重尾使用分布、相邻 token 语义相关性弱）不利于 next-token prediction，使 AR 建模不稳定。

**VoT Tokenizer** 通过**闭环对齐**范式解决这一错位：不是量化原始像素，而是做**语义量化**——用预训练 VLM 的冻结 ViT 特征作为量化目标，确保离散 token 落在高层语义流形上。优化目标有三个：(i) **VLM 对齐损失**使 VLM 能"读"并使用量化后的 token；(ii) **特征重建损失**恢复教师 ViT 特征空间；(iii) **向量量化损失**稳定码本学习。前两个目标高度互补：VLM 对齐提供语言接地的监督，重建提供稠密、保信息的约束。此外观察到**扩大教师 VLM 规模**能持续改善 tokenizer 质量与下游语义对齐——更强的教师提供更干净、更具组合性的监督。

**贡献**：
1. **VoT 框架**：在 VLM 与扩散解码器之间插入离散视觉思维分支，解耦语义推理与像素渲染，提升跨模态语义对齐；
2. **VLM 对齐的 VoT tokenizer**：量化 VLM 语义特征而非像素，闭环目标结合三种损失；证明 VLM 对齐与重建互补、扩大教师规模有益；
3. **高效的统一 MoT 集成与强结果**：通过克隆 VLM 专家初始化 VoT 分支、冻结 VLM 分支，配合两阶段模型训练，在语义对齐与推理驱动的 T2I 上取得大幅提升。

## 2. 相关工作（要点式）

- **统一多模态理解与生成**：统一自回归模型（Show-o、Janus、Unified-IO）把图像当离散 token 序列在单一 Transformer 里做两个任务，但因"面向重建的视觉编码"与"面向语义的语言建模"之间的张力，往往在生成保真或理解深度上有折损；基于扩散的统一方案（把理解模块嫁接到扩散骨干上）则受静态条件传递的模态鸿沟之苦。VoT 的差异在于插入显式、可解释的中间层（离散视觉思维），保留 VLM 与 DiT 各自专长。
- **视觉 tokenization 策略**：VQ-VAE/VQGAN 确立了"重建目标 + 离散压缩"范式；LFQ、MaskGIT 改进码本效率与生成多样性，但本质上仍是重建中心。SEED、LaVIT 把视觉 token 对齐到语言模型 embedding，但属事后对齐（post-hoc）；Emu 为 VLM 生成的 embedding 引入视觉解码器但用连续表示。共同缺陷：缺少既能保视觉信息、又能让 VLM 语义可读的闭环训练。VoT tokenizer 用统一目标补上这一缺口。
- **视觉生成中的推理与规划**：DALL-E 3、Imagen 靠精细文本编码提升 prompt 遵循，但无显式中间推理；CoT 机制多用于理解而非生成。LayoutGPT 用 LLM 做布局规划，但规划与生成分两阶段。VoT 用统一词表把规划**嵌入生成过程**：范式从"文本→图像"变为"文本→思维→图像"（text-to-thought-to-image）。

## 3. VoT 总览

框架在 VLM 与扩散解码器之间插入离散视觉思维层：VLM 先产出结构化 VoT token 作为显式视觉计划，再由统一模型中的扩散分支渲染。整体分三个阶段：

- **Stage 0：VLM 对齐的 VoT tokenizer 训练**——在 VLM 语义空间中学习离散码本（特征重建 + VLM 对齐 + 向量量化三损失）；
- **Stage 1：VoT 分支预训练**——冻结预训练 VLM 与扩散分支，训练轻量 VoT 专家，从文本 prompt 预测视觉 token（目标来自配对图像）；
- **Stage 2：VoT-DiT 联合训练**——联合微调 VoT 与扩散分支，VLM 分支保持冻结，以保留多模态理解并降低训练成本。

## 4. VLM 对齐的 VoT Tokenizer 训练

传统 tokenizer 优先像素保真；VoT tokenizer 通过专用架构与统一对齐目标，**最小化视觉 token 与 VLM 语义空间的分布偏移**。

### 4.1 语义对齐的 tokenizer 架构

- **基础编码器 + 适配器（Foundation & Adapter Encoder）**：目标 VLM（Qwen2.5-VL）的 ViT 记为 $\mathcal{E}_{\text{frozen}}$，冻结以保留其预对齐的鲁棒语义特征；在其后接一个轻量可训练适配器 $\mathcal{E}_{\text{adapt}}$（1 层 ViT block），将连续 ViT 特征投影到适合量化的流形：
  $$\mathbf{h}=\mathcal{E}_{\text{adapt}}(\mathcal{E}_{\text{frozen}}(\mathbf{x})).$$
- **SimVQ 头**：使用基于相似度的向量量化（SimVQ）机制，可学习码本 $\mathcal{C}=\{c_k\}_{k=1}^{K}\subset\mathbb{R}^d$。对特征向量 $\mathbf{h}_i$ 按欧氏距离最近邻选择离散索引：
  $$k=\mathop{\arg\min}_{j\in\{1,\dots,K\}}\|\mathbf{h}_i-c_j\|_2^2.$$
  为防码本坍缩，用熵损失正则化相似度 logits $s_{i,j}=-\|\mathbf{h}_i-c_j\|_2^2$；量化特征经直通估计器（straight-through estimator）反传梯度：$\mathbf{z}_{st}=\mathbf{h}+\text{sg}[\mathbf{z}_q-\mathbf{h}]$。
- **深层语义解码器**：解码器 $\mathcal{D}$ 从量化 token 重建连续语义特征 $\hat{\mathbf{f}}=\mathcal{D}(\mathbf{z}_{st})$。不同于常见的轻量设计，这里用 **6 层** Transformer 解码器。消融显示解码器从 2 层加深到 6 层持续提升 AutoEval（69.32→70.37）。原因：更强的解码器增强了"在教师特征空间中**逆量化**"的能力，减轻码本保存细粒度细节的压力，让 token 专注于语言接地的语义，解码器则平滑量化噪声。

### 4.2 训练目标

训练中教师 ViT 与 VLM 均冻结，共三个互补目标：

1. **特征重建损失（视觉接地）**：教师 ViT 产生 token 特征 $\mathbf{f}=\mathcal{E}_{\text{frozen}}(\mathbf{x})\in\mathbb{R}^{N\times d}$，经量化与直通估计得 $\mathbf{z}_{st}$，解码器重建 $\hat{\mathbf{f}}=\mathcal{D}(\mathbf{z}_{st})$，用逐 token 的 MSE：
   $$\mathcal{L}_{rec}=\frac{1}{N}\|\mathbf{f}-\hat{\mathbf{f}}\|_2^2.$$
2. **VLM 对齐损失（语义可读性）**：把 tokenizer 产生的视觉 embedding $\mathbf{v}$（如 $\mathbf{v}=\mathbf{z}_{st}$ 或其投影）喂入冻结 VLM，对配对 caption $\mathbf{y}$ 最小化标准语言建模交叉熵：
   $$\mathcal{L}_{align}=-\sum_{t=1}^{T}\log P_{\text{VLM}}\left(y_t\mid y_{<t},\mathbf{v}\right).$$
3. **向量量化损失（稳定量化）**：标准 VQ 目标（码本损失 + 承诺损失）：
   $$\mathcal{L}_{VQ}=\|\text{sg}[\mathbf{h}]-\mathbf{z}_q\|_2^2+\beta\|\mathbf{h}-\text{sg}[\mathbf{z}_q]\|_2^2.$$

**总目标**：
$$\mathcal{L}_{tok}=\mathcal{L}_{rec}+\lambda_{align}\mathcal{L}_{align}+\lambda_{VQ}\mathcal{L}_{VQ}.$$

## 5. 带 VoT 的统一视觉语言模型

### 5.1 三分支 MoT 骨干

把 Mixture-of-Transformers（MoT）扩展为**三分支**设计：文本（txt）、VoT、扩散（diff），各分支有独立的 LayerNorm 与 Q/K/V 投影：
$$\mathbf{Q}_m=\mathbf{W}_m^Q\cdot\text{LN}_m(\mathbf{h}_m^{(\ell)}),\quad \mathbf{K}_m=\mathbf{W}_m^K\cdot\text{LN}_m(\mathbf{h}_m^{(\ell)}),\quad \mathbf{V}_m=\mathbf{W}_m^V\cdot\text{LN}_m(\mathbf{h}_m^{(\ell)}).$$

跨模态信息交换通过**拼接所有分支的 KV 构造全局上下文**：
$$\mathbf{K}=[\mathbf{K}_{\text{txt}};\mathbf{K}_{\text{vot}};\mathbf{K}_{\text{diff}}],\qquad \mathbf{V}=[\mathbf{V}_{\text{txt}};\mathbf{V}_{\text{vot}};\mathbf{V}_{\text{diff}}],$$
统一注意力在联合空间上运作，并由分支专属掩码 $\mathbf{M}_m$ 调制：
$$\text{Attn}_m=\text{softmax}\left(\frac{\mathbf{Q}_m\mathbf{K}^{\top}}{\sqrt{d_k}}+\mathbf{M}_m\right)\mathbf{V}.$$

文本与 VoT 分支用**因果掩码**（支持自回归生成），扩散分支用**双向注意力**。输出经分支专属 FFN 得到下一层隐状态。

**实现要点**：VoT 通过逐层 VoT 专家 + 码本尺寸的 embedding/预测接口接入；VoT 专家通过**复制一部分预训练 VLM 专家**初始化，VLM 分支全程冻结。推理时：先以文本为条件自回归生成 VoT token，再由扩散分支以"文本 + VoT 的 KV cache"为条件渲染最终图像。

### 5.2 训练配方

- **Stage 1：VoT 分支预训练**。骨干从 **Mogao**（14B MoT 统一模型）初始化，加载 VLM 与扩散分支的预训练权重；VoT 专家复制自 VLM 专家；仅随机初始化 VoT 专属组件（std 0.0125）：视觉 token embedding 表 $\mathbf{E}_{\mathrm{vot}}\in\mathbb{R}^{65,536\times 2,048}$ 与 VoT 预测头。用 Stage 0 的冻结 tokenizer 提取离散监督目标。训练目标为 65,536 类码本上的自回归交叉熵，每图预测 1,024 个 VoT token。
- **Stage 2：VoT-DiT 联合训练**。从 Stage 1 检查点出发，联合更新 VoT 分支与扩散分支（含 VoT 专家、扩散专家与共享组件），联合优化"VoT 自回归预测损失 + 扩散速度预测损失"，让扩散损失经共享注意力层反传，收紧 token 预测与渲染质量的耦合。为支持推理时的 CFG，训练时做结构化条件丢弃：VoT token 以 0.5 概率丢弃、有 VoT 时文本以 0.1 概率丢弃，得到约 50% 纯文本、45% 文本+VoT、5% 纯 VoT 样本；另外随机屏蔽 0–20% 的 VoT KV，防止扩散分支过度依赖 VoT 并提高对不完整条件的鲁棒性。

## 6. 实验

### 6.1 实现细节

- **数据**：Stage 0 用图文对（原生长宽比，约 434×434 至 1024×1024，总像素 188K–1M），最大序列 16,384 token，单样本上限 12,284 token；Stage 1/2 用大规模 T2I 数据集。
- **模型**：视觉 tokenizer 用 Qwen2.5-VL-7B 的 ViT 作冻结视觉编码器；SimVQ 量化器含 1 层编码器、65,536 项码本（$d=128$）、6 层解码器，学习率 3e-4。统一模型从 Mogao-14B 初始化，新增 7B 参数 VoT 分支，**总计 21B**。
- **算力预算**（附录A）：预训练（tokenizer + 统一模型）360,000 GPU 时（75.0%）、继续训练（CT）50,000（10.4%）、SFT 64,000（13.3%）、RLHF 6,000（1.3%），合计 **480,000 GPU 时**。

### 6.2 文生图生成

在 GenEval（553 个 prompt、每 prompt 生成 4 图，用目标检测器度量物体类别/数量/颜色/空间关系）上：

| 类型 | 模型 | 单物体 | 双物体 | 计数 | 颜色 | 位置 | 颜色属性 | 总分 |
|---|---|---|---|---|---|---|---|---|
| 生成专用 | FLUX.1-dev | 0.98 | 0.93 | 0.75 | 0.93 | 0.68 | 0.65 | 0.82 |
| 统一模型 | Janus-Pro-7B | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| 统一模型 | BAGEL | 0.99 | 0.94 | 0.81 | 0.88 | 0.64 | 0.63 | 0.82 |
| 统一模型 | Mogao | 1.00 | 0.97 | 0.83 | 0.93 | 0.84 | 0.80 | 0.89 |
| **统一模型** | **VoT（本文）** | **1.00** | **0.98** | **0.86** | **0.95** | **0.85** | **0.82** | **0.91** |

VoT 总分 **0.91**，超过生成专用模型（FLUX.1-dev 0.82）与统一模型（BAGEL 0.82、Mogao 0.89），在计数等组合性要求高的任务上优势明显（0.86 vs 0.83）。这支持了核心假设：显式离散 VoT 层把 VLM 的规划与推理能力传递到生成过程，帮助满足隐式条件化方法常失败的结构性约束。在内部指令遵循基准上，VoT 超过 Seedream 4.0、GPT-Image-1 等基线。定性对比中，VoT 能准确渲染细粒度属性（如表情）与反事实场景（"马骑宇航员"），而 GPT-Image-1 常出现属性泄漏或依赖训练分布偏差。

### 6.3 消融研究

六个配置（A–F）逐步验证 tokenizer 设计：

| 实验 | 教师 VLM | 编码/解码深度 | $\mathcal{L}_{align}$ | $\mathcal{L}_{rec}$ | AutoEval |
|---|---|---|---|---|---|
| Mogao（基线） | – | – | – | – | 53.12 |
| A | Qwen2-VL 2B | –/– | ✓ | – | 58.67 |
| B | Qwen2-VL 2B | –/2 | ✓ | ✓ | 60.98 |
| C | Qwen2-VL 2B | 1/2 | ✓ | ✓ | 64.64 |
| D | Qwen2.5-VL 7B | 1/2 | ✓ | ✓ | 69.32 |
| E（VoT） | Qwen2.5-VL 7B | 1/6 | ✓ | ✓ | 70.37 |
| F（VoT，SFT+RLHF后） | Qwen2.5-VL 7B | 1/6 | ✓ | ✓ | **76.22** |

主要结论：
- **重建与适配器的作用**：加入重建损失后 VLM 语言损失从 ~1.93 降至 ~1.79，再加 1 层适配器降至 ~1.76；AutoEval 相应 58.67→60.98→64.64。
- **AR 可预测性 ≠ 有用性**：Exp.A 的自回归损失最低（~5.08），但 tokenizer 语言损失最高（~1.93）、AutoEval 只有 58.67——容易被预测的 token 不等于对下游有用；重建与适配器在保留更多信息、略微提高 AR 难度的同时，切实改善语义对齐与下游效用。
- **教师规模效应**：把教师从 Qwen2-VL 2B 换成 Qwen2.5-VL 7B，语言损失降至 ~1.48、AutoEval 69.32，且在含重建的配置中 AR 损失反而最低——**更强的教师给出更干净的语义监督，诱导出更结构化、对自回归更友好的视觉词表**。
- **VoT 与后训练的收益**：相同 T2I 训练数据下，加入 VoT token 使 AutoEval 从 53.12 升至 70.37；SFT+RLHF 后进一步升至 76.22，说明后训练与 VoT 互补。

## 7. 结论

VoT 通过在 VLM 与扩散解码器之间插入离散视觉规划层，把语义理解与视觉合成桥接起来：利用 VLM 在渲染前生成结构化 VoT token 作为显式计划，成功地把高层推理与像素级生成解耦，显著提升跨模态对齐与可解释性。作者希望 VoT 成为具备连贯视觉思维与创作能力的下一代多模态智能体的奠基一步。

**局限性（附录B）**：MoT 架构使用冻结 VLM 骨干 + 并行专家，系统仅支持"推理→生成"的**单向迁移**；该设计保留了 VLM 既有能力，但未来将探索双向优化策略（在不灾难性遗忘的前提下同时提升理解与生成）。

---

## 创新点提炼

1. **"文本→思维→图像"的显式中间表示**：首次在统一多模态模型中于 VLM 与扩散分支之间插入离散"视觉思维"token 层，把 VLM 从"文本编码器"升级为"多模态规划器"，在渲染像素前显式规划物体与布局，解决了静态 embedding 条件化导致的"知道却画不准"瓶颈，并带来天然的可解释性与可控性。
2. **语义量化而非像素量化**：VoT tokenizer 用冻结 VLM 的 ViT 语义特征作为量化目标（而非重建像素），通过"VLM 对齐 + 特征重建 + 向量量化"闭环三目标，让离散 token 既被 VLM 语义可读、又保留生成所需视觉信息，纠正了传统 VQ tokenizer"像素碎片"式 token 与语言语义空间的错位。
3. **教师规模定律**：实验证明扩大教师 VLM（2B→7B）持续改善 token 质量与下游语义对齐——更强教师提供更干净、更组合化的监督，诱导出对自回归建模更友好的视觉词表。
4. **高效的三分支 MoT 集成**：把 MoT 扩展为文本/VoT/扩散三分支，VoT 专家由复制预训练 VLM 专家初始化、VLM 分支全程冻结，21B 总参数下仅需约 6 万 GPU 时即可完成新增分支训练，兼顾稳定性与效率。
5. **强实证效果**：GenEval 总分 0.91（超 BAGEL/Mogao 0.89 与 FLUX.1-dev 0.82），消融确认每个组件（重建、适配器、深层解码器、VoT token、后训练）的独立贡献，其中"AR 损失最低≠下游最优"的发现对 tokenizer 设计有普遍指导意义。
