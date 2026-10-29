# 《InstructPix2Pix: Learning to Follow Image Editing Instructions》中文精读翻译

## 论文信息

- **英文标题**：InstructPix2Pix: Learning to Follow Image Editing Instructions
- **中文标题**：InstructPix2Pix：学习遵循图像编辑指令
- **作者**：Tim Brooks*、Aleksander Holynski*、Alexei A. Efros（\* 表示同等贡献）
- **机构**：加州大学伯克利分校（University of California, Berkeley）
- **arXiv 链接**：<https://arxiv.org/abs/2211.09800>（CVPR 2023）
- **提出的框架**：InstructPix2Pix

## 一句话核心总结

InstructPix2Pix 把"指令引导图像编辑"变成一个可规模化的监督学习问题：仅用 700 条人工标注三元组微调 GPT-3，从 LAION 真实图像描述出发生成 45 万余条"编辑指令 + 编辑前后描述"文本三元组，再用 Stable Diffusion 结合 Prompt-to-Prompt 合成对应的前后图像对并经 CLIP 方向相似度过滤，在其上训练一个以"输入图像 + 编辑指令"为双条件的潜在扩散模型，从而首次实现前向传播即完成、零样本泛化到真实图像与人类自然书写指令的快速图像编辑。

---

## 摘要

我们提出一种根据人类指令（human instructions）编辑图像的方法：给定一张输入图像和一段告诉模型"做什么"的书面指令，我们的模型会遵循这些指令对图像进行编辑。为获得该问题的训练数据，我们组合了两个大型预训练模型的知识——一个语言模型（GPT-3）与一个文生图模型（text-to-image model，Stable Diffusion）——来生成一个大规模的图像编辑样本数据集。我们的条件扩散模型 InstructPix2Pix 在这一生成的数据上训练，并在推理时泛化到真实图像与用户书写的指令。由于编辑在前向传播（forward pass）中完成，且不需要逐样本微调（per-example fine-tuning）或反演（inversion），我们的模型可以快速编辑图像，仅需数秒。我们在多样化的输入图像与书面指令集合上展示了令人信服的编辑结果。

---

## 1 引言

我们提出一种教会生成模型遵循人类书写的图像编辑指令的方法。由于该任务的训练数据难以大规模获取，我们提出一种生成配对数据集的方法，组合多个在不同模态上预训练的大型模型：一个大语言模型（GPT-3（Brown et al., 2020））和一个文生图模型（Stable Diffusion（Rombach et al., 2022））。这两个模型捕捉了关于语言与图像的互补知识，可以组合起来为一个横跨两种模态的任务创建配对训练数据。

利用生成的配对数据，我们训练一个条件扩散模型：给定一张输入图像和一段描述"如何编辑它"的文本指令，生成编辑后的图像。我们的模型直接在前向传播中完成图像编辑，不需要任何额外的示例图像、输入/输出图像的完整描述、或逐样本微调。尽管完全在合成样本（即生成的书面指令与生成的图像）上训练，我们的模型对任意**真实**图像与自然的人类书写指令实现了零样本（zero-shot）泛化。我们的模型支持直观的图像编辑，能够遵循人类指令执行多种多样的编辑：替换物体、改变图像风格、改变场景设定、转换艺术媒介等。精选示例见图 1。

> **图 1**：给定一张图像与一条"如何编辑该图像"的指令，我们的模型执行相应的编辑。模型不需要对输入或输出图像的完整描述，且在前向传播中完成编辑，无需逐例反演或微调。

> **图 2**：我们的方法由两部分组成：生成一个图像编辑数据集，以及在该数据集上训练一个扩散模型。(a) 首先使用微调后的 GPT-3 生成编辑指令与编辑后描述；(b) 然后使用 Stable Diffusion（Rombach et al., 2022）结合 Prompt-to-Prompt（Hertz et al., 2022）从成对描述生成成对图像；(c) 我们用该流程创建了一个包含超过 450,000 个训练样本的数据集；(d) 最后，我们的 InstructPix2Pix 扩散模型在生成的数据上训练，以依据指令编辑图像。在推理时，模型能泛化为依据人类书写的指令编辑真实图像。

---

## 2 先前工作

### 2.1 组合大型预训练模型

近期工作表明，大型预训练模型可以被组合起来解决任何单一模型都无法独立完成的多模态任务，例如图像描述（image captioning）与视觉问答（这类任务需要大语言模型与图文模型的双重知识）。组合预训练模型的技术包括：在新任务上联合微调（Alayrac et al., 2022; Mokady et al., 2021; Wang et al., 2021; Li et al., 2019）、通过提示（prompting）进行通信（Zeng et al., 2022; Tiong et al., 2022）、组合基于能量模型的概率分布（Du et al., 2020; Liu et al., 2022）、用一个模型的反馈引导另一个模型（Tewel et al., 2021），以及迭代优化（Li et al., 2022）。我们的方法与先前工作的相似之处在于利用了两个预训练模型（GPT-3 与 Stable Diffusion）的互补能力，不同之处在于我们使用这些模型来**生成配对的多模态训练数据**。

### 2.2 基于扩散的生成模型

扩散模型（diffusion models）（Sohl-Dickstein et al., 2015）的近期进展带来了最先进的图像合成（Song et al., 2019; Ho et al., 2020; Dhariwal et al., 2021; Ho et al., 2022; Saharia et al., 2022a; Saharia et al., 2022b），以及视频（Ho et al., 2022; Singer et al., 2022）、音频（Kong et al., 2020）、文本（Li et al., 2022）和网络参数（Peebles et al., 2022）等其他模态的生成模型。近期的文生图扩散模型（Nichol et al., 2021; Rombach et al., 2022; Ramesh et al., 2022; Saharia et al., 2022c）已经能够从任意文本描述生成逼真的图像。

### 2.3 用于图像编辑的生成模型

图像编辑模型传统上针对单一编辑任务，例如风格迁移（Gatys et al., 2015; Gatys et al., 2016）或图像域之间的转换（Isola et al., 2017; Zhu et al., 2017; Huang et al., 2018; Liu et al., 2019; Ojha et al., 2021）。大量编辑方法将图像反演（Abdal et al., 2019; Abdal et al., 2020; Alaluf et al., 2022; Epstein et al., 2022）或编码（Tov et al., 2021; Richardson et al., 2021; Chai et al., 2021）到某个潜空间（如 StyleGAN（Karras et al., 2019; Karras et al., 2020）），随后通过操纵潜向量进行编辑。近期模型则利用 CLIP（Radford et al., 2021）嵌入以文本引导图像编辑（Kwon et al., 2022; Patashnik et al., 2021; Zheng et al., 2022; Gal et al., 2022b; Nichol et al., 2021; Kim et al., 2022; Avrahami et al., 2022; Crowson et al., 2022）。我们与其中一种方法 Text2Live（Bar-Tal et al., 2022）进行了比较——这是一种优化"颜色 + 不透明度"加性图像层以最大化 CLIP 相似度目标的编辑方法。

近期工作使用预训练的文生图扩散模型进行图像编辑（Meng et al., 2021; Kawar et al., 2022; Avrahami et al., 2022; Ramesh et al., 2022; Hertz et al., 2022）。虽然某些文生图模型原生具备编辑能力（例如 DALLE-2 可以生成图像变体、重绘局部区域、操纵 CLIP 嵌入（Ramesh et al., 2022）），但用这些模型进行**定向**（targeted）编辑并不平凡，因为在多数情况下它们无法保证相似的文本提示会产生相似的图像。Hertz et al.（2022）的近期工作 Prompt-to-Prompt 解决了这一问题：使相似文本提示下的多个生成图像彼此同化（assimilate），从而可以对生成图像进行孤立的编辑。我们在生成训练数据时使用了该方法。为编辑非生成的（即真实的）图像，SDEdit（Meng et al., 2021）使用预训练模型对输入图像加噪，再以新的目标提示去噪。我们将 SDEdit 作为基线进行比较。其他近期工作还包括：给定描述与用户绘制遮罩的局部重绘（Ramesh et al., 2022; Avrahami et al., 2022）；从少量图像中学习特定物体或概念并生成其新图像（Gal et al., 2022a; Ruiz et al., 2022）；通过反演（并微调）单张图像、随后以新的文本描述重新生成来进行编辑（Kawar et al., 2022）。与这些方法相比，我们的模型只需一张图像与一条"如何编辑该图像"的指令（即不是任何图像的完整描述），并在前向传播中直接完成编辑，无需用户绘制的遮罩、额外图像、或逐例反演与微调。

### 2.4 学习遵循指令

我们的方法与现有基于文本的图像编辑工作（Meng et al., 2021; Ruiz et al., 2022; Gal et al., 2022a; Hertz et al., 2022; Kawar et al., 2022; Bar-Tal et al., 2022）的区别在于：它支持从"告诉模型执行什么动作"的**指令**进行编辑，而非文本标签、描述或对输入/输出图像的描述。遵循编辑指令的一个关键好处是：用户可以直接用自然的书面文本告诉模型要做什么，无需提供额外信息，例如示例图像、或对输入与输出图像间保持不变的视觉内容的描述。指令富有表现力、精确且直观易写，使用户能够轻松隔离要更改的特定物体或视觉属性。我们"遵循书面图像编辑指令"的目标受到近期"教大语言模型更好地遵循人类语言任务指令"工作的启发（Ouyang et al., 2022; Mishra et al., 2021; Wei et al., 2021）。

### 2.5 用生成模型生成训练数据

深度模型通常需要大量训练数据。互联网数据集合通常是合适的选择，但可能并不以监督所需的形式存在，例如特定模态的配对数据。随着生成模型持续改进，将它们用作下游任务廉价而充足的训练数据来源的兴趣日益增长（Shrivastava et al., 2017; Ravuri et al., 2019; Viazovetskyi et al., 2020; Tritrong et al., 2021; Li et al., 2022; Peebles et al., 2022）。本文使用两个不同的现成生成模型（语言模型、文生图模型）来为我们的编辑模型产生训练数据。

---

## 3 方法

我们将基于指令的图像编辑（instruction-based image editing）视为一个监督学习问题：(1) 首先，在训练之前生成一个由文本编辑指令与编辑前后图像组成的配对训练数据集（3.1 节，图 2a–c）；(2) 然后，在该生成数据集上训练一个图像编辑扩散模型（3.2 节，图 2d）。尽管使用生成的图像与编辑指令进行训练，我们的模型仍能够泛化到使用任意人类书写指令编辑**真实**图像。方法总览见图 2。

### 3.1 生成多模态训练数据集

我们组合两个在不同模态上运行的大规模预训练模型——一个大语言模型（Brown et al., 2020）与一个文生图模型（Rombach et al., 2022）——来生成一个多模态训练数据集，其中包含文本编辑指令以及编辑前后的对应图像。下面两小节详细描述该流程的两个步骤。3.1.1 节描述微调 GPT-3（Brown et al., 2020）以生成一组文本编辑的过程：给定一个描述图像的提示，产出一条描述"要做的更改"的文本指令，以及一条描述"更改后图像"的提示（图 2a）。随后在 3.1.2 节，我们描述用文生图模型（Rombach et al., 2022）将两个文本提示（即编辑前与编辑后）转换为一对对应图像的过程（图 2b）。

#### 3.1.1 生成指令与成对描述

我们首先完全在文本域内操作：利用一个大语言模型接收图像描述（caption），产出编辑指令与编辑后的文本描述。例如，如图 2a 所示，给定输入描述 *"photograph of a girl riding a horse"*（女孩骑马的照片），我们的语言模型可以同时生成一条合理的编辑指令 *"have her ride a dragon"*（让她骑一条龙）以及恰当修改后的输出描述 *"photograph of a girl riding a dragon"*。在文本域内操作使我们能够生成大规模且多样化的编辑集合，同时保持图像变化与文本指令之间的对应关系。

我们的模型通过在一个相对较小的人类书写"编辑三元组"数据集上微调 GPT-3 训练得到，每个三元组包含：(1) 输入描述、(2) 编辑指令、(3) 输出描述。为构造微调数据集，我们从 LAION-Aesthetics V2 6.5+（Schuhmann et al., 2022）数据集中采样了 700 条输入描述，并人工书写了指令与输出描述。我们书写指令与输出描述的示例见表 1a。使用这些数据，我们以默认训练参数对 GPT-3 Davinci 模型微调了一个 epoch。

得益于 GPT-3 的广博知识与泛化能力，微调后的模型能够生成既有创意又合理的指令与描述。GPT-3 生成数据的示例见表 1b。我们的数据集是通过该训练后的模型生成大量编辑与输出描述而创建的，其中输入描述是来自 LAION-Aesthetics 的真实图像描述（排除了描述重复或图像 URL 重复的样本）。我们选择 LAION 数据集是因为它规模巨大、内容多样（包括专有名词与流行文化引用）、媒介丰富（照片、绘画、数字艺术作品）。LAION 的一个潜在缺点是噪声较大，包含不少无意义或无描述性的 caption——但我们发现，数据集噪声可以通过数据过滤（3.1.2 节）与无分类器引导（3.2.1 节）的组合得到缓解。我们最终生成的指令与描述语料由 **454,445** 个样本组成。

| | 输入 LAION 描述 | 编辑指令 | 编辑后描述 |
| --- | --- | --- | --- |
| **人工书写（700 条编辑）** | Yefim Volkov, Misty Morning | make it afternoon（改成下午） | Yefim Volkov, Misty Afternoon |
| | girl with horse at sunset | change the background to a city（把背景换成城市） | girl with horse at sunset in front of city |
| | painting-of-forest-and-pond | Without the water.（去掉水） | painting-of-forest |
| | … | … | … |
| **GPT-3 生成（>450,000 条编辑）** | Alex Hill, Original oil painting on canvas, Moonlight Bay | in the style of a coloring book（换成涂色书风格） | Alex Hill, Original coloring book illustration, Moonlight Bay |
| | The great elf city of Rivendell, sitting atop a waterfall as cascades of water spill around it | Add a giant red dragon（加一条巨型红龙） | The great elf city of Rivendell, sitting atop a waterfall as cascades of water spill around it with a giant red dragon flying overhead |
| | Kate Hudson arriving at the Golden Globes 2015 | make her look like a zombie（把她变成僵尸） | Zombie Kate Hudson arriving at the Golden Globes 2015 |
| | … | … | … |

> **表 1**：我们标注一个小的文本数据集、微调 GPT-3，并用该微调模型生成一个大规模的文本三元组数据集。对于标注样本与生成样本，输入描述均使用来自 LAION 的真实图像描述。高亮文本（译注：原表中为指令与编辑后描述两列）由 GPT-3 生成。

#### 3.1.2 从成对描述生成成对图像

> **图 3**：使用 Stable Diffusion（Rombach et al., 2022）在有/无 Prompt-to-Prompt（Hertz et al., 2022）下生成的图像对。(a) 无 Prompt-to-Prompt；(b) 有 Prompt-to-Prompt。两种情况下对应的描述均为 "photograph of a girl riding a horse" 与 "photograph of a girl riding a dragon"。

接下来，我们使用一个预训练的文生图模型，将一对描述（分别指编辑前与编辑后的图像）转换为一对图像。将一对描述转换为一对对应图像的挑战之一在于：文生图模型不保证图像之间的 consistency（一致性），即使条件提示的变化非常微小。例如，两个非常相似的提示 "a picture of a cat" 与 "a picture of a black cat" 可能产生截然不同的猫的图像。这不适合我们的用途——我们打算将配对数据用作"训练模型编辑图像"的监督信号（而非产生另一张随机图像）。因此我们使用 Prompt-to-Prompt（Hertz et al., 2022），一个旨在促使文生图扩散模型的多次生成彼此相似的新方法。它通过在若干去噪步中借用交叉注意力（cross attention）权重来实现。图 3 展示了有无 Prompt-to-Prompt 时采样图像的对比。

虽然这极大地帮助了生成图像的同化，但不同的编辑可能需要图像空间中不同程度的改变。例如，幅度更大的更改——如改变大尺度图像结构的更改（移动物体、替换为形状不同的物体等）——可能要求生成图像对之间的相似度更低。幸运的是，Prompt-to-Prompt 有一个可以控制两图相似度的参数：共享注意力权重的去噪步比例 $p$。不幸的是，仅从描述与编辑文本出发确定最优的 $p$ 值是困难的。因此，我们对每个描述对生成 100 个样本图像对，每个使用随机的 $p\sim\mathcal{U}(0.1,0.9)$，并用一个基于 CLIP 的度量来过滤这些样本：Gal et al.（2022b）提出的 CLIP 空间中的方向相似度（directional similarity）。该度量衡量两张图像之间的变化（在 CLIP 空间中）与两条图像描述之间的变化的一致程度。执行这一过滤不仅有助于最大化图像对的多样性与质量，也使我们的数据生成对 Prompt-to-Prompt 与 Stable Diffusion 的失败更加鲁棒。

### 3.2 InstructPix2Pix

我们使用生成的训练数据来训练一个条件扩散模型，使其根据书面指令编辑图像。我们的模型基于 Stable Diffusion——一个大规模文生图潜在扩散模型（latent diffusion model）。

扩散模型（Sohl-Dickstein et al., 2015）通过一系列去噪自编码器学习生成数据样本，这些自编码器估计数据分布的 score（Hyvärinen et al., 2005）（一个指向更高数据密度区域的方向）。潜在扩散（Rombach et al., 2022）通过在一个预训练变分自编码器（variational autoencoder）（Kingma et al., 2013）（编码器 $\mathcal{E}$、解码器 $\mathcal{D}$）的潜空间中运行，提升了扩散模型的效率与质量。对于图像 $x$，扩散过程对编码后的潜变量 $z=\mathcal{E}(x)$ 加噪，产生带噪潜变量 $z_{t}$，其噪声水平随时间步 $t\in T$ 递增。我们学习一个网络 $\epsilon_{\theta}$，在图像条件 $c_{I}$ 与文本指令条件 $c_{T}$ 下，预测添加到带噪潜变量 $z_{t}$ 上的噪声。我们最小化如下潜在扩散目标：

$$
L=\mathbb{E}_{\mathcal{E}(x),\mathcal{E}(c_{I}),c_{T},\epsilon\sim\mathcal{N}(0,1),t}\Big[\|\epsilon-\epsilon_{\theta}(z_{t},t,\mathcal{E}(c_{I}),c_{T}))\|_{2}^{2}\Big] \tag{1}
$$

Wang et al.（2022）表明，对图像翻译任务而言，微调一个大型图像扩散模型优于从头训练一个模型，尤其在配对训练数据有限时。因此，我们用一个预训练的 Stable Diffusion checkpoint 初始化模型权重，以利用其庞大的文生图生成能力。为支持图像条件化，我们在第一个卷积层添加了额外的输入通道，将 $z_{t}$ 与 $\mathcal{E}(c_{I})$ 拼接在一起。扩散模型所有可用权重均从预训练 checkpoint 初始化，而作用于新增输入通道的权重初始化为零。我们复用原本为 caption 设计的同一文本条件机制，改为将文本编辑指令 $c_{T}$ 作为其输入。更多训练细节见补充材料（附录 A.3）。

#### 3.2.1 双条件无分类器引导

> **图 4**：两个条件输入上的无分类器引导权重。$s_{I}$ 控制与输入图像的相似度，$s_{T}$ 控制与编辑指令的一致性。矩阵中每个单元格对应一对 $(s_{I}, s_{T})$ 取值下的编辑结果（编辑指令："Turn him into a cyborg!"，把他变成赛博格）。行对应 $s_{I}=1.0/1.2/1.6$，列对应 $s_{T}=3/7.5/15$。

无分类器扩散引导（classifier-free diffusion guidance）（Ho et al., 2022）是一种权衡扩散模型生成样本质量与多样性的方法。它常用于类条件与文本条件的图像生成，以提升生成图像的视觉质量，并使采样图像更好地对应其条件。无分类器引导有效地将概率质量移向"隐式分类器 $p_{\theta}(c|z_{t})$ 给条件 $c$ 赋予高似然"的数据。其实现方式为：联合训练扩散模型以进行条件与无条件去噪，并在推理时组合两个 score 估计。无条件去噪的训练只需在训练期间以一定频率将条件设为固定的空值 $c=\varnothing$。推理时，在引导尺度 $s\ge 1$ 下，修正后的 score 估计 $\tilde{e_{\theta}}(z_{t},c)$ 朝条件项 $e_{\theta}(z_{t},c)$ 方向外推、并远离无条件项 $e_{\theta}(z_{t},\varnothing)$：

$$
\tilde{e_{\theta}}(z_{t},c)=e_{\theta}(z_{t},\varnothing)+s\cdot(e_{\theta}(z_{t},c)-e_{\theta}(z_{t},\varnothing)) \tag{2}
$$

对我们的任务而言，score 网络 $e_{\theta}(z_{t},c_{I},c_{T})$ 有两个条件：输入图像 $c_{I}$ 与文本指令 $c_{T}$。我们发现对两个条件均利用无分类器引导是有益的（译注：原文 "We find if beneficial" 疑为 "We find it beneficial" 之笔误）。Liu et al.（2022）证明条件扩散模型可以组合来自多个不同条件值的 score 估计。我们将同一概念应用于模型的两个独立条件输入。训练时，我们对 5% 的样本仅设 $c_{I}=\varnothing_{I}$，对 5% 的样本仅设 $c_{T}=\varnothing_{T}$，对 5% 的样本同时设 $c_{I}=\varnothing_{I}$ 且 $c_{T}=\varnothing_{T}$。因此，我们的模型能够对两个（或其中任一）条件输入进行条件化或无条件去噪。我们引入两个引导尺度 $s_{I}$ 与 $s_{T}$，可用于权衡"生成样本与输入图像对应的强度"以及"生成样本与编辑指令对应的强度"。我们的修正 score 估计如下：

$$
\begin{split}
\tilde{e_{\theta}}(z_{t},c_{I},c_{T})=&\;e_{\theta}(z_{t},\varnothing,\varnothing)\\
&+s_{I}\cdot(e_{\theta}(z_{t},c_{I},\varnothing)-e_{\theta}(z_{t},\varnothing,\varnothing))\\
&+s_{T}\cdot(e_{\theta}(z_{t},c_{I},c_{T})-e_{\theta}(z_{t},c_{I},\varnothing))
\end{split} \tag{3}
$$

图 4 展示了这两个参数对生成样本的影响。无分类器引导公式化的细节见附录 B。

---

## 4 结果

我们在多样化的真实照片与艺术作品上展示了基于指令的图像编辑结果，覆盖多种编辑类型与指令措辞（精选结果见图 1、5、6、7、11、12、15–19）。我们的模型成功完成了许多富有挑战性的编辑，包括替换物体、改变季节与天气、替换背景、修改材质属性、转换艺术媒介等多种其他编辑。

> **图 5**：《蒙娜丽莎》被转换为多种艺术媒介（"Make it a Modigliani painting" / "Make it a Miro painting" / "Make it an Egyptian sculpture" / "Make it a marble roman sculpture"）。

> **图 6**：《创造亚当》被赋予新的语境与主体（"Put them in outer space" / "Turn the humans into robots"；以 768 分辨率生成）。

> **图 7**：披头士标志性的《Abbey Road》专辑封面以多种方式转换（"Make it Paris" / "Make it Hong Kong" / "Make it Manhattan" / "Make it Prague" / "Make it evening" / "Put them on roller skates" / "Turn this into 1900s" / "Make it underwater" / "Make it Minecraft" / "Turn this into the space age" / "Make them into Alexander Calder sculptures" / "Make it a Claymation"）。

我们将本方法与两项近期工作 SDEdit（Meng et al., 2021）和 Text2Live（Bar-Tal et al., 2022）进行定性比较。我们的模型遵循"如何编辑图像"的指令，但先前工作（包括这两个基线方法）期望的是对图像（或编辑层）的描述，因此我们向它们提供"编辑后"文本描述而非编辑指令。我们还使用两个分别衡量图像一致性与编辑质量的指标与 SDEdit 进行定量比较（详见 4.1 节）。最后，我们在 4.2 节展示关于"生成训练数据的规模与质量如何影响模型性能"的消融研究。

### 4.1 基线比较

我们提供与 SDEdit（Meng et al., 2021）和 Text2Live（Bar-Tal et al., 2022）的定性比较，以及与 SDEdit 的定量比较。SDEdit 是一种用预训练扩散模型编辑图像的技术：将一张部分加噪的图像作为输入并去噪，产生一张新的编辑后图像。我们比较的是其公开的 Stable Diffusion 实现。Text2Live 是一种通过生成"颜色 + 不透明度"增强层（以文本提示为条件）来编辑图像的技术，我们比较的是作者发布的公开实现。

我们在图 9 中与两种方法进行定性比较。我们注意到：当内容大致不变、只改变风格时，SDEdit 效果尚可；但在期望更大改动时，它难以保持身份（identity）并隔离单个物体。此外，它需要期望图像的完整输出描述，而非一条编辑指令。另一方面，Text2Live 虽然能在涉及加性层的编辑上产生可信的结果，但其形式化限制了它能处理的编辑类别。

> **图 9**：与其他编辑方法的比较。输入图像或经编辑字符串（最后两列）、或经真实输出图像描述（中间两列）转换。我们将本方法与两项近期工作 SDEdit（Meng et al., 2021）和 Text2Live（Bar-Tal et al., 2022）比较。SDEdit 展示两种配置：以输出描述为条件（SDEdit-OC）与以编辑字符串为条件（SDEdit-E）。（译注：正文 4.1 节将前者写作 "OP"、图 9 中写作 "OC"，二者均指以输出描述为条件的配置，此处统一采用图 9 的记法。）

与 SDEdit 的定量比较见图 8。我们绘制了两个指标之间的权衡：CLIP 图像嵌入的余弦相似度（编辑后图像与输入图像的吻合程度）与 Gal et al.（2022b）提出的方向 CLIP 相似度（文本描述的变化与图像变化的吻合程度）。这两个指标是相互竞争的——输出图像越符合期望的编辑，其与输入图像的相似度（一致性）就越低。尽管如此，我们发现与 SDEdit 相比，在相同方向相似度取值下，我们的结果具有显著更高的图像一致性（CLIP 图像相似度）。

> **图 8**：我们绘制了与输入图像的一致性（Y 轴）和与编辑的一致性（X 轴）之间的权衡。两个指标均为越高越好。两种方法均固定文本引导为 7.5，并将我们的 $s_{I}$ 在 $[1.0,2.2]$ 内变化、SDEdit 的 strength（去噪量）在 $[0.3,0.9]$ 内变化。

> **图 11**：以不同指令循环应用我们的模型，产生复合（compound）编辑。

> **图 12**：通过改变潜在噪声，我们的模型可以为同一输入图像与指令产生多种可能的图像编辑（编辑指令："in a race car video game"）。

> **图 13**：失败案例。从左到右：我们的模型无法执行视角变化（"Zoom into the image"）；可能对图像做出不希望的过度更改（"Move it to Mars"）；有时无法隔离指定的物体（"Color the tie blue"）；难以重组或互换物体的位置（"Have the people swap places"）。

### 4.2 消融

在图 10 中，我们针对数据集规模的选择与 3.1 节所述的数据过滤方法提供了定量消融。我们发现：减小数据集规模通常会导致执行更大幅度（更显著）图像编辑的能力下降——模型只执行细微或风格化的图像调整（因此保持较高的图像相似度分数，但方向分数较低）。从数据生成中移除 CLIP 过滤则产生不同的影响：与输入图像的整体图像一致性降低。

> **图 10**：通过固定 $s_{T}$、扫描 $s_{I}\in[1.0,2.2]$，比较模型的消融变体（更小训练集、无 CLIP 过滤）。我们提出的配置表现最佳。

我们还在图 4 中分析了两个无分类器引导尺度的影响。增大 $s_{T}$ 会对图像施加更强的编辑（输出更符合指令）；增大 $s_{I}$ 则有助于保持输入图像的空间结构（输出更符合输入图像）。我们发现，$s_{T}$ 在 5–10 范围内、$s_{I}$ 在 1–1.5 范围内通常产生最佳结果。在实践中（包括论文中展示的结果），我们发现针对每个样本调整引导权重、以获得一致性与编辑强度之间的最佳平衡是有益的。

---

## 5 讨论与结论

我们展示了一种组合两个大型预训练模型（一个大语言模型与一个文生图模型）来生成数据集、进而训练扩散模型遵循书面图像编辑指令的方法。尽管我们的方法能够对图像产生风格、媒介及其他语境层面的丰富编辑，仍存在若干局限。

我们的模型受生成数据集视觉质量的限制，因此受用于生成图像的扩散模型（此处为 Stable Diffusion（Rombach et al., 2022））的限制。此外，本方法泛化到新编辑、并在视觉变化与文本指令之间建立正确关联的能力，受限于用于微调 GPT-3（Brown et al., 2020）的人工书写指令、GPT-3 创建指令与修改描述的能力，以及 Prompt-to-Prompt（Hertz et al., 2022）修改生成图像的能力。特别地，与 Stable Diffusion 和 Prompt-to-Prompt 一样，我们的模型在物体计数与空间推理上表现挣扎（例如 *"move it to the left of the image"*、*"swap their positions"*、*"put two cups on the table and one on the chair"*）。失败示例见图 13。此外，本方法所基于的数据与预训练模型存在有据可查的偏见（biases），因此我们的方法产生的编辑图像可能继承这些偏见或引入其他偏见（图 14：例如职业与性别之间的相关性）。

> **图 14**：我们的方法反映了其所基于的数据与模型的偏见，例如职业与性别之间的相关性（输入 / "Make them look like flight attendants" / "Make them look like doctors"）。

除缓解上述局限外，我们的工作也引出了一些开放问题：如何遵循空间推理指令、如何将指令与用户交互等其他条件模态结合、以及如何评估基于指令的图像编辑。结合人类反馈来改进模型是未来工作的另一个重要方向，人机回圈强化学习（human-in-the-loop reinforcement learning）等策略可用于改进模型与人类意图之间的对齐。

**致谢**：我们感谢 Ilija Radosavovic、William Peebles、Allan Jabri、Dave Epstein、Kfir Aberman、Amanda Buster 与 David Salesin。Tim Brooks 受美国国家科学基金会（NSF）研究生研究奖学金资助。额外资助由 SAP 的研究资助与 Google 的捐赠提供。

> **图 15–19**（讨论部分附图，定性结果摘录）：莱顿的《花园中的女士》迁移到新场景（"Make it a grocery store"）；梵高《戴草帽的自画像》呈现于不同媒介（"Convert to a realistic photo" / "Turn into a 3D model"）；一张风景照片的语境化编辑——值得注意的是，孤立更改也带来伴随的语境效应：添加小船同时在水面引入风浪涟漪，添加的城市天际线也倒映在湖面上（"Add boats on the water" / "Replace the mountains with a city skyline"）；一张城市照片被编辑为一天中的不同时刻（"It is now midnight" / "Add a beautiful sunset"）；维米尔《戴珍珠耳环的少女》的多种编辑（"Apply face paint" / "What would she look like as a bearded man?" / "Put on a pair of sunglasses" / "She should look 100 years old" / "What if she were in an anime?" / "Make her terrifying" / "Make her more sad" / "Make her James Bond" / "Turn her into Dwayne The Rock Johnson"）。

---

## 附录要点（压缩概括）

1. **GPT-3 微调与采样格式（附录 A.1）**：微调时，文本提示为输入描述拼接 "\n##\n" 分隔符；文本补全为指令与编辑后描述的拼接，二者以 "\n%%\n" 分隔，末尾追加 "\nEND" 作为停止符。推理时对新输入描述采样文本补全，使用 temperature=0.7、frequency_penalty=0.1；排除输入与输出描述相同的生成结果。

2. **配对图像生成设置（附录 A.2）**：使用 Stable Diffusion v1.5 checkpoint 的指数滑动平均（EMA）权重与改进的 ft-MSE 自编码器权重；以 100 步去噪、Euler ancestral 采样器及 Karras et al.（2022）提出的去噪方差调度生成图像；确保每个生成图像对中的两张图使用相同的潜在噪声（初始噪声与随机采样中引入的噪声均一致）。

3. **对 Prompt-to-Prompt 的适配改动**：原始 Prompt-to-Prompt 依据具体编辑类型（词交换、添加短语、增减词语权重）以不同方式替换第二张图的交叉注意力权重；本文则改为在前 $p$ 比例的步中替换第二张图像的**自注意力（self-attention）**权重，并对所有编辑统一使用同一注意力权重替换策略。

4. **数据过滤阈值**：每对描述生成 100 对图像（译注：原文 "We generation 100 pairs" 疑为 "We generate" 之笔误）；过滤条件包括：图像-图像 CLIP 相似度阈值 0.75（保证两图不过于不同）、图像-描述 CLIP 相似度阈值 0.2（保证图像与其描述对应）、方向 CLIP 相似度阈值 0.2（保证前后描述的变化与前后图像的变化对应）。对通过全部过滤的图像对，按方向 CLIP 相似度排序，每个描述对最多保留 4 个样本。

5. **训练配置（附录 A.3）**：在 8 张 40GB NVIDIA A100 GPU 上训练 10,000 步、耗时 25.5 小时；训练分辨率 256×256，总 batch size 为 1024；采用随机水平翻转与裁剪增广（先将图像随机缩放到 256–288 像素之间、再裁剪到 256）；学习率 $10^{-4}$（无 warm up）；模型从 Stable Diffusion v1.5 checkpoint 的 EMA 权重初始化，其余训练设置沿用公开 Stable Diffusion 代码库。

6. **推理与速度**：虽然模型在 256×256 分辨率下训练，我们发现它在推理时对 512×512 分辨率泛化良好；论文中的结果均以 512 分辨率、100 步去噪（Euler ancestral 采样器 + Karras et al. 方差调度）生成。在 A100 GPU 上，用本模型编辑一张图像约需 **9 秒**。

7. **双条件 CFG 的概率推导（附录 B）**：生成模型学习 $P(z|c_{I},c_{T})$（在输入图像与文本指令条件下图像潜变量 $z=\mathcal{E}(x)$ 的分布）。将条件概率按链式法则分解：$P(z|c_{T},c_{I})=\frac{P(z,c_{T},c_{I})}{P(c_{T},c_{I})}=\frac{P(c_{T}|c_{I},z)P(c_{I}|z)P(z)}{P(c_{T},c_{I})}$。取对数得 $\log(P(z|c_{T},c_{I}))=\log(P(c_{T}|c_{I},z))+\log(P(c_{I}|z))+\log(P(z))-\log(P(c_{T},c_{I}))$；对 $z$ 求梯度（即取 score）后得 $\nabla_{z}\log(P(z|c_{T},c_{I}))=\nabla_{z}\log(P(z))+\nabla_{z}\log(P(c_{I}|z))+\nabla_{z}\log(P(c_{T}|c_{I},z))$，与式 (3) 中的三项一一对应：$s_{I}$ 将概率质量移向隐式分类器 $p_{\theta}(c_{I}|z_{t})$ 给图像条件赋予高似然的数据，$s_{T}$ 对应隐式分类器 $p_{\theta}(c_{T}|c_{I},z_{t})$。这些隐式分类器由"有无相应条件输入的估计之差"学得。存在其他可行的分解方式（如交换 $c_{T}$ 与 $c_{I}$ 的位置），但作者发现该分解在实践中效果更好。

---

## 创新点提炼

1. **开辟"指令引导图像编辑"这一任务范式**。本文首次将图像编辑的条件从"输入/输出图像的完整描述"转变为一条自然语言**编辑指令**（直接告诉模型做什么，如 "swap the car with a truck"），用户无需示例图像、遮罩、完整描述或逐例微调，一句话即可完成编辑；指令富有表现力、精确且直观，这一接口定义奠定了后续指令编辑研究的基调。

2. **以"大模型知识组合"自动合成多模态配对训练数据**。仅用 700 条人工标注的三元组微调 GPT-3 Davinci，即可从 LAION 真实图像描述出发生成 454,445 条"（输入描述，编辑指令，编辑后描述）"文本三元组，一举破解该任务"配对数据无法从互联网直接获得"的根本瓶颈；这种"组合生成模型造训练数据"的思路影响深远。

3. **从文本对到图像对的可靠合成流水线**。用 Stable Diffusion + Prompt-to-Prompt（共享注意力权重）保证前后图像大体一致；针对"最优一致性参数 $p$ 无法从文本预知"的问题，对每个描述对随机采样 100 个 $p\sim\mathcal{U}(0.1,0.9)$ 的图像对，再用 CLIP 图像相似度、图像-描述相似度与方向 CLIP 相似度三重阈值过滤并按方向相似度择优保留，兼顾数据多样性、质量与对上游模型失败的鲁棒性。

4. **双条件潜在扩散架构 InstructPix2Pix**。在 Stable Diffusion 基础上新增图像条件通路：将 $\mathcal{E}(c_{I})$ 与 $z_{t}$ 拼接后送入第一层卷积（新增通道权重零初始化），并复用原文本条件机制输入编辑指令；从预训练 checkpoint 初始化并利用"微调优于从头训练"的结论，使全部合成数据仅需 8×A100 上 25.5 小时、1 万步、256 分辨率的训练即得到强泛化编辑器。

5. **双条件无分类器引导及其概率论解释**。将 CFG 从单条件扩展到（图像，指令）双条件：训练时按 5%/5%/5% 随机丢弃图像条件、指令条件或双条件；推理时通过式 (3) 的三项分解得到两个可独立调节的引导尺度 $s_{I}$、$s_{T}$，实现"保持输入图 vs 执行编辑强度"的连续事后调控；附录 B 进一步说明该分解恰对应条件概率的链式分解与两个隐式分类器，为架构选择给出理论依据。

6. **零样本泛化与快速编辑的完整实证**。完全在合成数据（生成指令 + 生成图像）上训练，却能零样本泛化到任意真实图像与人类自然书写指令；编辑在前向传播中完成，无需反演或逐例微调，A100 上约 9 秒/图；定量上在"CLIP 图像相似度–方向 CLIP 相似度"权衡曲线上，相同编辑强度下显著优于 SDEdit，并通过消融验证了数据规模与 CLIP 过滤的必要性。

---

*注：本翻译基于 ar5iv/arXiv HTML（LaTeXML）转换的 Markdown 版本，已忽略站点导航残留与 LaTeX 转义符；参考文献列表从略，正文引用保留 (作者, 年份) 格式。原文中个别疑似笔误（如 "We find if beneficial"、"We generation 100 pairs"、SDEdit 配置缩写在正文与图 9 中分别为 OP/OC）已在相应位置以译注形式标明。*
