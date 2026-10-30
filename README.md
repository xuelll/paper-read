# 视觉生成与奖励模型论文精读 · 中英对照阅读器

> 62 篇视觉生成与奖励模型论文（2021–2026）的中文精读翻译与中英对照阅读站点。

**[📖 在线阅读](https://xuelll.github.io/paper-read/)** —— 支持桌面与手机浏览

## 这是什么

一个自建的论文精读库站点：每篇论文一个页面，左侧为 arXiv 官方 HTML 原文（公式原生渲染），右侧为深度中文精读翻译（KaTeX 公式 + 表格），两栏滚动同步、章节互跳；窄屏（手机）自动切换为上下布局。

- **覆盖方向**：图像编辑奖励、T2I 奖励谱系（ImageReward / HPS / PickScore 等演化）、量规化评估生态、原生统一生成架构、视觉工具智能体（CodeV / DeepEyes / Beacon 等）
- **配套汇总**：[创新点总结（62 篇）](https://xuelll.github.io/paper-read/web/summary.html) —— 论文清单与主题分类、逐篇核心创新、跨论文共性主题、研究趋势与跨论文创新机会
- 站点为纯静态页面，由 `web/build_web.ps1` 从本地源文件构建，托管于 GitHub Pages

## 仓库结构

| 目录 / 文件 | 内容 |
|---|---|
| `web/` | 站点本体：62 个论文页 + 首页 + 创新点总结页 + arXiv 原文 HTML（`web/en_src/`） |
| `translations/` | 62 篇中文精读翻译 Markdown |
| `创新点总结.md` | 跨论文创新点汇总（总结页的源文件） |
| `web/build_web.ps1` | 站点构建脚本 |
| `index.html` | 仓库根跳转页（重定向到 `web/index.html`） |

## 阅读方式

- **桌面端**：双栏对照阅读，顶栏可开关滚动同步；点击任一侧章节标题，另一侧自动跳转到对应章节
- **手机端**：自动切换上下布局；超宽表格自动缩放并在卡片内横向滑动
- **公式**：中文侧由 KaTeX（CDN）渲染，英文侧为 arXiv 原生 MathML；离线或 CDN 不可达时回退为 LaTeX 原文显示
- **英文原文****：来自 arXiv 官方 HTML 版本；个别论文无 HTML 版本时提供 arXiv 跳转链接

## 论文清单

62 篇论文的完整清单（含发布年月、arXiv 号、机构、主题分类）见[创新点总结 · 第一部分](https://xuelll.github.io/paper-read/web/summary.html)。

---

*本项目为个人学习笔记整理，论文原文版权归原作者所有；英文原文来自 arXiv 官方 HTML 版本。*
