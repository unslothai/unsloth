<h1 align="center" style="margin:0;">
  <a href="https://unsloth.ai/docs"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png">
    <img alt="Unsloth 标志" src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png" height="80" style="max-width:100%;">
  </picture></a>
</h1>
<h3 align="center" style="margin: 0; margin-top: 0;">
Unsloth 是首款支持本地运行与训练大模型的桌面应用。
</h3>

<p align="center">
  <a href="#-功能特性">功能特性</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-免费-notebooks">免费 Notebooks</a> •
  <a href="https://unsloth.ai/docs">官方文档</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <a href="https://unsloth.ai/docs/desktop">
    <img height="400" alt="unsloth desktop" src="https://unsloth.ai/cgi/image/unsloth_qwen3.8_final_ut2eqWnYJ-SLmu0s7x522.png?format=raw" />
  </a>
</p>

## ⚡ 快速开始
下载适用于您操作系统的 Unsloth Desktop 原生桌面客户端：
<table>
  <tr>
    <td><b>操作系统平台</b></td>
    <td><b>下载链接</b></td>
  </tr>
  <tr>
    <td><b>Windows</b></td>
    <td><a href='https://github.com/unslothai/unsloth/releases/latest/download/Unsloth-Desktop-Windows.exe'>立即下载 (.exe)</a></td>
  </tr>
  <tr>
    <td><b>macOS</b></td>
    <td><a href='https://github.com/unslothai/unsloth/releases/latest/download/Unsloth-Desktop-MacOS.dmg'>立即下载 (.dmg)</a></td>
  </tr>
  <tr>
    <td><b>Linux / Ubuntu (deb)</b></td>
    <td><a href='https://github.com/unslothai/unsloth/releases/latest/download/Unsloth-Desktop-Ubuntu.deb'>立即下载 (.deb)</a></td>
  </tr>
  <tr>
    <td><b>Linux (AppImage)</b></td>
    <td><a href='https://github.com/unslothai/unsloth/releases/latest/download/Unsloth-Desktop-Linux.AppImage'>立即下载 (.AppImage)</a></td>
  </tr>
</table>

也可从 [Unsloth 官网](https://unsloth.ai/download) 或 [GitHub Releases 发布页](https://github.com/unslothai/unsloth/releases) 获取。

若偏好通过脚本一键安装：

#### macOS, Linux, WSL:
```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```
#### Windows:
```powershell
irm https://unsloth.ai/install.ps1 | iex
```
#### 社区链接：

- [Discord 社区](https://discord.gg/unsloth)
- [𝕏 (Twitter)](https://x.com/UnslothAI)
- [Reddit 社区](https://reddit.com/r/unsloth)

## ⭐ 功能特性
Unsloth 完美支持 **Windows、Linux、WSL** 与 **macOS**。全面兼容 **多 GPU 集群、NVIDIA、AMD、Intel 显卡、CPU** 以及 **Vulkan** 加速后端。

### AI 运行与开发生态
* **运行与训练全模态模型：** 支持大语言模型、扩散生成、向量嵌入与音频模型：[Qwen3.8](https://unsloth.ai/docs/models/qwen3.8)、[Kimi K3](https://unsloth.ai/docs/models/kimi-k3)、MiniMax-H3、[Muse Glimmer](https://unsloth.ai/docs/models/muse-glimmer)、[DeepSeek-V4](https://unsloth.ai/docs/models/deepseek-v4)、[Gemma 4](https://unsloth.ai/docs/models/gemma-4)。
* **智能体与工具调用（Agents & Tools）：** 将本地模型无缝接入 [Claude Code](https://unsloth.ai/docs/basics/claude-code)、[Codex](https://unsloth.ai/docs/basics/codex) 与 [MCP (模型上下文协议)](https://unsloth.ai/docs/basics/mcp)，支持工具调用与代码沙箱执行。
* **搜索与 RAG 增强：** 具备隐私无限制的网页搜索、深度调研（Deep Research）、滑动上下文自动压缩（Auto-compaction）与 RAG 检索增强。
* **图像与视频多模态：** 运行与训练[图像生成](https://unsloth.ai/docs/basics/diffusion-image)、视频扩散模型或视觉多模态大模型。
* **远程与局域网访问：** 通过[局域网 (LAN)](https://unsloth.ai/docs/basics/lan) 在任意设备访问本地模型，或通过安全的 [Cloudflare HTTPS 隧道](https://unsloth.ai/docs/basics/how-to-serve-local-llms-anywhere-secure-remote-access-with-cloudflare-and-unsloth) 进行全局远程访问。
* **统一接入接口：** 通过兼容 [OpenAI API](https://unsloth.ai/docs/basics/api) 的格式提供模型推理服务，支持绑定 ChatGPT / Codex 订阅及[主流云端供应商](https://unsloth.ai/docs/integrations/connections)。

### 模型训练与部署
* **超快微调（Fine-tuning）：** 训练 LLM、Diffusion、TTS 和 Embedding 模型速度提升 2 倍，显存占用降低 70%，且[完全无精度损失](https://unsloth.ai/blog#training)。
* **全套对齐技术：** 全面支持[强化学习 (RL)](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)、LoRA、QLoRA、全量微调（Full Fine-Tuning）、预训练、GRPO、DPO 与 FP8 精度。
* **导出与量化部署：** 一键[导出](https://unsloth.ai/docs/new/studio/export)或部署模型，支持 [GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)、NVFP4、FP8 等主流工业级量化格式。
* **数据集构建：** 通过 [Data Recipes](https://unsloth.ai/docs/new/studio/data-recipe) 从 PDF、CSV、DOCX 等非结构化文档一键解析生成训练数据集。
  
## 🚀 Unsloth Start 一键连接智能体

[Unsloth Start](https://unsloth.ai/docs/integrations/unsloth-start) 仅需一行命令即可将 [Claude Code](https://unsloth.ai/docs/basics/claude-code)、[Codex](https://unsloth.ai/docs/basics/codex) 及各类智能体直接接入本地大模型：

```bash
unsloth start claude --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

| 目标智能体 | 启动命令 |
| --- | --- |
| Claude Code | `unsloth start claude` |
| OpenAI Codex | `unsloth start codex` |
| Hermes Agent | `unsloth start hermes` |
| OpenClaw | `unsloth start openclaw` |
| OpenCode | `unsloth start opencode` |

## 📥 安装方式
Unsloth 提供三种使用方式：**[Unsloth Desktop](https://unsloth.ai/download)**（桌面客户端）、**[Unsloth Studio](https://unsloth.ai/docs/new/studio/)**（Web UI 界面）以及 **Unsloth Core**（Python 代码库版本）。

### 1. Unsloth Desktop 桌面版（推荐）

<table>
  <tr>
    <td><b>操作系统平台</b></td>
    <td><b>下载安装包</b></td>
  </tr>
  <tr>
    <td><b>Windows</b></td>
    <td><a href='https://github.com/unslothai/unsloth/releases/latest/download/Unsloth-Desktop-Windows.exe'>下载 Windows 安装包 (.exe)</a></td>
  </tr>
  <tr>
    <td><b>macOS</b></td>
    <td><a href='https://github.com/unslothai/unsloth/releases/latest/download/Unsloth-Desktop-MacOS.dmg'>下载 macOS 镜像 (.dmg)</a></td>
  </tr>
  <tr>
    <td><b>Linux / Ubuntu (deb)</b></td>
    <td><a href='https://github.com/unslothai/unsloth/releases/latest/download/Unsloth-Desktop-Ubuntu.deb'>下载 Ubuntu 安装包 (.deb)</a></td>
  </tr>
  <tr>
    <td><b>Linux (AppImage)</b></td>
    <td><a href='https://github.com/unslothai/unsloth/releases/latest/download/Unsloth-Desktop-Linux.AppImage'>下载 Linux AppImage</a></td>
  </tr>
</table>

### 2. Unsloth Studio（Web UI 界面版）

#### macOS, Linux, WSL 安装：
```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

#### Windows (PowerShell) 安装：
```powershell
irm https://unsloth.ai/install.ps1 | iex
```

#### 启动 Web 界面：
```bash
unsloth studio
```

#### 安全 HTTPS 部署：
```bash
unsloth studio --secure
```

#### Docker 容器运行：
使用官方 [Docker 镜像](https://hub.docker.com/r/unsloth/unsloth) ```unsloth/unsloth```：
```bash
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 8000:8000 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
```

### 3. Unsloth Core（Python 代码开发版）
#### Linux, WSL:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```
#### Windows:
```powershell
winget install -e --id Python.Python.3.13
winget install --id=astral-sh.uv -e
uv venv unsloth_env --python 3.13
.\unsloth_env\Scripts\activate
uv pip install unsloth --torch-backend=auto
```

#### AMD、Intel、DGX Spark 与 Blackwell 架构支持：
请参阅 [Blackwell RTX 50 系列指南](https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth) 与 [DGX Spark 指南](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth)。在 **AMD** 和 **Intel** 显卡上的配置请参考 [AMD 指南](https://unsloth.ai/docs/basics/amd) 与 [Intel 指南](https://unsloth.ai/docs/get-started/install/intel)。

## 📒 免费 Notebooks 体验

使用官方预配置的 Colab Notebooks 免费开启模型训练：

| 目标模型 | 免费 Notebook 体验 | 训练速度提升 | 显存节省 |
|-----------|---------|--------|----------|
| **Unsloth Studio**      | [▶️ 免费启动](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb)               | - | - |
| **Gemma 4 (E2B)**      | [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Vision.ipynb)               | 1.5倍更快 | 节省 50% |
| **Qwen3.5 (4B)**      | [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb)               | 1.5倍更快 | 节省 60% |
| **gpt-oss (20B)**      | [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)               | 2倍更快 | 节省 70% |
| **Qwen3.5 GSPO**      | [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision_GRPO.ipynb)               | 2倍更快 | 节省 60% |
| **gpt-oss (20B): GRPO** | [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)               | 2倍更快 | 节省 80% |
| **Qwen3: Advanced GRPO**| [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)               | 2倍更快 | 节省 70% |
| **embeddinggemma (300M)**| [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma_(300M).ipynb)               | 2倍更快 | 节省 20% |
| **Llama 3.1 (8B) Alpaca** | [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)               | 2倍更快 | 节省 70% |
| **Llama 3.2 对话微调** | [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb) | 2倍更快 | 节省 70% |
| **Orpheus-TTS (3B)**    | [▶️ 免费启动](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)               | 1.5倍更快 | 节省 50% |

- 浏览全部专用 Notebook：[Kaggle Notebooks](https://github.com/unslothai/notebooks?tab=readme-ov-file#-kaggle-notebooks)、[GRPO 强化推理](https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks)、[TTS 语音](https://unsloth.ai/docs/get-started/unsloth-notebooks#text-to-speech-tts-notebooks)、[Embedding 向量微调](https://unsloth.ai/docs/new/embedding-finetuning) 与 [Vision 视觉多模态](https://unsloth.ai/docs/get-started/unsloth-notebooks#vision-multimodal-notebooks)
- 查阅[全量支持模型目录](https://unsloth.ai/docs/get-started/unsloth-model-catalog) 与 [全量 Notebooks 清单](https://unsloth.ai/docs/get-started/unsloth-notebooks)

## 🦥 最新动态

- **AMD 全流程支持**：在 Windows、WSL 与 Linux 上全面支持 AMD GPU 进行模型训练、强化学习（RL）、对话及部署。[指南](https://unsloth.ai/docs/basics/amd)
- **通用智能体本地模型**：通过 `unsloth start` 将 Claude Code、Codex、Hermes、OpenCode 等智能体直连 Unsloth 提供的 OpenAI / Anthropic 兼容 API。[指南](https://unsloth.ai/docs/basics/api)
- **GLM-5.2 超大模型**：通过 Unsloth Dynamic GGUFs 在本地运行智谱 744B 参数、100 万长上下文开源模型。[指南](https://unsloth.ai/docs/models/glm-5.2)
- **DeepSeek-V4 支持**：在本地流畅运行 DeepSeek-V4-Flash，具备修正后的多轮对话与工具调用行为。[指南](https://unsloth.ai/docs/models/deepseek-v4)
- **Gemma 4 支持**：支持运行与训练 Gemma 4 文本、图像和音频模型，具备 QAT、MTP、GGUF 与 MLX 深度优化。[指南](https://unsloth.ai/docs/models/gemma-4)
- **MCP 服务器集成**：通过模型上下文协议（Model Context Protocol）将本地模型连接至文件系统、应用程序、数据库与外部工具。[指南](https://unsloth.ai/docs/basics/mcp)

## 💚 社区与生态链接

| 资源类别 | 访问链接 |
| --- | --- |
| <img width="16" src="https://cdn.prod.website-files.com/6257adef93867e50d84d30e2/66e3d80db9971f10a9757c99_Symbol.svg" /> **Discord** | [加入 Discord 开发者社区](https://discord.gg/unsloth) |
| <img width="15" src="https://redditinc.com/hs-fs/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" /> **Reddit** | [加入 r/unsloth 社区](https://reddit.com/r/unsloth) |
| 📚 **官方文档与 Wiki** | [查阅官方文档](https://unsloth.ai/docs) |
| <img width="13" src="https://upload.wikimedia.org/wikipedia/commons/0/09/X_(formerly_Twitter)_logo_late_2025.svg" /> **Twitter (X)** | [关注 @unslothai](https://twitter.com/unslothai) |
| 🔮 **模型仓库** | [Unsloth 模型目录](https://unsloth.ai/docs/get-started/unsloth-model-catalog) |
| ✍️ **技术博客** | [阅读技术博客](https://unsloth.ai/blog) |

### 引用 Unsloth

若在学术研究或项目中使用了 Unsloth，请使用以下格式进行引用：

```bibtex
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {https://github.com/unslothai/unsloth},
  year = {2023}
}
```

### 开源许可证 (License)

Unsloth 采用 **Apache 2.0** 与 **AGPL-3.0** 双重许可证模式：Unsloth Core 核心算法库基于 **[Apache 2.0](https://github.com/unslothai/unsloth?tab=Apache-2.0-1-ov-file)** 许可证发布；部分扩展组件（如 Unsloth Studio UI 界面）基于 **[AGPL-3.0](https://github.com/unslothai/unsloth?tab=AGPL-3.0-2-ov-file)** 许可证发布。

### 致谢 (Acknowledgments)

- 感谢 [llama.cpp 库](https://github.com/ggml-org/llama.cpp) 赋能 Unsloth 本地模型推理与 GGUF 格式保存
- 感谢 Hugging Face 团队及其 [transformers](https://github.com/huggingface/transformers) 与 [TRL](https://github.com/huggingface/trl) 生态库
- 感谢 PyTorch 与 [Torch AO](https://github.com/unslothai/unsloth/pull/3391) 团队的贡献
- 感谢 NVIDIA 及其 [NeMo DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) 库的贡献
- 感谢每一位为 Unsloth 提交代码、建议与日常使用 Unsloth 的开发者！
---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年8月31日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
