// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocaleCode } from "./messages";

const TRANSLATABLE_ATTRIBUTES = [
  "title",
  "placeholder",
  "aria-label",
  "aria-description",
] as const;

const EXACT_TEXT_ZH_CN: Record<string, string> = {
  "Configuration": "配置",
  "Close configuration": "关闭配置",
  "Chat inference settings": "聊天推理设置",
  "Preset name": "预设名称",
  "Choose a preset": "选择预设",
  "Open preset list": "打开预设列表",
  "Edit System Prompt": "编辑系统提示词",
  "Prompt editor": "提示词编辑器",
  "This prompt is part of the current configuration and saves with the preset.": "此提示词属于当前配置，并会随预设一起保存。",
  "Use this for longer edits. Save writes back to the active configuration only.": "用于较长内容编辑。保存仅会写回当前配置。",
  "You are a helpful assistant...": "你是一个乐于助人的助手...",
  "Max Tool Calls Per Message": "每条消息最大工具调用次数",
  "Max Tool Call Duration": "单次工具调用最大时长",
  "Data Recipes": "数据配方",
  "Create and manage local recipe workflows.": "创建并管理本地数据配方流程。",
  "No recipes yet": "暂无配方",
  "Browse Learning Recipes below to understand how recipe workflows work.": "浏览下方学习配方，了解配方工作流的使用方式。",
  "Learning Recipes": "学习配方",
  "Start from a prebuilt recipe to learn patterns, then edit and run.": "从预构建配方开始学习流程模式，然后编辑并运行。",
  "Instruction from Answer": "从答案生成指令",
  "Start from seed answer fields and generate matching user instructions for SFT pairs.": "从种子答案字段出发，生成匹配的用户指令用于 SFT 样本对。",
  "PDF Document QA": "PDF 文档问答",
  "Unstructured PDF chunks transformed into grounded question-answer training pairs.": "将非结构化 PDF 分块转换为有依据的问答训练样本对。",
  "OCR Document Extraction": "OCR 文档抽取",
  "Use image context from seed data to generate OCR-style extraction outputs.": "利用种子数据中的图像上下文，生成 OCR 风格的抽取结果。",
  "Text to Python": "文本转 Python",
  "Instruction-to-code pairs for training models that generate clean Python implementations.": "用于训练模型生成高质量 Python 实现的指令-代码样本对。",
  "Text to SQL": "文本转 SQL",
  "Natural language to SQL pairs, including schema-aware query construction patterns.": "自然语言到 SQL 的样本对，包含基于 schema 的查询构建模式。",
  "Structured Outputs + Jinja Expressions": "结构化输出 + Jinja 表达式",
  "Support ticket triage dataset with structured JSON outputs and Jinja if/else refs.": "用于工单分流的数据集，包含结构化 JSON 输出和 Jinja if/else 引用。",
  "Easy": "简单",
  "Starter": "入门",
  "Intermediate": "中级",
  "Last updated": "最近更新",
  "Created": "创建于",
  "Soon": "即将推出",
  "Export Model": "导出模型",
  "Export fine-tuned or base models for deployment": "导出微调模型或基础模型用于部署",
  "Export Configuration": "导出配置",
  "Select source, method, and quantization": "选择来源、导出方式和量化级别",
  "Training Run": "训练运行",
  "Model Source": "模型来源",
  "Use Hugging Face / Local Model": "使用 Hugging Face / 本地模型",
  "Use Training Checkpoints": "使用训练检查点",
  "No training runs found": "未找到训练运行",
  "Select a training run…": "选择一个训练运行…",
  "Select a training run first": "请先选择训练运行",
  "No checkpoints found": "未找到检查点",
  "Select a checkpoint…": "选择一个检查点…",
  "Loading checkpoints…": "正在加载检查点…",
  "Choose where to save your exported model.": "选择导出模型的保存位置。",
  "Your token is only used for this export request.": "你的令牌仅用于本次导出请求。",
  "Compare in Chat": "在聊天中对比",
  "Compare complete": "对比完成",
  "Compare failed": "对比失败",
  "Send to both models...": "发送给两个模型...",
  "Add Attachment": "添加附件",
  "Upload audio": "上传音频",
  "Reasoning effort": "推理强度",
  "Think:": "思考：",
  "Export Complete": "导出完成",
  "Model successfully pushed to Hugging Face Hub.": "模型已成功推送到 Hugging Face Hub。",
  "Model saved locally.": "模型已保存到本地。",
  "Saved to": "保存到",
  "Done": "完成",
  "Save Locally": "保存到本地",
  "Push to Hub": "推送到 Hub",
  "Username / Org": "用户名 / 组织",
  "Model Name": "模型名称",
  "HF Write Token": "HF 写入令牌",
  "Get token": "获取令牌",
  "Private repository": "私有仓库",
  "Export finished and pushed to Hugging Face Hub.": "导出完成并已推送到 Hugging Face Hub。",
  "Export finished successfully.": "导出成功。",
  "Checkpoint": "检查点",
  "Start Export": "开始导出",
  "guided tour": "引导教程",
  "Quick tour": "快速引导",
  "Let's get you oriented.": "先带你快速熟悉界面。",
  "Skip tour": "跳过引导",
  "Skip": "跳过",
  "Tip: `Esc` skips. Tour blocks clicks so you can read.": "提示：按 `Esc` 可跳过。引导期间会阻止点击，便于阅读。",
  "General": "通用",
  "Profile": "个人资料",
  "Appearance": "外观",
  "Chat": "聊天",
  "API Keys": "API 密钥",
  "About": "关于",
  "Settings": "设置",
  "Close settings": "关闭设置",
  "Manage your Unsloth Studio preferences.": "管理你的 Unsloth Studio 偏好设置。",
  "Global preferences for Unsloth Studio.": "Unsloth Studio 的全局偏好设置。",
  "Account": "账户",
  "Hugging Face token": "Hugging Face 令牌",
  "Used to load gated models and push artifacts.": "用于加载受限模型并推送产物。",
  "Hide token": "隐藏令牌",
  "Show token": "显示令牌",
  "Chat defaults": "聊天默认设置",
  "Auto-title new chats": "自动为新聊天命名",
  "Generate a short title from the first message.": "根据第一条消息自动生成简短标题。",
  "Getting started": "快速开始",
  "Start onboarding": "开始引导",
  "Open the setup wizard again without changing your account.": "在不更改账户的情况下重新打开设置向导。",
  "Danger zone": "危险区域",
  "Reset all local preferences": "重置所有本地偏好",
  "Clears theme, tokens, sidebar state, and presets. Chats and API keys are not affected.": "清除主题、令牌、侧边栏状态和预设。聊天和 API 密钥不受影响。",
  "Reset preferences": "重置偏好",
  "Reset all local preferences?": "要重置所有本地偏好吗？",
  "This clears your theme, tokens, and stored settings, then reloads Studio. Chats and API keys are not affected.": "这会清除你的主题、令牌和已保存设置，然后重新加载 Studio。聊天和 API 密钥不受影响。",
  "Cancel": "取消",
  "Reset and reload": "重置并重新加载",
  "Playground": "游乐场",
  "Your Chats": "你的聊天",
  "No threads yet": "暂无会话",
  "Learn more in docs": "在文档中了解更多",
  "What's new": "新功能",
  "Fine-tuning Studio": "微调工作台",
  "Training in progress": "训练进行中",
  "Viewing past run": "正在查看历史运行",
  "View past training runs": "查看历史训练运行",
  "Configure and start training": "配置并开始训练",
  "Loading training runtime...": "正在加载训练运行时...",
  "Back to history": "返回历史",
  "Configure": "配置",
  "Current Run": "当前运行",
  "History": "历史",
  "Model": "模型",
  "Dataset": "数据集",
  "Model weights": "模型权重",
  "Dataset Preview": "数据集预览",
  "Fetching dataset preview from Hugging Face...": "正在从 Hugging Face 获取数据集预览...",
  "Loading preview...": "正在加载预览...",
  "Train": "训练",
  "Recipes": "配方",
  "Export": "导出",
  "Recents": "最近",
  "Delete": "删除",
  "Search": "搜索",
  "Compare": "对比",
  "New Chat": "新建聊天",
  "Close sidebar": "关闭侧边栏",
  "Open sidebar": "打开侧边栏",
  "Studio": "工作台",
  "Light Mode": "浅色模式",
  "Dark Mode": "深色模式",
  "Guided Tour": "引导教程",
  "Learn More": "了解更多",
  "Feedback": "反馈",
  "Shutdown": "关闭服务",
  "Language": "语言",
  "Retry": "重试",
  "Save": "保存",
  "Saving...": "保存中...",
  "Saving…": "保存中…",
  "Load more": "加载更多",
  "Loading...": "加载中...",
  "Loading…": "加载中…",
  "Loading models…": "正在加载模型…",
  "Loading variants…": "正在加载变体…",
  "Loading variants...": "正在加载变体...",
  "Loading recipes": "正在加载配方",
  "Loading Model 1…": "正在加载模型 1…",
  "Loading Model 2…": "正在加载模型 2…",
  "Generating with Model 1…": "正在使用模型 1 生成…",
  "Generating with Model 2…": "正在使用模型 2 生成…",
  "New Recipe": "新建配方",
  "Start Empty": "从空白开始",
  "Start from Learning Recipe": "从学习配方开始",
  "Learning Recipe": "学习配方",
  "Start with source data": "从源数据开始",
  "Loading recipe": "正在加载配方",
  "Chat with your model": "与你的模型对话",
  "Run GGUFs, safetensors, vision and audio models": "运行 GGUF、safetensors、视觉和音频模型",
  "Stop dictation": "停止语音输入",
  "Stop generating": "停止生成",
  "Delete message": "删除消息",
  "Export as Markdown": "导出为 Markdown",
  "Next": "下一步",
  "Back": "返回",
  "Continue": "继续",
  "Start Server": "启动服务",
  "Stop server": "停止服务",
  "Stopping…": "正在停止…",
  "Run": "运行",
  "Running...": "运行中...",
  "Running…": "运行中…",
  "Error": "错误",
  "Warning": "警告",
  "Success": "成功",
  "Unknown error": "未知错误",
  "AI assist failed.": "AI 辅助失败。",
  "Failed to load preview": "加载预览失败",
  "Failed to load local datasets.": "加载本地数据集失败。",
  "Dataset uploaded": "数据集已上传",
  "Search Hugging Face datasets...": "搜索 Hugging Face 数据集...",
  "Search local datasets...": "搜索本地数据集...",
  "Loading local datasets...": "正在加载本地数据集...",
  "Train Split Start": "训练集起始位置",
  "Train Split End": "训练集结束位置",
  "Hugging Face Dataset": "Hugging Face 数据集",
  "Failed to load local models": "加载本地模型失败",
  "Local Model": "本地模型",
  "Hugging Face Model": "Hugging Face 模型",
  "Search models...": "搜索模型...",
  "Search datasets...": "搜索数据集...",
  "Upload Dataset": "上传数据集",
  "Start Onboarding": "开始引导",
  "Chart Settings": "图表设置",
  "Loading model": "正在加载模型",
  "Loading dataset": "正在加载数据集",
  "Export chat history": "导出聊天记录",
  "Exporting…": "导出中…",
  "Exporting...": "导出中...",
  "Delete selected preset": "删除所选预设",
  "Save as New": "另存为新预设",
  "Web Search": "网络搜索",
  "Search Hugging Face models": "搜索 Hugging Face 模型",
  "Search trained models": "搜索已训练模型",
  "Search datasets": "搜索数据集",
  "Search Hugging Face models or pick from our recommended list.": "搜索 Hugging Face 模型或从推荐列表中选择。",
  "Search for a command to run...": "搜索要运行的命令...",
  "Search for a command to run…": "搜索要运行的命令…",
  "Search Hugging Face models...": "搜索 Hugging Face 模型...",
  "Search trained models...": "搜索已训练模型...",
  "Run completed": "运行完成",
  "Run in progress": "运行进行中",
  "Run status": "运行状态",
  "Run summary": "运行摘要",
  "Model usage": "模型使用情况",
  "Next step": "下一步",
  "Failed to delete message": "删除消息失败",
  "Chat settings could not be persisted": "聊天设置无法保存",
  "Failed to add folder": "添加文件夹失败",
  "Failed to remove folder": "移除文件夹失败",
  "Failed to delete model": "删除模型失败",
  "Failed to load variants": "加载变体失败",
  "Delete cached model?": "删除缓存模型？",
  "Cancel adding folder": "取消添加文件夹",
  "Add scan folder by path": "按路径添加扫描文件夹",
  "Add by typing a path": "输入路径添加",
  "Delete training run?": "删除训练运行？",
  "Delete run": "删除运行",
  "No training runs yet. Start your first training run in the Configure tab.": "还没有训练记录。请先在“配置”标签中启动第一次训练。",
  "Failed to load training runs": "加载训练记录失败",
  "Failed to delete training run. Please try again.": "删除训练记录失败，请重试。",
  "Training history": "训练历史",
  "Progress": "进度",
  "Method": "方法",
  "Parameters": "参数",
  "Hyperparameters": "超参数",
  "Start training": "开始训练",
  "Cancel Training": "取消训练",
  "Continue Training": "继续训练",
  "Stop / save": "停止 / 保存",
  "Stop and Save": "停止并保存",
  "Start with a small run to sanity-check loss + sample quality.": "先用一次小规模运行验证损失和样本质量。",
  "loading_model": "正在加载模型",
  "loading_dataset": "正在加载数据集",
  "error": "错误",
  "Fine-tune large language models for text generation": "微调大语言模型以进行文本生成",
  "Train and run LLMs locally": "在本地训练和运行 LLM",
  "Data Recipe output.": "数据配方输出。",
  "Recipe name": "配方名称",
  "Save current settings to this preset": "将当前设置保存到该预设",
  "Use this for longer edits. Save writes back to the active preset.": "用于更长内容编辑。保存会写回当前预设。",
  "Couldn't load API keys.": "无法加载 API 密钥。",
  "Couldn't revoke key.": "无法吊销密钥。",
  "Failed to load API keys": "加载 API 密钥失败",
  "Failed to create API key": "创建 API 密钥失败",
  "Failed to revoke API key": "吊销 API 密钥失败",
  "Model alias": "模型别名",
  "Code language": "代码语言",
  "Select language": "选择语言",
  "Chart": "图表",
  "Use this for longer edits. Save writes back to the active preset": "用于更长内容编辑。保存会写回当前预设",
  "Enable custom code": "启用自定义代码",
  "Could not load image": "无法加载图片",
  "Image is still too large after compression. Try a smaller file.": "压缩后图片仍然过大，请尝试更小的文件。",
  "Progress stream unavailable": "进度流不可用",
  "Job stream unavailable.": "任务流不可用。",
  "Upload failed": "上传失败",
  "Failed to remove file": "删除文件失败",
  "Request failed": "请求失败",
  "AI assist failed": "AI 辅助失败",
  "Request failed (": "请求失败（",
  "Could not use this image.": "无法使用该图片。",
  "Invalid image dimensions": "图片尺寸无效",
  "Canvas not available": "画布不可用",
  "Failed to shut down server": "关闭服务器失败",
  "Could not reach server": "无法连接到服务器",
  "Unsloth Studio has stopped.": "Unsloth Studio 已停止。",
  "You can now close this tab.": "你现在可以关闭此标签页。",
};

const NORMALIZED_TEXT_ZH_CN = new Map<string, string>(
  Object.entries(EXACT_TEXT_ZH_CN).map(([key, value]) => [normalizeKey(key), value]),
);

const PHRASE_TEXT_ZH_CN: Array<readonly [string, string]> = Array.from(
  new Map<string, string>([
  ["Configure training hyperparameters", "配置训练超参数"],
  ["Number of full passes over the dataset.", "对数据集进行完整遍历的次数。"],
  ["Override total optimizer steps.", "覆盖总优化器步数。"],
  ["Each epoch is one full pass over your dataset.", "每个 epoch 表示完整遍历一次数据集。"],
  ["Limits training to a fixed number of optimizer steps.", "将训练限制为固定数量的优化器步数。"],
  ["Maximum number of tokens per training sample.", "每个训练样本的最大 token 数。"],
  ["Max sequence length for training samples", "训练样本的最大序列长度"],
  ["Enter a custom value", "输入自定义值"],
  ["Step size for weight updates. Lower values train slower but more stably.", "权重更新的步长。数值越小训练越慢但更稳定。"],
  ["Recommended: 2e-4 for LoRA, 2e-5 for full fine-tune", "推荐：LoRA 使用 2e-4，完整微调使用 2e-5"],
  ["Dimension of the low-rank matrices. Higher = more capacity.", "低秩矩阵的维度。越高代表容量越大。"],
  ["Scaling factor for LoRA updates. Usually 2x rank.", "LoRA 更新的缩放系数，通常为 rank 的 2 倍。"],
  ["Dropout probability for LoRA layers to reduce overfitting.", "LoRA 层的 dropout 概率，用于降低过拟合。"],
  ["Optimization algorithm. 8-bit variants reduce memory usage.", "优化算法。8-bit 变体可降低内存占用。"],
  ["Fused is recommended for vision models.", "视觉模型建议使用 Fused。"],
  ["How the learning rate changes over training.", "学习率在训练过程中的变化方式。"],
  ["Linear decays steadily; cosine decays in a curve.", "Linear 稳定衰减；cosine 以曲线方式衰减。"],
  ["Samples processed per step. Higher uses more VRAM.", "每步处理的样本数。越高占用的 VRAM 越多。"],
  ["Simulates larger batch sizes without extra VRAM.", "在不增加 VRAM 的情况下模拟更大的 batch。"],
  ["L2 regularization to prevent overfitting.", "L2 正则化，用于防止过拟合。"],
  ["Monitor and control training", "监控并控制训练"],
  ["No training data yet", "暂无训练数据"],
  ["Start training to see loss progress", "开始训练以查看 loss 变化"],
  ["Text model is not compatible with a multimodal dataset.", "文本模型与多模态数据集不兼容。"],
  ["Switch to a vision model or choose a text-only dataset.", "请切换到视觉模型或选择纯文本数据集。"],
  ["This model does not support audio.", "该模型不支持音频。"],
  ["Switch to an audio-capable model or choose a non-audio dataset.", "请切换到支持音频的模型或选择非音频数据集。"],
  ["Load a saved YAML config", "加载已保存的 YAML 配置"],
  ["Download current config as YAML", "将当前配置下载为 YAML"],
  ["Reset to model defaults", "重置为模型默认值"],
  ["Search Hugging Face models or pick from our recommended list.", "搜索 Hugging Face 模型，或从推荐列表中选择。"],
  ["Path to a locally downloaded model or a custom HF repo.", "本地下载模型路径或自定义 HF 仓库。"],
  ["QLoRA uses 4-bit quantization for lowest VRAM.", "QLoRA 使用 4-bit 量化以降低 VRAM 占用。"],
  ["LoRA uses 16-bit. Full updates all weights.", "LoRA 使用 16-bit。Full 会更新全部权重。"],
  ["Get or update token", "获取或更新令牌"],
  ["Use the popup tabs to switch between Hugging Face and local recipe outputs.", "使用弹出标签页在 Hugging Face 与本地配方输出之间切换。"],
  ["Browsing Local datasets.", "正在浏览本地数据集。"],
  ["Browsing Hugging Face.", "正在浏览 Hugging Face。"],
  ["Current selection stays Local.", "当前选择保持为本地。"],
  ["Current selection stays Hugging Face.", "当前选择保持为 Hugging Face。"],
  ["No local datasets yet.", "暂无本地数据集。"],
  ["No local datasets match search.", "没有匹配搜索条件的本地数据集。"],
  ["Open Data Recipes", "打开数据配方"],
  ["Local dataset metadata", "本地数据集元数据"],
  ["Eval dataset", "评估数据集"],
  ["Optional. If not provided, a small portion will be split from the training data.", "可选。若未提供，将从训练数据中切分出一小部分。"],
  ["Format of your training data. Auto-detect works for most datasets.", "训练数据格式。自动检测适用于大多数数据集。"],
  ["Only train on a subset of your training split by specifying a start row index (inclusive, 0-based).", "通过指定起始行索引（包含，0 基）仅训练训练集的子集。"],
  ["Leave empty to start from the first row.", "留空则从第一行开始。"],
  ["Last row index to include from the training split (inclusive, 0-based).", "从训练集中包含到的最后行索引（包含，0 基）。"],
  ["For example, set Start to 0 and End to 99 to train on the first 100 rows.", "例如将 Start 设为 0、End 设为 99，即训练前 100 行。"],
  ["Leave empty to use all remaining rows.", "留空则使用其余所有行。"],
  ["Hugging Face Dataset", "Hugging Face 数据集"],
  ["No dataset selected", "未选择数据集"],
  ["Search local datasets...", "搜索本地数据集..."],
  ["Search Hugging Face datasets...", "搜索 Hugging Face 数据集..."],
  ["Checking token…", "正在检查令牌…"],
  ["Loading local datasets...", "正在加载本地数据集..."],
  ["Scanning local models...", "正在扫描本地模型..."],
  ["No local models found. Enter path manually.", "未找到本地模型，请手动输入路径。"],
  ["local/cached models found", "个本地/缓存模型已找到"],
  ["Search models...", "搜索模型..."],
  ["No models found", "未找到模型"],
  ["Searching...", "搜索中..."],
  ["Searching…", "搜索中…"],
  ["Training Config", "训练配置"],
  ["Upload eval file", "上传评估文件"],
  ["Uploading...", "上传中..."],
  ["Checking dataset...", "正在检查数据集..."],
  ["Loading model...", "正在加载模型..."],
  ["Start Training", "开始训练"],
  ["Target Modules", "目标模块"],
  ["Training Hyperparameters", "训练超参数"],
  ["LoRA Settings", "LoRA 设置"],
  ["Learning Rate", "学习率"],
  ["Context Length", "上下文长度"],
  ["Max Steps", "最大步数"],
  ["Epochs", "轮次"],
  ["Use Max Steps", "使用最大步数"],
  ["Use Epochs", "使用轮次"],
  ["Rank", "秩"],
  ["Alpha", "Alpha"],
  ["Dropout", "Dropout"],
  ["Vision layers", "视觉层"],
  ["Language layers", "语言层"],
  ["Attention modules", "注意力模块"],
  ["MLP modules", "MLP 模块"],
  ["Enable LoRA", "启用 LoRA"],
  ["Train with LoRA", "使用 LoRA 训练"],
  ["Stable Rank", "稳定秩"],
  ["Memory Efficient", "内存高效"],
  ["Optimization", "优化"],
  ["Schedule", "调度"],
  ["Memory", "内存"],
  ["Optimizer", "优化器"],
  ["LR scheduler", "学习率调度器"],
  ["Batch Size", "批大小"],
  ["Grad Accum", "梯度累积"],
  ["Weight Decay", "权重衰减"],
  ["Rows", "行数"],
  ["Columns", "列数"],
  ["Batches", "批次"],
  ["Updated", "更新时间"],
  ["Choose dataset", "选择数据集"],
  ["Local", "本地"],
  ["Use the popup tabs to switch between Hugging Face and local recipe outputs.", "使用弹出标签页在 Hugging Face 与本地配方输出之间切换。"],
  ["Searching...", "搜索中..."],
  ["No datasets found", "未找到数据集"],
  ["No local datasets yet.", "暂无本地数据集。"],
  ["No local datasets match search.", "没有匹配搜索条件的本地数据集。"],
  ["Browsing", "正在浏览"],
  ["Current selection stays", "当前选择保持为"],
  ["Local dataset metadata", "本地数据集元数据"],
  ["Data Recipe output.", "数据配方输出。"],
  ["Eval dataset", "评估数据集"],
  ["Upload eval file", "上传评估文件"],
  ["Optional. If not provided, a small portion will be split from the training data.", "可选。若未提供，将从训练数据中切分出一小部分。"],
  ["Target Format", "目标格式"],
  ["Format of your training data. Auto-detect works for most datasets.", "训练数据格式。自动检测适用于大多数数据集。"],
  ["Only train on a subset of your training split by specifying a start row index (inclusive, 0-based).", "通过指定起始行索引（包含，0 基）仅训练训练集的子集。"],
  ["Leave empty to start from the first row.", "留空则从第一行开始。"],
  ["Last row index to include from the training split (inclusive, 0-based).", "从训练集中包含到的最后行索引（包含，0 基）。"],
  ["For example, set Start to 0 and End to 99 to train on the first 100 rows.", "例如将 Start 设为 0、End 设为 99，即训练前 100 行。"],
  ["Leave empty to use all remaining rows.", "留空则使用其余所有行。"],
  ["Hugging Face Dataset", "Hugging Face 数据集"],
  ["No dataset selected", "未选择数据集"],
  ["Clear", "清除"],
  ["Unable to load dataset splits. This dataset may be private or gated. Add a Hugging Face token with access and try again.", "无法加载数据集切分。该数据集可能为私有或受限。请添加有权限的 Hugging Face 令牌后重试。"],
  ["We can’t load subset/split options for this Hub dataset because it relies on a legacy custom script.", "无法为该 Hub 数据集加载子集/切分选项，因为它依赖旧版自定义脚本。"],
  ["Dataset not found. Check the dataset name and try again.", "未找到数据集。请检查数据集名称后重试。"],
  ["Unable to load dataset split options for this dataset.", "无法加载该数据集的切分选项。"],
  ["Merged Model", "合并模型"],
  ["Full 16-bit model ready for inference.", "可直接推理的完整 16-bit 模型。"],
  ["Merges adapter weights into the base model. Best for direct deployment with vLLM or TGI.", "将适配器权重合并到基础模型中。最适合直接部署到 vLLM 或 TGI。"],
  ["LoRA Only", "仅 LoRA"],
  ["Lightweight adapter files (~100 MB). Needs base model.", "轻量级适配器文件（约 100 MB），需要基础模型。"],
  ["Exports only the trained adapter. Pair with the base model at inference time to save storage.", "仅导出训练后的适配器。推理时与基础模型配合可节省存储。"],
  ["GGUF / Llama.cpp", "GGUF / Llama.cpp"],
  ["Quantized formats for local AI runners.", "适用于本地 AI 运行器的量化格式。"],
  ["Converts to GGUF for llama.cpp, Ollama, and other local runners. Pick a quantization level below.", "转换为 GGUF，可用于 llama.cpp、Ollama 及其他本地运行器。请在下方选择量化等级。"],
  ["Select a training checkpoint to export from", "选择要导出的训练检查点"],
  ["Choose an export method based on your use case", "根据你的使用场景选择导出方式"],
  ["Pick quantization levels if using GGUF", "若使用 GGUF，请选择量化等级"],
  ["Click Export and choose your destination", "点击导出并选择目标位置"],
  ["Test your model and compare outputs in Chat", "在聊天中测试模型并对比输出"],
  ["Pick training run", "选择训练运行"],
  ["Start by selecting the training run. Each run groups the checkpoints produced by that specific fine-tuning job.", "先选择训练运行。每个运行会归集该次微调任务产出的检查点。"],
  ["Pick checkpoint", "选择检查点"],
  ["Pick which checkpoint to export. If you trained multiple checkpoints, it’s worth exporting 1-2 candidates and testing in Chat.", "选择要导出的检查点。如果训练出了多个检查点，建议导出 1-2 个候选并在聊天中测试。"],
  ["Export method", "导出方式"],
  ["Choose the packaging. GGUF is for llama.cpp-style runtimes (pick a quant). Safetensors is for HF/Transformers-style usage. If you’re unsure, start with safetensors.", "选择打包格式。GGUF 适用于 llama.cpp 类运行时（需选择量化等级）；Safetensors 适用于 HF/Transformers 场景。不确定时建议先用 safetensors。"],
  ["Export to local or push to HF Hub. After export, test in Chat and compare against base to confirm behavior is what you expect.", "可导出到本地或推送到 HF Hub。导出后请在聊天中测试，并与基础模型对比确认行为符合预期。"],
  ["Pick a model", "选择模型"],
  ["This selects what’s loaded for inference. Hub = base models. Fine-tuned = trained Studio outputs, including LoRA adapters and full finetunes.", "这里决定推理时加载的模型。Hub=基础模型；Fine-tuned=Studio 训练产物，包括 LoRA 适配器和完整微调模型。"],
  ["Two tabs", "两个标签页"],
  ["Hub: search Hugging Face models. Fine-tuned: local Studio outputs you’ve trained or exported. If results look off, compare base vs fine-tuned outputs to see what changed.", "Hub：搜索 Hugging Face 模型。Fine-tuned：你训练或导出的本地 Studio 产物。如果结果异常，可对比基础模型与微调模型输出来定位变化。"],
  ["Settings sidebar", "设置侧边栏"],
  ["Sampling (temperature/top-p/top-k) + system prompt live here. If you want more deterministic outputs, lower temperature first.", "采样参数（temperature/top-p/top-k）和系统提示词在这里设置。若希望结果更稳定，可先降低 temperature。"],
  ["Compare mode", "对比模式"],
  ["Compare any two models side-by-side.", "并排对比任意两个模型。"],
  ["Pick a different model for each side and see how they respond to the same prompt.", "为左右两侧选择不同模型，查看它们对同一提示词的响应差异。"],
  ["Side-by-side threads", "并排会话"],
  ["Same prompt, 2 threads. If LoRA is worse than base, it’s usually data formatting, too many epochs, or a bad checkpoint choice.", "相同提示词，双线程对比。若 LoRA 表现差于基础模型，通常是数据格式、训练轮次过多或检查点选择不佳导致。"],
  ["Desktop authentication failed. Update or repair the managed Studio install, then restart Studio.", "桌面端认证失败。请更新或修复 Studio 安装后重启。"],
  ["Cancel Training", "取消训练"],
  ["Do you want to cancel the current training run?", "你要取消当前训练任务吗？"],
  ["Continue Training", "继续训练"],
  ["> unsloth training starts...", "> unsloth 训练开始..."],
  ["> Preparing model and dataset...", "> 正在准备模型和数据集..."],
  ["> We are getting everything ready for your run...", "> 正在为你的训练做好准备..."],
  ["starting training...", "正在开始训练..."],
  ["waiting for first step...", "等待第一步..."],
  ["Model weights", "模型权重"],
  ["Advanced", "高级"],
  ["Target Format", "目标格式"],
  ["Train Split Start", "训练集起始位置"],
  ["Train Split End", "训练集结束位置"],
  ["Local dataset", "本地数据集"],
  ["Clear", "清除"],
  ]).entries(),
).sort((a, b) => b[0].length - a[0].length);

let observer: MutationObserver | null = null;
const textOriginal = new WeakMap<Text, string>();
const attrOriginal = new WeakMap<Element, Map<string, string>>();
const touchedTextNodes = new Set<Text>();
const touchedAttrElements = new Set<Element>();

function normalizeKey(input: string): string {
  return input
    .trim()
    .replace(/\s+/g, " ")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"');
}

function translateExact(locale: LocaleCode, input: string): string {
  if (locale !== "zh-CN") return input;

  const key = normalizeKey(input);
  if (!key) return input;

  const translated = NORMALIZED_TEXT_ZH_CN.get(key);
  if (translated) {
    const leading = input.match(/^\s*/)?.[0] ?? "";
    const trailing = input.match(/\s*$/)?.[0] ?? "";
    return `${leading}${translated}${trailing}`;
  }

  let output = input;
  for (const [from, to] of PHRASE_TEXT_ZH_CN) {
    if (!output.includes(from)) continue;
    output = output.split(from).join(to);
  }
  return output;
}

function hasClassPrefix(element: Element, prefix: string): boolean {
  for (const className of element.classList) {
    if (className.startsWith(prefix)) return true;
  }
  return false;
}

function shouldSkipNode(node: Node): boolean {
  const scopeRoot =
    node instanceof Element ? node : node.parentElement;
  const forceTranslate = Boolean(
    scopeRoot?.closest("[data-force-translate='true']"),
  );

  let current: Element | null =
    node instanceof Element
      ? node
      : node.parentElement;

  while (current) {
    const tag = current.tagName;
    if (
      tag === "CODE" ||
      tag === "PRE" ||
      tag === "SCRIPT" ||
      tag === "STYLE" ||
      tag === "NOSCRIPT" ||
      tag === "TEXTAREA" ||
      tag === "KBD" ||
      tag === "SAMP"
    ) {
      return true;
    }
    if (current.hasAttribute("data-no-translate")) return true;
    if (current.getAttribute("contenteditable") === "true") return true;
    if (hasClassPrefix(current, "aui-") && !forceTranslate) return true;
    current = current.parentElement;
  }

  return false;
}

function applyTextTranslation(textNode: Text, locale: LocaleCode): void {
  if (shouldSkipNode(textNode)) return;

  if (!textOriginal.has(textNode)) {
    textOriginal.set(textNode, textNode.nodeValue ?? "");
    touchedTextNodes.add(textNode);
  }

  const source = textOriginal.get(textNode) ?? textNode.nodeValue ?? "";
  const translated = translateExact(locale, source);

  if (textNode.nodeValue !== translated) {
    textNode.nodeValue = translated;
  }
}

function applyAttributeTranslation(element: Element, locale: LocaleCode): void {
  if (shouldSkipNode(element)) return;

  for (const name of TRANSLATABLE_ATTRIBUTES) {
    const value = element.getAttribute(name);
    if (!value) continue;

    if (!attrOriginal.has(element)) {
      attrOriginal.set(element, new Map<string, string>());
      touchedAttrElements.add(element);
    }

    const perElement = attrOriginal.get(element);
    if (!perElement) continue;

    if (!perElement.has(name)) {
      perElement.set(name, value);
    }

    const source = perElement.get(name) ?? value;
    const translated = translateExact(locale, source);
    if (translated !== value) {
      element.setAttribute(name, translated);
    }
  }
}

function translateSubtree(root: Node, locale: LocaleCode): void {
  if (root.nodeType === Node.TEXT_NODE) {
    applyTextTranslation(root as Text, locale);
    return;
  }

  if (root.nodeType !== Node.ELEMENT_NODE) return;

  const rootElement = root as Element;
  applyAttributeTranslation(rootElement, locale);

  const walker = document.createTreeWalker(rootElement, NodeFilter.SHOW_TEXT);
  let textNode = walker.nextNode();
  while (textNode) {
    applyTextTranslation(textNode as Text, locale);
    textNode = walker.nextNode();
  }

  for (const name of TRANSLATABLE_ATTRIBUTES) {
    const nodes = rootElement.querySelectorAll(`[${name}]`);
    for (const node of nodes) {
      applyAttributeTranslation(node, locale);
    }
  }
}

function restoreOriginalText(): void {
  for (const textNode of touchedTextNodes) {
    if (!textNode.isConnected) continue;
    const original = textOriginal.get(textNode);
    if (typeof original === "string" && textNode.nodeValue !== original) {
      textNode.nodeValue = original;
    }
  }

  for (const element of touchedAttrElements) {
    if (!element.isConnected) continue;
    const originalMap = attrOriginal.get(element);
    if (!originalMap) continue;
    for (const [name, value] of originalMap.entries()) {
      if (element.getAttribute(name) !== value) {
        element.setAttribute(name, value);
      }
    }
  }
}

function disconnectObserver(): void {
  if (!observer) return;
  observer.disconnect();
  observer = null;
}

function observeDom(locale: LocaleCode): void {
  disconnectObserver();

  observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type === "childList") {
        for (const node of mutation.addedNodes) {
          translateSubtree(node, locale);
        }
      } else if (mutation.type === "characterData") {
        if (mutation.target.nodeType === Node.TEXT_NODE) {
          applyTextTranslation(mutation.target as Text, locale);
        }
      } else if (mutation.type === "attributes") {
        const target = mutation.target;
        if (target.nodeType === Node.ELEMENT_NODE) {
          applyAttributeTranslation(target as Element, locale);
        }
      }
    }
  });

  observer.observe(document.body, {
    subtree: true,
    childList: true,
    characterData: true,
    attributes: true,
    attributeFilter: [...TRANSLATABLE_ATTRIBUTES],
  });
}

export function syncAutoDomTranslations(locale: LocaleCode): void {
  disconnectObserver();

  if (locale === "en") {
    restoreOriginalText();
    return;
  }

  translateSubtree(document.body, locale);
  observeDom(locale);
}
