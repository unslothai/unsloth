// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const exportTourSteps: TourStep[] = [
  {
    id: "training-run",
    target: "export-training-run",
    title: "选择训练运行",
    body: (
      <>
        先选择训练运行。每次运行都会汇总该次微调任务生成的所有检查点。
      </>
    ),
  },
  {
    id: "checkpoint",
    target: "export-checkpoint",
    title: "选择检查点",
    body: (
      <>
        选择要导出的检查点。如果你训练了多个检查点，建议先导出 1-2 个候选并在 Chat 中测试。
      </>
    ),
  },
  {
    id: "method",
    target: "export-method",
    title: "导出方式",
    body: (
      <>
        选择打包格式。GGUF 适用于 llama.cpp 类运行时（需选择量化），Safetensors 适用于
        HF/Transformers 场景。不确定时可先从 Safetensors 开始。
      </>
    ),
  },
  {
    id: "cta",
    target: "export-cta",
    title: "开始导出",
    body: (
      <>
        可导出到本地或推送到 HF Hub。导出后请在 Chat 中与基座模型对比，确认行为符合预期。
      </>
    ),
  },
];
