// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingMethod } from "@/types/training";

export type ExportMethod = "merged" | "lora" | "gguf";

export const EXPORT_METHODS = (t: (key: string) => string): Array<{
  value: ExportMethod;
  title: string;
  description: string;
  tooltip: string;
  badge?: string;
}> => [
  {
    value: "merged",
    title: t("export.mergedModel"),
    description: t("export.mergedModelDesc"),
    tooltip: t("export.mergedModelTooltip"),
  },
  {
    value: "lora",
    title: t("export.loraOnly"),
    description: t("export.loraOnlyDesc"),
    tooltip: t("export.loraOnlyTooltip"),
  },
  {
    value: "gguf",
    title: t("export.ggufLlamaCpp"),
    description: t("export.ggufLlamaCppDesc"),
    tooltip: t("export.ggufLlamaCppTooltip"),
  },
];

export const QUANT_OPTIONS = [
  { value: "q3_k_m", label: "Q3_K_M", size: "~3.5 GB" },
  { value: "q4_0", label: "Q4_0", size: "~4.1 GB" },
  { value: "q4_k_m", label: "Q4_K_M", size: "~4.8 GB", recommended: true },
  { value: "q5_0", label: "Q5_0", size: "~5.0 GB" },
  { value: "q5_k_m", label: "Q5_K_M", size: "~5.6 GB" },
  { value: "q8_0", label: "Q8_0", size: "~8.2 GB" },
  { value: "f16", label: "F16", size: "~14.2 GB" },
  { value: "f32", label: "F32", size: "~28.4 GB" },
];

export function getEstimatedSize(
  method: ExportMethod | null,
  quantLevels: string[],
) {
  const sizeOf = (v: string) =>
    QUANT_OPTIONS.find((q) => q.value === v)?.size ?? "—";
  if (method === "gguf" && quantLevels.length > 0) {
    if (quantLevels.length === 1) {
      return sizeOf(quantLevels[0]);
    }
    const total = quantLevels
      .map((q) => Number.parseFloat(sizeOf(q).replace(/[^0-9.]/g, "")))
      .reduce((a, b) => a + b, 0);
    return `~${total.toFixed(1)} GB (${quantLevels.length} files)`;
  }
  if (method === "merged") {
    return "~14.2 GB";
  }
  if (method === "lora") {
    return "~100 MB";
  }
  return "—";
}

export const METHOD_LABELS: Record<TrainingMethod, string> = {
  qlora: "QLoRA",
  lora: "LoRA",
  full: "全量微调",
};

export const GUIDE_STEPS = [
  "选择要导出的训练检查点",
  "根据使用场景选择导出方式",
  "若使用 GGUF，请选择量化等级",
  "点击导出并选择输出目标",
  "在 Chat 中测试并对比模型效果",
];
