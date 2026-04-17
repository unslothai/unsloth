// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioMethodStep: TourStep = {
  id: "method",
  target: "studio-method",
  title: "训练方式：QLoRA / LoRA / 全量",
  body: (
    <>
      LoRA 只训练小规模适配器参数（速度快，常用默认）；QLoRA 在 4-bit 基座上训练 LoRA（显存占用更低）；全量微调会更新全部权重（成本最高，通常需要更多数据）。{" "}
      <ReadMore href="https://unsloth.ai/docs/basics/lora-hyperparameters-guide" />
    </>
  ),
};
