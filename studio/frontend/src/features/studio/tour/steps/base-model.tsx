// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioBaseModelStep: TourStep = {
  id: "base-model",
  target: "studio-base-model",
  title: "Hugging Face 模型",
  body: (
    <>
      可粘贴 <span className="font-mono">org/model</span> 或直接搜索。优先选择与你任务接近的基础模型（chat/instruct 或 base）。
      小模型迭代更快，等提示词与数据效果稳定后再升级到更大模型。{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/what-model-should-i-use" />
    </>
  ),
};
