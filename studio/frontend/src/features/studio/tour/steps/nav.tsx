// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioNavStep: TourStep = {
  id: "nav",
  target: "navbar",
  title: "快速导览",
  body: (
    <>
      在 Studio 中依次选择基础模型、数据集与超参数，然后开始训练。启动后会看到实时损失与指标。Chat 可对比基础模型与 LoRA 适配器效果，Export 可打包检查点用于部署。{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-for-beginners" />
    </>
  ),
};
