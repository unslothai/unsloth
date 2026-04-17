// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioParamsStep: TourStep = {
  id: "params",
  target: "studio-params",
  title: "调整超参数",
  body: (
    <>
      先用稳妥参数起步，再逐步迭代。通常建议从 1-3 个 epoch 开始（更高可能更快过拟合）。不确定时一次只改一个参数，并观察训练集与验证集损失变化。{" "}
      <ReadMore href="https://unsloth.ai/docs/basics/lora-hyperparameters-guide" />
    </>
  ),
};
