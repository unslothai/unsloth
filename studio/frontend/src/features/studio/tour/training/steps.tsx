// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const studioTrainingTourSteps: TourStep[] = [
  {
    id: "nav",
    target: "navbar",
    title: "训练视图",
    body: (
      <>
        训练进行时该视图会实时更新。你可以观察损失、速度与预计剩余时间，需要中止或保存时可使用停止按钮。
      </>
    ),
  },
  {
    id: "progress",
    target: "studio-training-progress",
    title: "进度与预计时间",
    body: (
      <>
        阶段会显示当前任务（加载模型/数据集、配置、训练）。前几步的预计时间偏粗略，运行一会后会更稳定。
      </>
    ),
  },
  {
    id: "train-loss",
    target: "studio-training-loss",
    title: "训练损失",
    body: (
      <>
        训练损失通常应整体下降。绝对值会受数据集与分词器影响，因此更应关注趋势而非某个“固定阈值”。若损失过低（如低于约 0.2）可能过拟合；若高位平台，通常需要更好的数据格式、更多数据或调整超参数。
      </>
    ),
  },
  {
    id: "eval-loss",
    target: "studio-eval-loss",
    title: "验证损失（Eval）",
    body: (
      <>
        Eval 损失是关键校验指标。如果训练损失持续下降而验证损失上升，通常意味着过拟合。要跟踪它，请配置验证集和 `eval_steps`（`eval_steps=1` 可能很慢）。
      </>
    ),
  },
  {
    id: "stop",
    target: "studio-training-stop",
    title: "停止 / 保存",
    body: (
      <>
        你可以随时停止训练。“停止并保存”会保留检查点/适配器，便于后续导出或对比。
      </>
    ),
  },
];
