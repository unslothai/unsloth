// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const studioStartStep: TourStep = {
  id: "start",
  target: "studio-start",
  title: "开始训练",
  body: (
    <>
      启动训练。若一开始就报错，先检查 HF 令牌、本地路径与数据集访问权限。建议先跑一个小规模任务验证损失与样例输出，再投入长时间训练。
    </>
  ),
};
