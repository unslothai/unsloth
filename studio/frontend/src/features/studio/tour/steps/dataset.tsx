// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioDatasetStep: TourStep = {
  id: "dataset",
  target: "studio-dataset",
  title: "数据集",
  body: (
    <>
      可在 Hub 搜索或粘贴 <span className="font-mono">user/dataset</span>。先预览几行数据：格式通常比规模更重要。
      我们会尝试自动转换为受支持的训练格式；若无法可靠识别，会提示你手动映射字段。
      如果后续聊天效果异常，优先检查数据集格式与模板。{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide" />
    </>
  ),
};
