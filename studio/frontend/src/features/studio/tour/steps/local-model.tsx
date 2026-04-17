// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioLocalModelStep: TourStep = {
  id: "local-model",
  target: "studio-local-model",
  title: "本地模型路径",
  body: (
    <>
      如果你已经本地下载了权重（例如{" "}
      <span className="font-mono">./models/...</span>），可直接使用，避免重复下载。
      目录结构应与 Hugging Face 模型一致（包含 config、tokenizer、weights）。{" "}
      <ReadMore href="https://unsloth.ai/docs/basics/fine-tuning-llms-guide" />
    </>
  ),
};
