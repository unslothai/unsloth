// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export const studioSaveStep: TourStep = {
  id: "save",
  target: "studio-save",
  title: "保存配置",
  body: (
    <>
      将训练配置保存为 YAML 文件。基于同一基线重复运行，能更清楚判断改动是否真的有效（而不是偶然结果）。
    </>
  ),
};
