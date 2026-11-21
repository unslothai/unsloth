// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Model paths the backend loads onto the NPU (`lemonade:<Lemonade model id>`). */
export const NPU_MODEL_PREFIX = "lemonade:";

export function isNpuModelId(
  value: string | null | undefined,
): value is string {
  return typeof value === "string" && value.startsWith(NPU_MODEL_PREFIX);
}
