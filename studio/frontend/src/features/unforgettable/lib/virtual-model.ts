// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const VIRTUAL_MODEL_ID = "unforgettable";
export const VIRTUAL_MODEL_PREFIX = "unforgettable/";

export function isVirtualModel(model: string | null | undefined): boolean {
  if (!model) return false;
  return model === VIRTUAL_MODEL_ID || model.startsWith(VIRTUAL_MODEL_PREFIX);
}
