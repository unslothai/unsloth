// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelType } from "@/types/training";

export function trainingModelMatchesTypeConstraint(
  modelType: ModelType,
  requiredType: ModelType | undefined,
  hasModelTypeSignal = true,
): boolean {
  return (
    requiredType === undefined ||
    !hasModelTypeSignal ||
    modelType === requiredType
  );
}
