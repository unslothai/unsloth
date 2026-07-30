// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";

export function normalizeTrainingStartError(message: string): string {
  const normalized = message.toLowerCase();
  if (
    normalized.includes("failed to check dataset format") &&
    normalized.includes("dataset scripts are no longer supported")
  ) {
    return translate("studio.training.legacyDatasetScriptUnsupported");
  }
  return message;
}
