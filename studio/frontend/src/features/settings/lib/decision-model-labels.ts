// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

export const DECISION_MODEL_LABELS: Record<string, TranslationKey> = {
  "laya-multilingual": "settings.apiKeys.decisionApi.modelMultilingual",
  "laya-english": "settings.apiKeys.decisionApi.modelEnglish",
  "laya-typed-decisions": "settings.apiKeys.decisionApi.modelTypedDecisions",
  "clef-flash": "settings.apiKeys.decisionApi.modelClefFlash",
  clef: "settings.apiKeys.decisionApi.modelClef",
};

/** Cloudflare Clef models are named by themselves; the Laya labels read as "Laya <label>". */
export function isClefDecisionModel(name: string | null | undefined): boolean {
  return (
    name === "clef" || name === "clef-flash" || !!name?.startsWith("clef-ft:")
  );
}
