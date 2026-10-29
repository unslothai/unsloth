// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelChoice } from "../types";

/** A saved model param as a choice: null keeps "use the loaded model"; a bare repo string still loads. */
export function modelChoiceOf(value: unknown): ModelChoice | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") return { repo: value, variant: null };
  const choice = value as Partial<ModelChoice>;
  return { repo: choice.repo ?? "", variant: choice.variant ?? null };
}

/** What the picker returns, stored in the same shape the backend reads. */
export function choiceFromPick(
  id: string,
  meta: { ggufVariant?: string | null } | undefined,
): ModelChoice {
  return { repo: id, variant: meta?.ggufVariant ?? null };
}

/** Fine-tune trains safetensors weights; a GGUF repo can't be its base model. */
export function isTrainableRepo(id: string): boolean {
  return !/gguf/i.test(id);
}
