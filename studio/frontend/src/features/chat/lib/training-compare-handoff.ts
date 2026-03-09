// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

const TRAINING_COMPARE_HANDOFF_KEY = "chat:training-compare-handoff:v1";
const HANDOFF_MAX_AGE_MS = 15 * 60 * 1000;

export type TrainingCompareHandoff = {
  intent: "compare";
  baseModel: string | null;
  requestedAt: number;
};

export function setTrainingCompareHandoff(baseModel: string | null): void {
  if (typeof window === "undefined") return;

  const payload: TrainingCompareHandoff = {
    intent: "compare",
    baseModel,
    requestedAt: Date.now(),
  };
  window.sessionStorage.setItem(
    TRAINING_COMPARE_HANDOFF_KEY,
    JSON.stringify(payload),
  );
}

export function getTrainingCompareHandoff(): TrainingCompareHandoff | null {
  if (typeof window === "undefined") return null;

  const raw = window.sessionStorage.getItem(TRAINING_COMPARE_HANDOFF_KEY);
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as Partial<TrainingCompareHandoff>;
    if (parsed.intent !== "compare") return null;
    if (typeof parsed.requestedAt !== "number") return null;
    if (Date.now() - parsed.requestedAt > HANDOFF_MAX_AGE_MS) {
      clearTrainingCompareHandoff();
      return null;
    }
    return {
      intent: "compare",
      baseModel:
        typeof parsed.baseModel === "string" ? parsed.baseModel : null,
      requestedAt: parsed.requestedAt,
    };
  } catch {
    clearTrainingCompareHandoff();
    return null;
  }
}

export function clearTrainingCompareHandoff(): void {
  if (typeof window === "undefined") return;
  window.sessionStorage.removeItem(TRAINING_COMPARE_HANDOFF_KEY);
}
