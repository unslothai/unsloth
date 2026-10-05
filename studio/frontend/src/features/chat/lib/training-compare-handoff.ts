// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const TRAINING_COMPARE_HANDOFF_KEY = "chat:training-compare-handoff:v1";
const HANDOFF_MAX_AGE_MS = 15 * 60 * 1000;

export type TrainingCompareHandoff = {
  intent: "compare";
  baseModel: string | null;
  outputDir: string | null;
  requestedAt: number;
};

type TrainingCompareCandidate = {
  id: string;
  baseModel: string;
  updatedAt?: number;
  exportType?: "lora" | "merged" | "gguf";
};

export function setTrainingCompareHandoff(
  baseModel: string | null,
  outputDir: string | null = null,
): void {
  if (typeof window === "undefined") return;

  const payload: TrainingCompareHandoff = {
    intent: "compare",
    baseModel,
    outputDir,
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
      outputDir:
        typeof parsed.outputDir === "string" ? parsed.outputDir : null,
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

export function normalizeModelRef(value: string | null | undefined): string {
  return value?.trim().toLowerCase() ?? "";
}

function normalizePath(path: string): string {
  return path.replace(/[/\\]+/g, "/").replace(/\/$/, "");
}

export function pickTrainingCompareTarget<T extends TrainingCompareCandidate>(
  loras: T[],
  handoff: Pick<TrainingCompareHandoff, "baseModel" | "outputDir">,
): T | null {
  const outputDir = handoff.outputDir ? normalizePath(handoff.outputDir) : "";
  const run = outputDir
    ? loras.find((lora) => normalizePath(lora.id) === outputDir)
    : undefined;
  if (run) return run;

  const adapterOnly = loras.filter((lora) => lora.exportType === "lora");
  if (adapterOnly.length === 0) return null;
  const sorted = [...adapterOnly].sort(
    (a, b) => (b.updatedAt ?? -1) - (a.updatedAt ?? -1),
  );
  const normalizedBase = normalizeModelRef(handoff.baseModel);
  if (!normalizedBase) return sorted[0] ?? null;

  const exact = sorted.find(
    (lora) => normalizeModelRef(lora.baseModel) === normalizedBase,
  );
  if (exact) return exact;

  const partial = sorted.find((lora) => {
    const normalizedLoraBase = normalizeModelRef(lora.baseModel);
    if (!normalizedLoraBase) return false;
    return (
      normalizedLoraBase.includes(normalizedBase) ||
      normalizedBase.includes(normalizedLoraBase)
    );
  });
  return partial ?? null;
}

export function trainingCompareSelection(target: TrainingCompareCandidate) {
  return {
    id: target.id,
    isLora: target.exportType === "lora",
    isDownloaded: true,
  };
}
