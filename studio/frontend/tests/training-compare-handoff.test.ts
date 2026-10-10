// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as handoff from "../src/features/chat/lib/training-compare-handoff.ts";

const OUTPUTS = "/home/u/.unsloth/studio/outputs";
const QWEN = "unsloth/Qwen3-4B-Instruct-2507";
const fullFinetune = {
  id: `${OUTPUTS}/unsloth_Qwen3-4B-Instruct-2507_1759100000`,
  baseModel: QWEN,
  updatedAt: 1759100000,
  exportType: "merged" as const,
};
const olderLlamaLora = {
  id: `${OUTPUTS}/unsloth_Llama-3.2-1B-Instruct_1759000000`,
  baseModel: "unsloth/Llama-3.2-1B-Instruct",
  updatedAt: 1759000000,
  exportType: "lora" as const,
};

function withSessionStorage(run: () => void): void {
  const values = new Map<string, string>();
  const prior = Object.getOwnPropertyDescriptor(globalThis, "window");
  Object.defineProperty(globalThis, "window", {
    value: {
      sessionStorage: {
        getItem: (key: string) => values.get(key) ?? null,
        setItem: (key: string, value: string) => values.set(key, value),
        removeItem: (key: string) => values.delete(key),
      },
    },
    configurable: true,
  });
  try {
    run();
  } finally {
    if (prior) Object.defineProperty(globalThis, "window", prior);
    else delete (globalThis as { window?: unknown }).window;
  }
}

function pick(
  loras: (typeof fullFinetune | typeof olderLlamaLora)[],
  outputDir: string | null,
) {
  return handoff.pickTrainingCompareTarget(loras, {
    baseModel: QWEN,
    outputDir,
  });
}

test("the compare handoff carries the finished run's folder to Chat", () => {
  withSessionStorage(() => {
    handoff.setTrainingCompareHandoff(QWEN, fullFinetune.id);
    assert.equal(
      handoff.getTrainingCompareHandoff()?.outputDir,
      fullFinetune.id,
    );
  });
});

test("a finished full fine-tune beats an older LoRA on another base", () => {
  const target = pick([fullFinetune, olderLlamaLora], `${fullFinetune.id}/`);
  assert.equal(target?.id, fullFinetune.id);
});

test("a finished LoRA run beats a newer LoRA on the same base", () => {
  const newerQwenLora = {
    ...olderLlamaLora,
    id: `${OUTPUTS}/unsloth_Qwen3-4B-Instruct-2507_1759200000`,
    baseModel: QWEN,
    updatedAt: 1759200000,
  };
  const finishedQwenLora = {
    ...newerQwenLora,
    id: `${OUTPUTS}/unsloth_Qwen3-4B-Instruct-2507_1759150000`,
    updatedAt: 1759150000,
  };
  const target = pick([newerQwenLora, finishedQwenLora], finishedQwenLora.id);
  assert.equal(target?.id, finishedQwenLora.id);
});

test("an adapter trained on another base model is never picked", () => {
  assert.equal(pick([olderLlamaLora], fullFinetune.id), null);
});

test("without the run's folder, the newest same-base LoRA is picked", () => {
  const qwenLora = {
    ...olderLlamaLora,
    id: `${OUTPUTS}/qwen_lora`,
    baseModel: QWEN,
  };
  assert.equal(
    pick([olderLlamaLora, qwenLora, fullFinetune], null)?.id,
    qwenLora.id,
  );
});

test("a finished full fine-tune loads as a local model, not a download", () => {
  assert.deepEqual(handoff.trainingCompareSelection(fullFinetune), {
    id: fullFinetune.id,
    isLora: false,
    isDownloaded: true,
  });
});
