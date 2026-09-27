// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { ActiveModelConfigState } from "../src/features/model-picker/hooks/use-active-model-config.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function activeConfig(patch: Record<string, unknown> = {}) {
  const state = {
    params: { checkpoint: "unsloth/Qwen3-0.6B-GGUF", maxSeqLength: 2048 },
    reasoningBudget: 32,
    reasoningBudgetMessage: "Conclude now.",
    loadedReasoningBudget: 32,
    loadedReasoningBudgetMessage: "Conclude now.",
    loadedReasoningBudgetRequested: -1,
    loadedReasoningBudgetMessageRequested: "",
    ...patch,
  };
  const { useActiveModelConfig } = loadWithStubs<{
    useActiveModelConfig: () => ActiveModelConfigState;
  }>(
    new URL(
      "../src/features/model-picker/hooks/use-active-model-config.ts",
      import.meta.url,
    ),
    {
      "@/features/chat": {
        isExternalModelId: () => false,
        useChatRuntimeStore: (select: (value: typeof state) => unknown) =>
          select(state),
      },
      "@/config/env": { usePlatformStore: () => ({ deviceType: "cpu" }) },
      react: { useMemo: (factory: () => unknown) => factory() },
      "../model-config/per-model-config": {
        isServedByLlamaCpp: () => true,
        residentIsServedByMlx: () => false,
      },
    },
  );
  const config = useActiveModelConfig().config!;
  const { currentRuntimePerModelConfig } = loadWithStubs<{
    currentRuntimePerModelConfig: () => NonNullable<
      ActiveModelConfigState["config"]
    >;
  }>(
    new URL(
      "../src/features/model-picker/model-config/apply-per-model-config.ts",
      import.meta.url,
    ),
    {
      "@/features/chat/stores/chat-runtime-store": {
        useChatRuntimeStore: { getState: () => state },
        normalizeSpeculativeType: (value: unknown) => value ?? null,
      },
      "@/features/chat/presets/preset-policy": {},
      "./config-signature": {},
      "./per-model-config": {},
    },
  );
  const snapshot = currentRuntimePerModelConfig();
  assert.equal(snapshot.reasoningBudget, config.reasoningBudget);
  assert.equal(snapshot.reasoningBudgetMessage, config.reasoningBudgetMessage);
  return config;
}

test("unrelated edits preserve inherited reasoning launch intent", () => {
  const config = activeConfig({ nBatch: 256 });
  assert.equal(config.reasoningBudget, -1);
  assert.equal(config.reasoningBudgetMessage, "");
  assert.equal(config.nBatch, 256);
});

test("an edited reasoning control overrides only its own inherited value", () => {
  const budget = activeConfig({ reasoningBudget: 0 });
  assert.equal(budget.reasoningBudget, 0);
  assert.equal(budget.reasoningBudgetMessage, "");
  const message = activeConfig({ reasoningBudgetMessage: "Finish." });
  assert.equal(message.reasoningBudget, -1);
  assert.equal(message.reasoningBudgetMessage, "Finish.");
});

test("explicit and legacy reasoning values remain available", () => {
  assert.equal(
    activeConfig({ loadedReasoningBudgetRequested: 32 }).reasoningBudget,
    32,
  );
  const legacy = activeConfig({
    loadedReasoningBudgetRequested: null,
    loadedReasoningBudgetMessageRequested: null,
  });
  assert.equal(legacy.reasoningBudget, 32);
  assert.equal(legacy.reasoningBudgetMessage, "Conclude now.");
});
