// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerStoreStubResolver,
  readSrc,
} from "./helpers/kit.ts";

registerStoreStubResolver();
const { store } = installLocalStorageFake();
const {
  normalizeLlamaCppConfig,
  customConfigSections,
  toggledLlamaCppConfig,
  llamaCppConfigPayload,
} =
  await import("../src/features/model-picker/model-config/llama-cpp-config.ts");
const {
  DEFAULT_PER_MODEL_CONFIG,
  PER_MODEL_CONFIG_STORAGE_KEY,
  savePerModelConfig,
  resolveInitialConfig,
} =
  await import("../src/features/model-picker/model-config/per-model-config.ts");
const { fromApiOverride, toApiOverride, resolveStoredOverride } =
  await import("../src/features/model-picker/api/model-overrides.ts");
const custom = {
  version: 1,
  mode: "custom",
  ini: "[*]\nfit=off\n[my-model]\nctx-size=56000\ntemp=0\n",
  section: "my-model",
} as const;
const managed = { version: 1, mode: "managed" } as const;

test("switching back to custom restores the last source instead of a blank one", () => {
  assert.deepEqual(toggledLlamaCppConfig(custom, null), managed);
  assert.deepEqual(
    toggledLlamaCppConfig(managed, {
      ini: custom.ini,
      section: custom.section,
    }),
    custom,
  );
  assert.deepEqual(toggledLlamaCppConfig(undefined, null), {
    version: 1,
    mode: "custom",
    ini: "[*]\n",
    section: null,
  });
  const editor = readSrc(
    "features/model-picker/components/custom-llama-config-editor.tsx",
  );
  // A refused load remounts the editor, so the remembered source has to live outside it.
  assert.match(editor, /^const lastCustomSource = new Map</m);
  assert.match(
    editor,
    /toggledLlamaCppConfig\(\s*value,\s*lastCustomSource\.get\(sourceKey\) \?\? null,?\s*\)/,
  );
  assert.match(
    editor,
    /if \(active\) lastCustomSource\.set\(sourceKey, \{ ini, section \}\)/,
  );
});

test("custom source and selection round-trip through storage and API without consuming legacy extras", () => {
  store.clear();
  const config = {
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaExtraArgs: ["--threads", "3"],
    llamaCppConfig: custom,
  };
  assert.equal(savePerModelConfig("org/model", "Q4", config), true);
  const read = resolveInitialConfig("org/model", "Q4").config;
  assert.deepEqual(read.llamaCppConfig, custom);
  assert.deepEqual(
    fromApiOverride(toApiOverride(read), DEFAULT_PER_MODEL_CONFIG)
      .llamaCppConfig,
    custom,
  );
  assert.deepEqual(read.llamaExtraArgs, ["--threads", "3"]);
  const records = JSON.parse(store.get(PER_MODEL_CONFIG_STORAGE_KEY)!);
  assert.equal(
    Object.values(records).some(
      (row: unknown) => (row as { version: number }).version === 8,
    ),
    true,
  );
  assert.equal(
    savePerModelConfig("org/model", "Q4", { ...read, llamaCppConfig: managed }),
    true,
  );
  const reset = resolveInitialConfig("org/model", "Q4").config;
  assert.deepEqual(reset.llamaCppConfig, managed);
  assert.deepEqual(reset.llamaExtraArgs, ["--threads", "3"]);
});

test("a per-quant managed tombstone blocks bare-model custom fallback locally and on the mirror", () => {
  store.clear();
  savePerModelConfig("org/model", null, {
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaCppConfig: custom,
  });
  savePerModelConfig("org/model", "Q4", {
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaCppConfig: managed,
  });
  assert.deepEqual(
    resolveInitialConfig("org/model", "Q4").config.llamaCppConfig,
    managed,
  );
  assert.deepEqual(
    resolveStoredOverride(
      {
        "org/model": { llama_cpp_config: custom },
        "org/model:Q4": { llama_cpp_config: managed },
      },
      ["org/model:Q4", "org/model"],
    )?.llama_cpp_config,
    managed,
  );
});

test("custom configuration advances only its own storage tier", () => {
  const storedVersion = () => {
    const records = JSON.parse(store.get(PER_MODEL_CONFIG_STORAGE_KEY) ?? "{}");
    return (Object.values(records)[0] as { version?: number } | undefined)
      ?.version;
  };

  store.clear();
  assert.equal(
    savePerModelConfig("org/reasoning", "Q4", {
      ...DEFAULT_PER_MODEL_CONFIG,
      reasoningBudget: 256,
      reasoningBudgetMessage: "Keep the proof short",
    }),
    true,
  );
  assert.equal(storedVersion(), 6);
  const reasoning = resolveInitialConfig("org/reasoning", "Q4").config;
  assert.equal(reasoning.reasoningBudget, 256);
  assert.equal(reasoning.reasoningBudgetMessage, "Keep the proof short");
  assert.equal(reasoning.llamaCppConfig, undefined);

  store.clear();
  assert.equal(
    savePerModelConfig("org/custom", "Q4", {
      ...DEFAULT_PER_MODEL_CONFIG,
      reasoningBudget: 256,
      reasoningBudgetMessage: "Keep the proof short",
      llamaCppConfig: custom,
    }),
    true,
  );
  assert.equal(storedVersion(), 8);
  const customAndReasoning = resolveInitialConfig("org/custom", "Q4").config;
  assert.equal(customAndReasoning.reasoningBudget, 256);
  assert.equal(
    customAndReasoning.reasoningBudgetMessage,
    "Keep the proof short",
  );
  assert.deepEqual(customAndReasoning.llamaCppConfig, custom);

  store.clear();
  assert.equal(
    savePerModelConfig("org/tuning", "Q4", {
      ...DEFAULT_PER_MODEL_CONFIG,
      loadMode: "mmap",
    }),
    true,
  );
  assert.equal(storedVersion(), 5);
  assert.equal(
    resolveInitialConfig("org/tuning", "Q4").config.loadMode,
    "mmap",
  );
});

test("absent and explicit reset remain different request values", () => {
  assert.deepEqual(llamaCppConfigPayload(undefined), {});
  assert.deepEqual(llamaCppConfigPayload(managed), {
    llama_cpp_config: managed,
  });
  assert.equal(
    "llama_cpp_config" in toApiOverride(DEFAULT_PER_MODEL_CONFIG),
    false,
  );
  assert.deepEqual(
    fromApiOverride({}, { ...DEFAULT_PER_MODEL_CONFIG, llamaCppConfig: custom })
      .llamaCppConfig,
    custom,
  );
});

test("UTF-8 cap rejects an oversized edit without erasing its saved predecessor", () => {
  store.clear();
  savePerModelConfig("org/model", "Q4", {
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaCppConfig: custom,
  });
  const oversized = { ...custom, ini: "é".repeat(32769) };
  assert.equal(normalizeLlamaCppConfig(oversized), undefined);
  assert.equal(
    savePerModelConfig("org/model", "Q4", {
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaCppConfig: oversized,
    }),
    false,
  );
  assert.deepEqual(
    resolveInitialConfig("org/model", "Q4").config.llamaCppConfig,
    custom,
  );
  assert.equal(
    normalizeLlamaCppConfig({ ...custom, ini: "é".repeat(32768) })?.mode,
    "custom",
  );
});

test("blank custom source is rejected without erasing its saved predecessor", () => {
  store.clear();
  savePerModelConfig("org/model", "Q4", {
    ...DEFAULT_PER_MODEL_CONFIG,
    llamaCppConfig: custom,
  });
  const blank = { ...custom, ini: " \n\t " };
  assert.equal(normalizeLlamaCppConfig(blank), undefined);
  assert.equal(
    savePerModelConfig("org/model", "Q4", {
      ...DEFAULT_PER_MODEL_CONFIG,
      llamaCppConfig: blank,
    }),
    false,
  );
  assert.deepEqual(
    resolveInitialConfig("org/model", "Q4").config.llamaCppConfig,
    custom,
  );
});

test("selector suggestions never implicitly select a sole named section", () => {
  assert.deepEqual(customConfigSections("[*]\n[only]\nctx-size=56000"), [
    "only",
  ]);
  assert.deepEqual(customConfigSections("[ preset]\nctx-size=56000"), [
    "preset",
  ]);
  // llama.cpp files keys above the first header under "default", not under [*].
  assert.deepEqual(customConfigSections("c=1\n[*]\nngl=-1\n[large]\nc=9"), [
    "default",
    "large",
  ]);
  assert.deepEqual(customConfigSections("; note\n[large]\nc=9"), ["large"]);
  assert.deepEqual(customConfigSections("[a]\r\nc=1\r\n[b] ; x\r\nc=2\r\n"), [
    "a",
    "b",
  ]);
  assert.deepEqual(
    normalizeLlamaCppConfig({ ...custom, section: null })?.mode,
    "custom",
  );
  // Backend validates null against the source; the frontend does not invent a selection.
  assert.equal(
    (
      llamaCppConfigPayload({ ...custom, section: null })
        .llama_cpp_config as typeof custom
    ).section,
    null,
  );
});

test("large per-model sources still obey the aggregate storage budget", () => {
  store.clear();
  const evicted: { modelId: string; ggufVariant: string | null }[] = [];
  for (let index = 0; index < 22; index += 1) {
    assert.equal(
      savePerModelConfig(
        `org/model-${index}`,
        null,
        {
          ...DEFAULT_PER_MODEL_CONFIG,
          llamaCppConfig: { ...custom, ini: "#".repeat(60_000) },
        },
        evicted,
      ),
      true,
    );
  }
  assert.ok(evicted.length > 0);
  assert.ok(
    new TextEncoder().encode(store.get(PER_MODEL_CONFIG_STORAGE_KEY)!).length <=
      1024 * 1024,
  );
  assert.equal(
    resolveInitialConfig("org/model-21", null).config.llamaCppConfig?.mode,
    "custom",
  );
});

test("a diffusion load sends managed in place of a custom config, never omits it", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.equal(
    runtime.match(
      /llamaCppConfigPayload\(loadLlamaCppConfig, \{\s*isDiffusion: targetIsDiffusion/g,
    )?.length,
    2,
    "validate and load both reset a diffusion target explicitly",
  );
  const compare = readSrc("features/chat/shared-composer.tsx");
  assert.equal(
    compare.match(
      /llamaCppConfigPayload\(ownConfig\.llamaCppConfig, \{\s*isDiffusion: resolvedIsDiffusion === true/g,
    )?.length,
    2,
    "compare validate and load reset a diffusion target too",
  );
  // Both GGUF autoload paths record what launched, including the default-model fallback.
  assert.equal(
    readSrc("features/chat/api/chat-adapter.ts").split(
      "...loadedLlamaCppConfigFields(loadResp",
    ).length - 1,
    2,
  );
  assert.deepEqual(llamaCppConfigPayload(custom, { isDiffusion: true }), {
    llama_cpp_config: managed,
  });
  assert.deepEqual(llamaCppConfigPayload(custom), { llama_cpp_config: custom });
  assert.deepEqual(llamaCppConfigPayload(managed, { isDiffusion: true }), {
    llama_cpp_config: managed,
  });
  // Unknown config still resets, or the backend inherits a saved custom source.
  assert.deepEqual(llamaCppConfigPayload(undefined, { isDiffusion: true }), {
    llama_cpp_config: managed,
  });
});

test("section suggestions trim padding the way the server does", () => {
  assert.deepEqual(customConfigSections("[*]\n[ fast ]\nc=1\n[fast]\n"), [
    "fast",
  ]);
});

test("a custom config locks the managed rows and keeps the editor outside them", () => {
  const page = readSrc(
    "features/model-picker/components/model-config-page.tsx",
  );
  const fieldsetEnd = page.indexOf("</fieldset>");
  assert.ok(page.indexOf("disabled={customActive}") < fieldsetEnd);
  assert.ok(page.indexOf("<CustomLlamaConfigEditor") > fieldsetEnd);
  assert.match(page, /!customActive &&\s*shouldRequestMemoryEstimate/);
  // Custom launches honour disable_vision, so its switch stays outside the lock.
  assert.ok(page.indexOf("hideVision={customActive}") < fieldsetEnd);
  // Hidden behind Advanced settings unless custom mode is already on.
  assert.match(
    page,
    /!resolvedIsDiffusion &&\s*!audioRuntimeGguf &&\s*\(showAdvanced \|\| customActive\)/,
  );
  assert.ok(page.indexOf("{customActive && <VisionRow") > fieldsetEnd);
});
