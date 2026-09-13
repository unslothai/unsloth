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
  llamaCppConfigPayload,
  customSamplingPayload,
  markSamplingFields,
  explicitSamplingFields,
} = await import(
  "../src/features/model-picker/model-config/llama-cpp-config.ts"
);
const {
  DEFAULT_PER_MODEL_CONFIG,
  PER_MODEL_CONFIG_STORAGE_KEY,
  savePerModelConfig,
  resolveInitialConfig,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);
const { fromApiOverride, toApiOverride, resolveStoredOverride } = await import(
  "../src/features/model-picker/api/model-overrides.ts"
);
const { getReplayedParams, pickRememberedParams } = await import(
  "../src/features/chat/lib/per-model-params.ts"
);
const { DEFAULT_INFERENCE_PARAMS } = await import(
  "../src/features/chat/types/runtime.ts"
);
const { snapshotQueuedChatRunSettings } = await import(
  "../src/features/chat/utils/queued-chat-run-settings.ts"
);
const custom = {
  version: 1,
  mode: "custom",
  ini: "[*]\nfit=off\n[my-model]\nctx-size=56000\ntemp=0\n",
  section: "my-model",
} as const;
const managed = { version: 1, mode: "managed" } as const;

test("removing a selected section clears the editor's stale selection", async () => {
  const ts = await import("typescript");
  const source = ts.createSourceFile(
    "editor.tsx",
    readSrc("features/model-picker/components/custom-llama-config-editor.tsx"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let callback = "";
  function visit(node: import("typescript").Node) {
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(source) === "textarea") {
      const change = node.attributes.properties.find(
        (attr) => ts.isJsxAttribute(attr) && attr.name.getText(source) === "onChange",
      );
      if (change && ts.isJsxAttribute(change) && change.initializer && ts.isJsxExpression(change.initializer)) {
        callback = change.initializer.expression?.getText(source) ?? "";
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.ok(callback);
  let changed: typeof custom | undefined;
  const onChange = new Function("value", "onChange", "customConfigSections", `return (${callback})`)(
    custom,
    (next: typeof custom) => { changed = next; },
    customConfigSections,
  );
  onChange({ target: { value: "[*]\nnp=1" } });
  assert.equal(changed?.section, null);
  onChange({ target: { value: custom.ini } });
  assert.equal(changed?.section, "my-model");
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
      (row: unknown) => (row as { version: number }).version === 6,
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

test("selector suggestions never implicitly select a sole named section", () => {
  assert.deepEqual(customConfigSections("[*]\n[only]\nctx-size=56000"), [
    "only",
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

test("automatic sampling yields to the preset; explicit equal values, zero and false survive", () => {
  assert.deepEqual(customSamplingPayload(custom, []), {
    sampling_fields_explicit: [],
  });
  const params = markSamplingFields(
    { ...DEFAULT_INFERENCE_PARAMS, temperature: 0 },
    "temperature",
    "enable_thinking",
  );
  assert.deepEqual(
    customSamplingPayload(custom, params.samplingFieldsExplicit),
    { sampling_fields_explicit: ["temperature", "enable_thinking"] },
  );
  assert.deepEqual(
    explicitSamplingFields({ temperature: 0, reasoningEnabled: false }),
    ["temperature", "enable_thinking"],
  );
  assert.deepEqual(customSamplingPayload(managed, []), {});
});

test("model memory and queued snapshots retain provenance separately from automatic values", () => {
  const params = markSamplingFields(
    { ...DEFAULT_INFERENCE_PARAMS, checkpoint: "org/model" },
    "temperature",
  );
  const remembered = pickRememberedParams(params);
  const replayed = getReplayedParams(
    true,
    { "org/model": remembered },
    { ...DEFAULT_INFERENCE_PARAMS },
    "org/model",
    true,
  );
  assert.deepEqual(replayed.samplingFieldsExplicit, ["temperature"]);
  const legacy = getReplayedParams(
    true,
    { "org/model": { temperature: 0, topP: 0.95 } },
    { ...DEFAULT_INFERENCE_PARAMS },
    "org/model",
    true,
  );
  assert.deepEqual(legacy.samplingFieldsExplicit, ["temperature", "top_p"]);
  const queued = snapshotQueuedChatRunSettings({
    params,
    loadedLlamaCppConfig: custom,
    llamaCppConfig: custom,
  } as never);
  const next = markSamplingFields(params, "top_p");
  assert.deepEqual(queued.params.samplingFieldsExplicit, ["temperature"]);
  assert.deepEqual(next.samplingFieldsExplicit, ["temperature", "top_p"]);
  assert.deepEqual(queued.loadedLlamaCppConfig, custom);
});

test("all ordinary load, preflight and estimate producers carry the config; rollback uses its resident source", () => {
  assert.equal(
    readSrc("features/chat/hooks/use-chat-model-runtime.ts").match(
      /llamaCppConfigPayload\(loadLlamaCppConfig\)/g,
    )?.length,
    2,
  );
  assert.match(
    readSrc("features/chat/api/chat-api.ts"),
    /llama_cpp_config: payload\.llama_cpp_config/,
  );
  assert.match(
    readSrc("features/model-picker/api/memory-estimate.ts"),
    /llama_cpp_config: payload\.llamaCppConfig/,
  );
  assert.match(
    readSrc("features/chat/shared-composer.tsx"),
    /llamaCppConfigPayload\(ownConfig\.llamaCppConfig\)/,
  );
  assert.match(
    readSrc("features/chat/api/chat-adapter.ts"),
    /llamaCppConfigPayload\(config\.llamaCppConfig\)/,
  );
  assert.match(
    readSrc("features/chat/hooks/use-chat-model-runtime.ts"),
    /llamaCppConfigPayload\(\s*stateBeforeUnload\.loadedLlamaCppConfig/,
  );
  assert.match(
    readSrc("features/chat/lib/apply-inference-status-to-store.ts"),
    /loadedLlamaCppConfig: status\.requested_llama_cpp_config/,
  );
});
