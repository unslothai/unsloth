// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { modelConfigInstanceKey } from "../src/features/model-picker/model-config/config-signature.ts";

const BASE_CONFIG = {
  customContextLength: null,
  maxSeqLength: null,
  kvCacheDtype: null,
  mlxKvBits: null,
  speculativeType: null,
  specDraftNMax: null,
  nParallel: null,
  nBatch: null,
  nUbatch: null,
  tensorParallel: false,
  chatTemplateOverride: null,
};

test("instance key changes for model, quant, and live custom-argument baseline", () => {
  const base = modelConfigInstanceKey("unsloth/Args-GGUF", "Q4_K_M", null);
  assert.equal(
    base,
    modelConfigInstanceKey("unsloth/Args-GGUF", "Q4_K_M", null),
  );
  assert.notEqual(
    base,
    modelConfigInstanceKey("unsloth/Other-GGUF", "Q4_K_M", null),
  );
  assert.notEqual(
    base,
    modelConfigInstanceKey("unsloth/Args-GGUF", "Q8_0", null),
  );
  assert.notEqual(
    modelConfigInstanceKey(
      "unsloth/Args-GGUF",
      "Q4_K_M",
      BASE_CONFIG,
    ),
    modelConfigInstanceKey("unsloth/Args-GGUF", "Q4_K_M", {
      ...BASE_CONFIG,
      llamaExtraArgs: [],
    }),
  );
});

test("selector keys the page from live loaded config, not ordinary local edits", () => {
  const selector = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    selector,
    /key=\{modelConfigInstanceKey\([\s\S]*?visibleLoadedConfig,[\s\S]*?\)\}/,
  );
  const keyStart = selector.indexOf("key={modelConfigInstanceKey(");
  const keyEnd = selector.indexOf("target={visibleConfigTarget}", keyStart);
  assert.ok(keyStart >= 0 && keyEnd > keyStart);
  assert.doesNotMatch(selector.slice(keyStart, keyEnd), /selectedConfig|initialConfig/);
});

test("editor blocking survives Advanced collapse and reset clears local draft", () => {
  const editor = readFileSync(
    new URL(
      "../src/features/model-picker/components/llama-extra-args-editor.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const page = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-config-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.doesNotMatch(editor, /return \(\) => onBlockingChange\(false\)/);
  assert.match(page, /setLlamaArgsResetGeneration/);
  assert.match(page, /key=\{llamaArgsResetGeneration\}/);
  assert.match(editor, /scrollIntoView\(\{ block: "nearest" \}\)/);
  assert.match(page, /MODEL_OVERRIDE_HYDRATION_MAX_ATTEMPTS/);
  assert.match(page, /rememberGenerationRef\.current \+= 1/);
  assert.match(page, /hydratedOverridesKeyRef\.current = hydrationKey/);
  assert.match(page, /hydratedOverridesKeyRef\.current = null/);
});
