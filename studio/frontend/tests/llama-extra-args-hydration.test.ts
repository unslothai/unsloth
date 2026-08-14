// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  activeLlamaArgumentsHydrationMatches,
  currentEffectiveLlamaLoadIdentity,
} from "../src/features/model-picker/model-config/active-arguments-hydration.ts";

const RESPONSE = {
  model_identifier: "unsloth/Args-GGUF",
  gguf_variant: "Q4_K_M",
  runtime_revision: "runtime-2",
  llama_extra_args: [],
};
const IDENTITY = {
  effectiveLoadIdentifier: "unsloth/Args-GGUF",
  ggufVariant: "q4_k_m",
  runtimeRevision: "runtime-2",
};
const ACTIVE_ARGUMENTS_ENDPOINT =
  /\/api\/inference\/llama-server\/active-arguments/;
const HYDRATION_MATCH_CALL = /activeLlamaArgumentsHydrationMatches\(/;
const LATE_EDIT_GUARD =
  /requestEditGeneration !== llamaArgsEditGenerationRef\.current/;
const HYDRATED_ARGS_FIELD = /llamaExtraArgs: \[\.\.\.hydratedArgs\]/;
const HYDRATED_BASELINE_FIELD = /loadedLlamaExtraArgs: \[\.\.\.hydratedArgs\]/;
const SHARED_ARGS_FIELD = /llama_extra_args/;
const RUNTIME_REVISION_FIELD = /runtime_revision\?: string \| null/;
const LEGACY_RESPONSE_READER = /readLlamaExtraArgsResponse/;
const SHARED_RESPONSE_ARGS_ACCESS =
  /(?:status|loadResp|loadResponse|resp)\.llama_extra_args/;

function source(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}

test("active hydration accepts an exact model, normalized variant, and revision", () => {
  assert.equal(
    activeLlamaArgumentsHydrationMatches(RESPONSE, IDENTITY, IDENTITY),
    true,
  );
  assert.deepEqual(RESPONSE.llama_extra_args, []);
});

test("active hydration rejects stale model, variant, and runtime races", () => {
  for (const current of [
    { ...IDENTITY, effectiveLoadIdentifier: "unsloth/Other-GGUF" },
    { ...IDENTITY, ggufVariant: "Q8_0" },
    { ...IDENTITY, runtimeRevision: "runtime-3" },
  ]) {
    assert.equal(
      activeLlamaArgumentsHydrationMatches(RESPONSE, IDENTITY, current),
      false,
    );
  }
  assert.equal(
    activeLlamaArgumentsHydrationMatches(
      { ...RESPONSE, runtime_revision: "runtime-1" },
      IDENTITY,
      IDENTITY,
    ),
    false,
  );
  assert.equal(
    activeLlamaArgumentsHydrationMatches(
      { ...RESPONSE, runtime_revision: null },
      { ...IDENTITY, runtimeRevision: null },
      { ...IDENTITY, runtimeRevision: null },
    ),
    false,
  );
});

test("effective load identity stays separate from the public persistence id", () => {
  const snapshotPath = "C:/cache/models--unsloth--Args-GGUF/snapshots/abc";
  assert.equal(
    currentEffectiveLlamaLoadIdentity({
      activeLoadId: snapshotPath,
      residentCheckpoint: "unsloth/Args-GGUF",
      selectedCheckpoint: "unsloth/Args-GGUF",
    }),
    snapshotPath,
  );
  assert.equal(
    activeLlamaArgumentsHydrationMatches(
      {
        ...RESPONSE,
        effective_load_identifier: snapshotPath,
        model_identifier: "unsloth/Args-GGUF",
      },
      { ...IDENTITY, effectiveLoadIdentifier: snapshotPath },
      { ...IDENTITY, effectiveLoadIdentifier: snapshotPath },
    ),
    true,
  );
});

test("editor uses only the UI hydration endpoint and guards late edits", () => {
  const api = source(
    "../src/features/model-picker/api/llama-server-arguments.ts",
  );
  const page = source(
    "../src/features/model-picker/components/model-config-page.tsx",
  );
  assert.match(api, ACTIVE_ARGUMENTS_ENDPOINT);
  assert.match(page, HYDRATION_MATCH_CALL);
  assert.match(page, LATE_EDIT_GUARD);
  assert.match(page, HYDRATED_ARGS_FIELD);
  assert.match(page, HYDRATED_BASELINE_FIELD);
  assert.match(
    page,
    /effectiveLoadIdentifier: target\.meta\.loadId \?\? target\.id/,
  );
});

test("shared load and status contracts neither disclose nor consume arguments", () => {
  const types = source("../src/features/chat/types/api.ts");
  for (const interfaceName of [
    "LoadModelResponse",
    "InferenceStatusResponse",
  ]) {
    const start = types.indexOf(`export interface ${interfaceName}`);
    const end = types.indexOf("\n}\n", start);
    assert.ok(start >= 0 && end > start);
    const contract = types.slice(start, end);
    assert.doesNotMatch(contract, SHARED_ARGS_FIELD);
    assert.match(contract, RUNTIME_REVISION_FIELD);
  }

  for (const path of [
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
    "../src/features/chat/api/chat-adapter.ts",
    "../src/features/chat/shared-composer.tsx",
  ]) {
    const contents = source(path);
    assert.doesNotMatch(contents, LEGACY_RESPONSE_READER);
    assert.doesNotMatch(contents, SHARED_RESPONSE_ARGS_ACCESS);
  }
});
