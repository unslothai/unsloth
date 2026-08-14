// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { serializeLlamaExtraArgsRequestBody } from "../src/features/model-picker/model-config/llama-extra-args.ts";

function source(relativePath: string): string {
  return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}

test("normal load, reload, and rollback use the shared omission contract", () => {
  const runtime = source(
    "../src/features/chat/hooks/use-chat-model-runtime.ts",
  );
  assert.match(
    runtime,
    /validateModel\([\s\S]*?llamaExtraArgsPayload\(\s*isGguf \? validateLlamaExtraArgs : undefined/,
  );
  assert.match(
    runtime,
    /loadModel\([\s\S]*?llamaExtraArgsPayload\(\s*isGguf \? loadLlamaExtraArgs : undefined/,
  );
  assert.match(
    runtime,
    /llamaExtraArgsPayload\(\s*stateBeforeUnload\.launchedLlamaExtraArgs/,
  );
  assert.match(
    runtime,
    /launchedLlamaExtraArgs:\s*isGguf && loadLlamaExtraArgs !== undefined[\s\S]*?\[\.\.\.loadLlamaExtraArgs\]/,
  );
});

test("compare validation and load use one argument source", () => {
  const compare = source("../src/features/chat/shared-composer.tsx");
  const calls = compare.match(
    /llamaExtraArgsPayload\(\s*targetIsGguf \? ownConfig\.llamaExtraArgs : undefined/g,
  );
  assert.equal(calls?.length, 2);
});

test("API auto-switch validates and loads the same remembered arguments", () => {
  const adapter = source("../src/features/chat/api/chat-adapter.ts");
  const calls = adapter.match(
    /llamaExtraArgsPayload\(config\.llamaExtraArgs\)/g,
  );
  assert.equal(calls?.length, 2);

  const overrides = source(
    "../src/features/model-picker/api/model-overrides.ts",
  );
  assert.match(
    overrides,
    /Object\.assign\(payload, llamaExtraArgsPayload\(config\.llamaExtraArgs\)\)/,
  );
});

test("GGUF status never treats a shared response as an argument baseline", () => {
  const status = source(
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
  );
  assert.doesNotMatch(status, /llama_extra_args/);
  assert.match(status, /runtimeRevision: status\.runtime_revision \?\? null/);
});

test("Run Settings adopts API-created flags only for a current untouched field", () => {
  const page = source(
    "../src/features/model-picker/components/model-config-page.tsx",
  );
  assert.match(page, /fetchModelOverrides\(\)[\s\S]*?findModelOverride\(/);
  assert.match(
    page,
    /hasLocalLlamaExtraArgs: Object\.hasOwn\([\s\S]*?current,[\s\S]*?"llamaExtraArgs"[\s\S]*?decision\.applyArgs[\s\S]*?llamaExtraArgs: \[\.\.\.serverArgs\]/,
  );
  assert.match(page, /requestRememberGeneration/);
  assert.match(page, /MODEL_OVERRIDE_HYDRATION_MAX_ATTEMPTS/);
});

test("runtime equality keeps absent and explicit-empty distinct", () => {
  const config = source(
    "../src/features/model-picker/model-config/apply-per-model-config.ts",
  );
  assert.match(
    config,
    /JSON\.stringify\(a\.llamaExtraArgs\) === JSON\.stringify\(b\.llamaExtraArgs\)/,
  );
});

test("local override eviction is not serialized as a flag clear", () => {
  const page = source(
    "../src/features/model-picker/components/model-config-page.tsx",
  );
  assert.match(
    page,
    /syncModelOverride\(dropped\.modelId, dropped\.ggufVariant, null, \{\s*keepLaunchFlags: true/,
  );

  const overrides = source(
    "../src/features/model-picker/api/model-overrides.ts",
  );
  assert.match(
    overrides,
    /remove: config === null && !options\?\.keepLaunchFlags/,
  );
  assert.doesNotMatch(overrides, /\{ llama_extra_args: \[\] \}/);
});

test("validateModel preserves the three states in its serialized JSON body", () => {
  const base = { model_path: "unsloth/Args-GGUF" };
  const bodies = [
    JSON.parse(serializeLlamaExtraArgsRequestBody(base, undefined)),
    JSON.parse(serializeLlamaExtraArgsRequestBody(base, [])),
    JSON.parse(
      serializeLlamaExtraArgsRequestBody(base, ["--fit-target", "1024"]),
    ),
  ] as Record<string, unknown>[];
  assert.equal(Object.hasOwn(bodies[0], "llama_extra_args"), false);
  assert.deepEqual(bodies[1].llama_extra_args, []);
  assert.deepEqual(bodies[2].llama_extra_args, ["--fit-target", "1024"]);

  const api = source("../src/features/chat/api/chat-api.ts");
  assert.match(
    api,
    /body: serializeLlamaExtraArgsRequestBody\(\{[\s\S]*?\}, payload\.llama_extra_args\)/,
  );
});

test("loadModel uses the same tri-state serializer as validateModel", () => {
  const api = source("../src/features/chat/api/chat-api.ts");
  assert.equal(
    api.match(/serializeLlamaExtraArgsRequestBody\(/g)?.length,
    2,
    "load and validate calls",
  );
  assert.match(
    api,
    /llama_extra_args: undefined,[\s\S]*?\}, payload\.llama_extra_args\)/,
  );
});

test("catalog failure blocks persistence unless a complete authoritative catalog is cached", () => {
  const editor = source(
    "../src/features/model-picker/components/llama-extra-args-editor.tsx",
  );
  const policy = source(
    "../src/features/model-picker/model-config/llama-extra-args.ts",
  );
  assert.match(editor, /cachedLlamaServerManagedPolicy\(\)/);
  assert.match(editor, /cachedAuthoritativeLlamaServerArguments\(\)/);
  assert.match(editor, /llamaExtraArgsCatalogBlocksPersistence\(/);
  assert.match(editor, /Studio ignores inherited LLAMA_ARG_\*/);
  assert.doesNotMatch(policy, /EMERGENCY_MANAGED_LLAMA_FLAG_GROUPS/);
});
