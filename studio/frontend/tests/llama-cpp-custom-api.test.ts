// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

// Execute the shipped API functions with a captured transport, without mounting UI barrels.
const source = readSrc("features/chat/api/chat-api.ts");
const load = source.slice(
  source.indexOf("export async function loadModel("),
  source.indexOf("export async function countChatInputTokens("),
);
const validateStart = source.indexOf("export async function validateModel(");
const validate = source.slice(
  validateStart,
  source.indexOf("/** Read a GGUF's header", validateStart),
);
const compiled = ts.transpileModule(`${load}\n${validate}`, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const custom = {
  version: 1,
  mode: "custom",
  ini: "[*]\ntemp=0\n",
  section: null,
};
const summary = {
  mode: "custom",
  section: null,
  digest: "test-digest",
  tuning: {},
  request_defaults: { temperature: 0 },
  diagnostics: [],
};
const request = {
  model_path: "model.gguf",
  hf_token: null,
  max_seq_length: 4096,
  load_in_4bit: false,
  is_lora: false,
  llama_cpp_config: custom,
};

function harness(validateResponse: unknown, loadResponse: unknown) {
  const calls: { url: string; body: Record<string, unknown> }[] = [];
  const grants: { token: string; operation: string }[] = [];
  const context = {
    exports: {} as {
      validateModel: (request: unknown) => Promise<unknown>;
      loadModel: (request: unknown) => Promise<unknown>;
    },
    prepareHfTokenForUse: async () => ({ proceed: true, token: null }),
    consumeNativePathToken: async (token: string, operation: string) => {
      grants.push({ token, operation });
      return { nativePathLease: `${token}:${operation}:fresh-grant` };
    },
    authFetch: async (url: string, init: RequestInit) => {
      calls.push({ url, body: JSON.parse(String(init.body)) });
      return url.endsWith("/validate") ? validateResponse : loadResponse;
    },
    parseJsonOrThrow: async (response: unknown) => response,
    withModelLoadNotice: async (
      _runtime: string,
      _model: string,
      run: () => Promise<unknown>,
    ) => run(),
    showCarveoutAdvice: () => {},
  };
  vm.runInNewContext(compiled, context);
  return { calls, grants, ...context.exports };
}

test("custom validation sends the exact source and rejects a server that silently ignores it", async () => {
  const api = harness({ valid: true, message: "Found" }, {});
  await assert.rejects(api.validateModel(request), /cannot validate custom/);
  assert.deepEqual(api.calls[0].body.llama_cpp_config, custom);
});

test("a direct custom load validates support before any load mutation", async () => {
  const api = harness({ valid: true }, { status: "loaded" });
  await assert.rejects(api.loadModel(request), /cannot validate custom/);
  assert.deepEqual(
    api.calls.map((call) => call.url),
    ["/api/inference/validate"],
  );
});

test("a successful custom response retains the backend summary and raw source", async () => {
  const response = {
    status: "loaded",
    requested_llama_cpp_config: custom,
    llama_cpp_config_summary: summary,
  };
  const api = harness(
    { valid: true, llama_cpp_config_summary: summary },
    response,
  );
  assert.deepEqual(await api.loadModel(request), response);
  assert.deepEqual(api.calls[1].body.llama_cpp_config, custom);
});

test("custom load refuses an unconfirmed success after preflight", async () => {
  const api = harness(
    { valid: true, llama_cpp_config_summary: summary },
    { status: "loaded" },
  );
  await assert.rejects(api.loadModel(request), /did not confirm the custom/);
});

test("managed calls keep the ordinary one-request contract", async () => {
  const api = harness({ valid: true }, { status: "loaded" });
  const { llama_cpp_config: _ignored, ...managed } = request;
  await api.loadModel(managed);
  assert.deepEqual(
    api.calls.map((call) => call.url),
    ["/api/inference/load"],
  );
  assert.equal("llama_cpp_config" in api.calls[0].body, false);
});

test("native custom loads mint validation grants without reusing the load grant", async () => {
  for (const token of ["selected-file", "previous-file-for-rollback"]) {
    const api = harness(
      { valid: true, llama_cpp_config_summary: summary },
      { status: "loaded", llama_cpp_config_summary: summary },
    );
    await api.loadModel({
      ...request,
      nativePathToken: token,
      nativePathLease: `${token}:load-model:original-grant`,
    });
    assert.deepEqual(api.grants, [{ token, operation: "validate-model" }]);
    assert.equal(
      api.calls[0].body.native_path_lease,
      `${token}:validate-model:fresh-grant`,
    );
    assert.equal(
      api.calls[1].body.native_path_lease,
      `${token}:load-model:original-grant`,
    );
    for (const call of api.calls) {
      assert.equal("nativePathToken" in call.body, false);
      assert.equal("nativePathLease" in call.body, false);
    }
  }
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(
    runtime,
    /nativePathLease: loadNativePathLease,\s*nativePathToken,/,
  );
  assert.match(
    runtime,
    /nativePathLease: rollbackNativePathLease,\s*nativePathToken: previousActiveNativePathToken,/,
  );
});

test("a native custom load without a redeemable token never reuses its load grant", async () => {
  const api = harness(
    { valid: true, llama_cpp_config_summary: summary },
    { status: "loaded", llama_cpp_config_summary: summary },
  );
  await assert.rejects(
    api.loadModel({ ...request, nativePathLease: "load-only-grant" }),
    /re-select the local model file/,
  );
  assert.equal(api.calls.length, 0);
  assert.equal(api.grants.length, 0);
});
