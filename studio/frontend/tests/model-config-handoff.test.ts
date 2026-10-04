// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ModelConfigHandoffRequest } from "../src/features/model-picker/model-config/model-config-handoff.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
  modelConfigHandoffForDestination,
  modelConfigTarget,
  modelConfigTargetIsResident,
  modelConfigTargetMatchesSelection,
  requestModelConfigHandoff,
  residentModelConfigTarget,
  useModelConfigHandoffStore,
} = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);
const { DEFAULT_PER_MODEL_CONFIG, resolveInitialConfig, savePerModelConfig } =
  await import("../src/features/model-picker/model-config/per-model-config.ts");

const FALLBACK_REQUEST_ID_RE = /^\d+-[a-z0-9]+$/;

function request(
  requestId: string,
  id: string,
  loadId: string,
): ModelConfigHandoffRequest {
  return {
    requestId,
    id,
    meta: {
      source: "hub",
      isLora: false,
      isDownloaded: true,
      loadId,
    },
  };
}

test("handoff request IDs work without native randomUUID", () => {
  assert.match(createModelConfigHandoffRequestId(null), FALLBACK_REQUEST_ID_RE);
  assert.match(
    createModelConfigHandoffRequestId({
      randomUUID: () => {
        throw new Error("unavailable");
      },
    }),
    FALLBACK_REQUEST_ID_RE,
  );
  assert.equal(
    createModelConfigHandoffRequestId({ randomUUID: () => "native-id" }),
    "native-id",
  );
});

test("only the matching consumer can clear the latest handoff", () => {
  const first = request("first", "org/first", "/cache/first");
  const second = request("second", "org/second", "/cache/second");

  requestModelConfigHandoff(first);
  requestModelConfigHandoff(second);
  clearModelConfigHandoff(first.requestId);
  assert.equal(useModelConfigHandoffStore.getState().request, second);

  clearModelConfigHandoff(second.requestId);
  assert.equal(useModelConfigHandoffStore.getState().request, null);
});

test("only the matching unoccupied active Chat destination receives a handoff", () => {
  const pending = request("run-request", "org/model", "/cache/model");
  assert.equal(
    modelConfigHandoffForDestination(pending, {
      active: true,
      newChatId: "run-request",
    }),
    pending,
  );

  const blockedDestinations = [
    { active: false, newChatId: "run-request" },
    { active: true, newChatId: "another-request" },
    { active: true, newChatId: "run-request", threadId: "thread" },
    { active: true, newChatId: "run-request", compareId: "compare" },
    { active: true, newChatId: "run-request", projectId: "project" },
  ];
  for (const destination of blockedDestinations) {
    assert.equal(modelConfigHandoffForDestination(pending, destination), null);
  }
  assert.equal(
    modelConfigHandoffForDestination(null, {
      active: true,
      newChatId: "run-request",
    }),
    null,
  );
});

test("a Windows model path stays opaque while its display name is portable", () => {
  const id = String.raw`C:\Users\models\Llama-3.gguf`;
  const meta = {
    source: "local" as const,
    isLora: false,
    isDownloaded: true,
    isGguf: true,
    loadId: id,
  };

  const target = modelConfigTarget(id, meta);

  assert.equal(target.id, id);
  assert.equal(target.configId, undefined);
  assert.equal(target.meta.loadId, id);
  assert.equal(target.displayName, "Llama-3");
  assert.equal(target.isGguf, true);
  assert.equal(target.apiLoadable, true);
});

test("a GGUF handoff preserves its public identity and exact variant", () => {
  const id = "unsloth/Llama-GGUF";
  const loadId = "/cache/models--unsloth--Llama-GGUF/snapshots/revision";
  const target = modelConfigTarget(id, {
    source: "hub",
    isLora: false,
    loadId,
    ggufVariant: "Q4_K_M",
    ggufFilename: "llama-q4_k_m.gguf",
    isDownloaded: true,
    isGguf: true,
  });

  assert.equal(target.id, loadId);
  assert.equal(target.configId, id);
  assert.equal(target.displayName, "Llama-GGUF · Q4_K_M");
  assert.equal(target.ggufVariant, "Q4_K_M");
  assert.equal(target.meta.ggufFilename, "llama-q4_k_m.gguf");
});

test("configuration residency follows public, loader, variant, and live state", () => {
  const target = modelConfigTarget("Org/Model", {
    source: "hub",
    isLora: false,
    loadId: String.raw`C:\Users\models\snapshots\revision`,
    ggufVariant: "Q4_K_M",
    isDownloaded: true,
    isGguf: true,
  });

  assert.equal(modelConfigTargetMatchesSelection(target, "org/model"), true);
  assert.equal(
    modelConfigTargetMatchesSelection(
      target,
      String.raw`c:\users\models\snapshots\revision`,
    ),
    true,
  );
  assert.equal(
    modelConfigTargetIsResident({
      target,
      selectedId: "org/model",
      activeGgufVariant: "q4_k_m",
      loaded: true,
    }),
    true,
  );
  assert.equal(
    modelConfigTargetIsResident({
      target,
      selectedId: "org/model",
      activeGgufVariant: "Q8_0",
      loaded: true,
    }),
    false,
  );
  assert.equal(
    modelConfigTargetIsResident({
      target,
      selectedId: "org/model",
      activeGgufVariant: "Q4_K_M",
      loaded: false,
    }),
    false,
  );
});

test("only local GGUF files may ignore a derived active variant", () => {
  const localPath = String.raw`C:\Models\Llama.gguf`;
  const localTarget = modelConfigTarget(localPath, {
    source: "local",
    isLora: false,
    loadId: localPath,
    isDownloaded: true,
    isGguf: true,
  });
  const hubTarget = modelConfigTarget("Org/Model.gguf", {
    source: "hub",
    isLora: false,
    loadId: "Org/Model.gguf",
    isDownloaded: true,
    isGguf: true,
  });

  assert.equal(
    modelConfigTargetIsResident({
      target: localTarget,
      selectedId: String.raw`c:\models\llama.gguf`,
      activeGgufVariant: "Q4_K_M",
    }),
    true,
  );
  assert.equal(
    modelConfigTargetIsResident({
      target: hubTarget,
      selectedId: "Org/Model.gguf",
      activeGgufVariant: "Q4_K_M",
    }),
    false,
  );
});

test("an Ollama load identity mirrors its settings only when the API can reach it", () => {
  const id = "ollama-manifest:library/model:latest";
  const link = "/home/u/.ollama/.studio_links/ab12/model-latest.gguf";
  const local = (loadId: string) =>
    ({
      source: "local",
      isLora: false,
      loadId,
      isDownloaded: true,
      isGguf: true,
    }) as const;
  const target = modelConfigTarget(id, local(id), "Llama 3.2");

  // The reference is what /v1/models advertises; the link it materializes the resolver skips.
  assert.equal(target.apiLoadable, true);
  assert.equal(modelConfigTarget(link, local(link), "").apiLoadable, false);
  assert.equal(target.configId, undefined);
  assert.equal(target.displayName, "Llama 3.2");
  assert.equal(
    modelConfigTargetIsResident({
      target,
      selectedId: id,
      activeGgufVariant: "Q4_K_M",
      loaded: true,
    }),
    true,
  );
});

test("non-GGUF weights mirror their settings, a LoRA adapter does not", () => {
  const repo = "mlx-community/Qwen3.5-9B-MLX-8bit";
  const snapshot =
    "/hf/models--mlx-community--Qwen3.5-9B-MLX-8bit/snapshots/abc";
  const cached = modelConfigTarget(repo, {
    source: "hub",
    isLora: false,
    loadId: snapshot,
    isDownloaded: true,
    isGguf: false,
  });
  assert.equal(cached.isGguf, false);
  assert.equal(cached.apiLoadable, true);
  // A local folder row leaves isGguf unset rather than false.
  const folder = "/Users/u/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit";
  assert.equal(
    modelConfigTarget(folder, { source: "local", isLora: false }).apiLoadable,
    true,
  );
  const adapter = "/Users/u/outputs/checkpoint-60";
  assert.equal(
    modelConfigTarget(adapter, { source: "lora", isLora: true }).apiLoadable,
    false,
  );
});

test("the resident settings key a cached non-GGUF repo by its repo id", () => {
  const snapshot =
    "/hf/models--mlx-community--Qwen3.5-9B-MLX-8bit/snapshots/abc";
  const resident = (overrides: {
    modelId?: string;
    ggufVariant?: string | null;
    isGguf?: boolean;
    isLora?: boolean;
  }) =>
    residentModelConfigTarget({
      modelId: snapshot,
      ggufVariant: null,
      isGguf: false,
      isLora: false,
      contextLength: null,
      ...overrides,
    });

  const mlx = resident({});
  assert.equal(mlx.id, snapshot);
  assert.equal(mlx.configId, "mlx-community/Qwen3.5-9B-MLX-8bit");
  assert.equal(mlx.apiLoadable, true);
  // The backend folds a quant's two spellings itself, so a GGUF keeps its load path.
  const gguf = resident({ isGguf: true, ggufVariant: "Q4_K_M" });
  assert.equal(gguf.configId, undefined);
  assert.equal(gguf.apiLoadable, true);
  // A loose file drops its label, and every other entry point keys it by its path.
  const file = resident({
    modelId: `${snapshot}/model-Q4_K_M.gguf`,
    isGguf: true,
    ggufVariant: "Q4_K_M",
  });
  assert.equal(file.ggufVariant, null);
  assert.equal(file.configId, undefined);
  const folder = "/Users/u/.lmstudio/models/org/Model-MLX";
  assert.equal(resident({ modelId: folder }).configId, undefined);
  assert.equal(
    resident({ modelId: "/Users/u/outputs/checkpoint-60", isLora: true })
      .apiLoadable,
    false,
  );
});

test("submitting a cached handoff adopts legacy path-keyed settings", () => {
  store.clear();
  const id = "unsloth/Llama-GGUF";
  const loadId = "/cache/models--unsloth--Llama-GGUF/snapshots/legacy";
  const ggufVariant = "Q4_K_M";
  savePerModelConfig(loadId, ggufVariant, {
    ...DEFAULT_PER_MODEL_CONFIG,
    maxSeqLength: 32768,
  });

  requestModelConfigHandoff({
    requestId: "legacy-settings",
    id,
    meta: {
      source: "hub",
      isLora: false,
      loadId,
      ggufVariant,
      isDownloaded: true,
      isGguf: true,
    },
  });

  const adopted = resolveInitialConfig(id, ggufVariant);
  assert.equal(adopted.remembered, true);
  assert.equal(adopted.config.maxSeqLength, 32768);
  assert.equal(resolveInitialConfig(loadId, ggufVariant).remembered, false);
});


test("full checkpoints mirror remembered engine settings to API loads", () => {
  const meta = { source: "hub", isLora: false, isGguf: false, isDownloaded: true } as const;
  assert.equal(modelConfigTarget("unsloth/Qwen3.5-0.8B", meta).apiLoadable, true);
  assert.equal(modelConfigTarget("local-adapter", { ...meta, isLora: true }).apiLoadable, false);
});
