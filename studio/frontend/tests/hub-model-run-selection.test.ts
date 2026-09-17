// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { SelectedModelView } from "../src/features/hub/types.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { createHubModelConfigHandoff, isHubModelRunEligible } = await import(
  "../src/features/hub/lib/model-run-selection.ts"
);
const { modelConfigTarget } = await import(
  "../src/features/model-picker/model-config/model-config-handoff.ts"
);

function selectedModel(
  overrides: Partial<SelectedModelView> = {},
): SelectedModelView {
  return {
    id: "Org/Model",
    loadId: "/cache/models--Org--Model/snapshots/revision",
    kind: "cache",
    displayId: "Org/Model",
    hubRepoId: "Org/Model",
    owner: "Org",
    title: "Model",
    summary: "Model",
    sourceLabel: "Hub cache",
    path: "/cache/models--Org--Model/snapshots/revision",
    isLocal: false,
    isGguf: false,
    modelFormat: "safetensors",
    isDownloaded: true,
    runtimeCanChat: true,
    capabilities: [],
    license: null,
    ...overrides,
  };
}

function localModel(
  id: string,
  overrides: Partial<SelectedModelView> = {},
): SelectedModelView {
  return selectedModel({
    id,
    loadId: id,
    kind: "local",
    displayId: id,
    hubRepoId: null,
    path: id,
    localSource: "custom",
    isLocal: true,
    ...overrides,
  });
}

function eligible(
  model: SelectedModelView,
  overrides: Partial<{
    isDataset: boolean;
    mediaRuntime: boolean;
    nonGgufRuntimeAvailable: boolean;
  }> = {},
): boolean {
  return isHubModelRunEligible({
    model,
    isDataset: false,
    mediaRuntime: false,
    nonGgufRuntimeAvailable: true,
    ...overrides,
  });
}

test("only complete chat-loadable models in supported formats are eligible", () => {
  assert.equal(eligible(selectedModel()), true);
  assert.equal(
    eligible(
      selectedModel({
        isGguf: true,
        modelFormat: "gguf",
        requiresVariant: true,
      }),
      { nonGgufRuntimeAvailable: false },
    ),
    true,
  );
  assert.equal(eligible(selectedModel(), { isDataset: true }), false);
  assert.equal(
    eligible(selectedModel(), { nonGgufRuntimeAvailable: false }),
    false,
  );
  assert.equal(eligible(selectedModel({ runtimeCanChat: false })), false);
  assert.equal(eligible(selectedModel({ isDownloaded: false })), false);
  assert.equal(eligible(selectedModel({ isPartial: true })), false);
  assert.equal(eligible(selectedModel({ loadId: null })), false);

  assert.equal(eligible(selectedModel({ modelFormat: "checkpoint" })), true);
  assert.equal(eligible(selectedModel({ modelFormat: "adapter" })), true);
  assert.equal(
    eligible(selectedModel({ modelFormat: "checkpoint" }), {
      nonGgufRuntimeAvailable: false,
    }),
    false,
  );
  assert.equal(
    eligible(selectedModel({ modelFormat: "adapter" }), {
      nonGgufRuntimeAvailable: false,
    }),
    false,
  );
  assert.equal(eligible(selectedModel({ modelFormat: "unknown" })), false);
});

test("complete Hub-backed media models are eligible for their dedicated runtime", () => {
  const mediaModel = selectedModel({
    pipelineTag: "text-to-image",
    runtimeCanChat: false,
  });

  assert.equal(
    eligible(mediaModel, {
      mediaRuntime: true,
      nonGgufRuntimeAvailable: false,
    }),
    true,
  );
  assert.equal(
    eligible(mediaModel, { mediaRuntime: true, isDataset: true }),
    false,
  );
  assert.equal(
    eligible(selectedModel({ isDownloaded: false }), { mediaRuntime: true }),
    false,
  );
  assert.equal(
    eligible(selectedModel({ isPartial: true }), { mediaRuntime: true }),
    false,
  );
  assert.equal(
    eligible(selectedModel({ hubRepoId: null }), { mediaRuntime: true }),
    false,
  );
  assert.equal(
    eligible(localModel("/models/image-model"), { mediaRuntime: true }),
    false,
  );
  assert.equal(
    eligible(
      localModel("/inactive-cache/image-model", {
        hubRepoId: "Org/Image-Model",
        localSource: "hf_cache",
        runtimeCanChat: false,
      }),
      { mediaRuntime: true },
    ),
    true,
  );
});

test("embedding-only non-GGUF models never enter the Chat run flow", () => {
  const embedding = { key: "embedding" as const, label: "Embeddings" };

  for (const pipelineTag of [
    "feature-extraction",
    "sentence-similarity",
    " text-embeddings-inference ",
  ]) {
    assert.equal(
      eligible(selectedModel({ pipelineTag, capabilities: [embedding] })),
      false,
      pipelineTag,
    );
  }
  assert.equal(eligible(selectedModel({ capabilities: [embedding] })), false);
  assert.equal(
    eligible(
      selectedModel({
        capabilities: [
          embedding,
          { key: "conversational", label: "Conversational" },
        ],
      }),
    ),
    true,
  );
  assert.equal(
    eligible(
      selectedModel({
        isGguf: true,
        modelFormat: "gguf",
        pipelineTag: "feature-extraction",
        capabilities: [embedding],
      }),
    ),
    true,
  );
});

test("checkpoint handoffs use the whole-model Chat identity", () => {
  const request = createHubModelConfigHandoff({
    requestId: "request-checkpoint",
    model: selectedModel({ modelFormat: "checkpoint" }),
    selection: {},
  });

  assert.equal(request?.id, "Org/Model");
  assert.equal(request?.meta.isLora, false);
  assert.equal(request?.meta.isGguf, false);
  assert.equal(
    createHubModelConfigHandoff({
      requestId: "request-checkpoint-variant",
      model: selectedModel({ modelFormat: "checkpoint" }),
      selection: { ggufVariant: "Q4_K_M" },
    }),
    null,
  );
});

test("adapter handoffs preserve loader identity and LoRA intent", () => {
  const loadId = String.raw`C:\Models\adapter`;
  const localRequest = createHubModelConfigHandoff({
    requestId: "request-adapter",
    model: localModel(loadId, {
      modelFormat: "adapter",
    }),
    selection: {},
  });

  assert.deepEqual(localRequest, {
    requestId: "request-adapter",
    id: loadId,
    displayName: "Model",
    meta: {
      source: "local",
      isLora: true,
      loadId,
      isDownloaded: true,
      isGguf: false,
      pipelineTag: null,
    },
  });

  const cachedLoadId =
    "/inactive-cache/models--Org--Adapter/snapshots/revision";
  const cachedRequest = createHubModelConfigHandoff({
    requestId: "request-cached-adapter",
    model: localModel(cachedLoadId, {
      hubRepoId: "Org/Adapter",
      localSource: "hf_cache",
      modelFormat: "adapter",
    }),
    selection: {},
  });

  assert.ok(cachedRequest);
  assert.equal(cachedRequest.id, "Org/Adapter");
  assert.equal(cachedRequest.meta.source, "hub");
  assert.equal(cachedRequest.meta.loadId, cachedLoadId);
  assert.equal(cachedRequest.meta.isLora, true);

  const target = modelConfigTarget(
    cachedRequest.id,
    cachedRequest.meta,
    cachedRequest.displayName,
  );
  assert.equal(target.id, cachedLoadId);
  assert.equal(target.configId, "Org/Adapter");
  assert.equal(target.meta.isLora, true);
});

test("MLX safetensors follow runtime availability", () => {
  const models = [
    selectedModel({ libraryName: " MLX " }),
    selectedModel({ tags: ["transformers", "MLX"] }),
    selectedModel({
      id: "mlx-community/Model",
      hubRepoId: "mlx-community/Model",
    }),
    selectedModel({
      id: "Org/Model-MLX-4bit",
      hubRepoId: "Org/Model-MLX-4bit",
    }),
    localModel(String.raw`C:\Models\mlx-community\Qwen2`),
  ];

  for (const model of models) {
    assert.equal(eligible(model), true);
    assert.equal(eligible(model, { nonGgufRuntimeAvailable: false }), false);
    assert.notEqual(
      createHubModelConfigHandoff({
        requestId: "request-mlx",
        model,
        selection: {},
      }),
      null,
    );
  }
});

test("GGUF remains runnable when its conversion retains an MLX identity", () => {
  const models = [
    selectedModel({
      id: "mlx-community/Model-MLX-GGUF",
      hubRepoId: "mlx-community/Model-MLX-GGUF",
      isGguf: true,
      modelFormat: "gguf",
      requiresVariant: true,
      tags: ["mlx", "gguf"],
      libraryName: "mlx",
    }),
    localModel(String.raw`C:\Models\mlx-community\Model.gguf`, {
      isGguf: true,
      modelFormat: "gguf",
      requiresVariant: true,
      tags: ["mlx", "gguf"],
      libraryName: "mlx",
    }),
  ];

  for (const model of models) {
    assert.equal(eligible(model), true);
    assert.notEqual(
      createHubModelConfigHandoff({
        requestId: "request-gguf",
        model,
        selection: {
          ggufVariant: "Q4_K_M",
          ggufFilename: "model-Q4_K_M.gguf",
        },
      }),
      null,
    );
  }
});

test("managed cache handoffs retain separate public, loader, and GGUF identities", () => {
  const loadId = String.raw`C:\Users\alice\.cache\models--Org--Model\snapshots\revision`;
  const ggufVariant = " builds/Q4 K M ";
  const ggufFilename = " builds/My Model Q4 K M.gguf ";
  const request = createHubModelConfigHandoff({
    requestId: "request-1",
    model: selectedModel({
      id: "cache:gguf:Org%2FModel",
      loadId,
      isGguf: true,
      modelFormat: "gguf",
      requiresVariant: true,
      task: "text-generation",
    }),
    selection: {
      ggufVariant,
      ggufFilename,
      expectedBytes: 4096,
    },
  });

  assert.deepEqual(request, {
    requestId: "request-1",
    id: "Org/Model",
    displayName: "Model",
    meta: {
      source: "hub",
      isLora: false,
      loadId,
      isDownloaded: true,
      isGguf: true,
      pipelineTag: "text-generation",
      ggufVariant,
      ggufFilename,
      expectedBytes: 4096,
    },
  });
});

test("matched Discover rows keep the Chat picker's local configuration identity", () => {
  const loadId = "/mnt/c/models/Org/Model";
  const request = createHubModelConfigHandoff({
    requestId: "request-2",
    model: selectedModel({
      id: loadId,
      loadId,
      kind: "local",
      hubRepoId: "Org/Model",
      isLocal: true,
      localSource: "custom",
    }),
    selection: {},
  });

  assert.equal(request?.id, loadId);
  assert.equal(request?.meta.source, "local");
  assert.equal(request?.meta.loadId, loadId);
});

test("opaque local loader identities remain unchanged on every desktop path style", () => {
  const loadIds = [
    String.raw`C:\Models\Family\Model.gguf`,
    String.raw`\\server\models\Model.gguf`,
    "/mnt/c/Models/Family/Model.gguf",
    "/home/alice/models/Model.gguf",
    "/Users/alice/Models/Model.gguf",
    "ollama-manifest:library/model:latest",
  ];

  for (const loadId of loadIds) {
    const request = createHubModelConfigHandoff({
      requestId: loadId,
      model: selectedModel({
        id: loadId,
        loadId,
        kind: "local",
        hubRepoId: null,
        path: loadId,
        isLocal: true,
        localSource: loadId.startsWith("ollama-") ? "ollama" : "custom",
        isGguf: true,
        modelFormat: "gguf",
      }),
      selection: {},
    });

    assert.equal(request?.id, loadId);
    assert.equal(request?.meta.loadId, loadId);
  }
});

test("opaque local identities retain the inventory display name", () => {
  const request = createHubModelConfigHandoff({
    requestId: "ollama-display",
    model: selectedModel({
      id: "ollama-manifest:%2Fhome%2Fhz%2F.ollama%2Fmodels%2Fmanifests%2Flibrary%2Fllama",
      loadId:
        "ollama-manifest:%2Fhome%2Fhz%2F.ollama%2Fmodels%2Fmanifests%2Flibrary%2Fllama",
      title: "Llama 3.2",
      kind: "local",
      hubRepoId: null,
      isLocal: true,
      localSource: "ollama",
      isGguf: true,
      modelFormat: "gguf",
    }),
    selection: {},
  });

  assert.equal(request?.displayName, "Llama 3.2");
});

test("invalid format and variant combinations fail closed", () => {
  assert.equal(
    createHubModelConfigHandoff({
      requestId: "missing-variant",
      model: selectedModel({
        isGguf: true,
        modelFormat: "gguf",
        requiresVariant: true,
      }),
      selection: {},
    }),
    null,
  );
  assert.equal(
    createHubModelConfigHandoff({
      requestId: "unexpected-variant",
      model: selectedModel(),
      selection: { ggufVariant: "Q4_K_M" },
    }),
    null,
  );
  assert.equal(
    createHubModelConfigHandoff({
      requestId: "partial",
      model: selectedModel({ isPartial: true }),
      selection: {},
    }),
    null,
  );
});
