// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createHubModelConfigHandoff,
  isHubModelRunEligible,
} from "../src/features/hub/lib/model-run-selection.ts";
import type { SelectedModelView } from "../src/features/hub/types.ts";

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
    safetensorsRuntimeAvailable: boolean;
  }> = {},
): boolean {
  return isHubModelRunEligible({
    model,
    isDataset: false,
    mediaRuntime: false,
    safetensorsRuntimeAvailable: true,
    ...overrides,
  });
}

test("only complete chat-loadable safetensors and GGUF models are eligible", () => {
  assert.equal(eligible(selectedModel()), true);
  assert.equal(
    eligible(
      selectedModel({
        isGguf: true,
        modelFormat: "gguf",
        requiresVariant: true,
      }),
      { safetensorsRuntimeAvailable: false },
    ),
    true,
  );
  assert.equal(eligible(selectedModel(), { isDataset: true }), false);
  assert.equal(eligible(selectedModel(), { mediaRuntime: true }), false);
  assert.equal(
    eligible(selectedModel(), { safetensorsRuntimeAvailable: false }),
    false,
  );
  assert.equal(eligible(selectedModel({ runtimeCanChat: false })), false);
  assert.equal(eligible(selectedModel({ isDownloaded: false })), false);
  assert.equal(eligible(selectedModel({ isPartial: true })), false);
  assert.equal(eligible(selectedModel({ loadId: null })), false);

  for (const modelFormat of ["adapter", "checkpoint", "unknown"] as const) {
    assert.equal(eligible(selectedModel({ modelFormat })), false, modelFormat);
  }
});

test("MLX inventory identities never enter the Hub Run handoff", () => {
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
  ];

  for (const model of models) {
    assert.equal(eligible(model), false);
    assert.equal(
      createHubModelConfigHandoff({
        requestId: "request-mlx",
        model,
        selection: {},
      }),
      null,
    );
  }

  assert.equal(
    eligible(
      selectedModel({
        id: "Org/XMLXModel",
        hubRepoId: "Org/XMLXModel",
      }),
    ),
    true,
  );
});

test("local MLX identities fail closed across desktop path styles", () => {
  const paths = [
    "/models/mlx-community/Qwen2",
    "/mnt/c/Models/mlx-community/Qwen2",
    "/Users/alice/Models/mlx-community/Qwen2",
    String.raw`C:\Models\mlx-community\Qwen2`,
    String.raw`\\server\share\mlx-community\Qwen2`,
    "/models/Org/Qwen2_MLX",
  ];

  for (const path of paths) {
    const model = localModel(path);
    assert.equal(eligible(model), false, path);
    assert.equal(
      createHubModelConfigHandoff({
        requestId: path,
        model,
        selection: {},
      }),
      null,
      path,
    );
  }
});

test("local MLX identity matching does not overreach", () => {
  const paths = [
    "/models/not-mlx-community/Qwen2",
    "/models/Org/MyMLX",
    String.raw`C:\Models\XMLX\Qwen2`,
    String.raw`\\mlx-community\share\Org\Qwen2`,
  ];

  for (const path of paths) {
    assert.equal(eligible(localModel(path)), true, path);
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
