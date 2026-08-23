// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TYPESCRIPT_FILE_RE = /\.tsx?$/;
const CHAT_IMPORT_RE = /@\/features\/chat(?:\/|["'])/;
const TRAINING_IMPORT_RE = /@\/features\/training(?:\/|["'])/;
const INFERENCE_API_RE = /\/api\/inference/;
const HUB_RUNTIME_OWNER_RE =
  /useChatRuntimeStore|useChatModelRuntime|getInferenceStatus|loadEmbeddingModelSettings|useHiddenEmbeddingModelIds|isConfiguredHiddenModelId|ModelConfigPage|PerModelConfig|ModelInspectorRuntime|inspectorRuntime|\bchatOnly\b|normalizeRuntime|routableToMediaPage|\brunId\b|\bruntime\?:\s*string/;
const CHAT_HEAVY_BARREL_RE =
  /from\s+["']@\/features\/(?:model-picker|settings)["']/;
const CHAT_ROUTE_RE = /to:\s*["']\/chat["']/;
const TRAINING_ROUTE_RE = /to:\s*["']\/train(?:\/|["'])/;
const RUN_ACTION_CLASS_RE = /hub-run-action-btn/;
const RUNTIME_LABEL_RE = />\s*(?:Run|Eject|Loaded)\s*</;
const RUNTIME_HANDLER_RE = /onUseInChat|onEject/;
const TRAINING_ACTION_RE = /\bonTrain\b|\bTrainIcon\b|>\s*Train\s*</;
const CARD_LOAD_HANDLER_RE = /onLoad(?:Local)?/;
const SETTLED_GGUF_ONLY_DEVICE_RE =
  /!\s*s\.capabilitiesUnknown\(\s*\)\s*&&\s*s\.isChatOnly\(\s*\)/;
const GGUF_ONLY_DEVICE_LABEL_RE = /GGUF-only device/;
const RUNTIME_FIELD_RE = /\bruntime\s*(\?)?\s*:\s*([^;]+);/g;
const MODEL_INVENTORY_RUNTIME_TYPE_RE = /^ModelInventoryRuntime\s*\|\s*null$/;
const STATIC_IMPORT_RE =
  /(?:import|export)\s+(?:type\s+)?(?:[^"'`;]*?\s+from\s+)?["']([^"']+)["']/g;
const DYNAMIC_IMPORT_RE = /\bimport\(\s*["']([^"']+)["']\s*\)/g;

function readHubSource(relativePath: string): string {
  return readFileSync(
    path.join(HERE, "..", "src/features/hub", relativePath),
    "utf8",
  );
}

function sourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      return sourceFiles(target);
    }
    return TYPESCRIPT_FILE_RE.test(entry.name) ? [target] : [];
  });
}

function resolveLocalModule(from: string, specifier: string): string | null {
  const base = specifier.startsWith("@/")
    ? path.join(SOURCE_ROOT, specifier.slice(2))
    : specifier.startsWith(".")
      ? path.resolve(path.dirname(from), specifier)
      : null;
  if (!base) {
    return null;
  }
  const candidates = TYPESCRIPT_FILE_RE.test(base)
    ? [base]
    : [
        `${base}.ts`,
        `${base}.tsx`,
        path.join(base, "index.ts"),
        path.join(base, "index.tsx"),
      ];
  return candidates.find((candidate) => existsSync(candidate)) ?? null;
}

function localDependencies(file: string): string[] {
  const source = readFileSync(file, "utf8");
  return [STATIC_IMPORT_RE, DYNAMIC_IMPORT_RE].flatMap((pattern) =>
    Array.from(source.matchAll(pattern), (match) => match[1])
      .map((specifier) => resolveLocalModule(file, specifier))
      .filter((dependency): dependency is string => dependency !== null),
  );
}

function dependencyClosure(entries: string[]): Set<string> {
  const visited = new Set<string>();
  const pending = [...entries];
  while (pending.length > 0) {
    const file = pending.pop();
    if (!file || visited.has(file)) {
      continue;
    }
    visited.add(file);
    pending.push(...localDependencies(file));
  }
  return visited;
}

const SOURCE_ROOT = path.join(HERE, "..", "src");
const HUB_SOURCE_ROOT = path.join(SOURCE_ROOT, "features/hub");
const MODEL_PICKER_SOURCE_ROOT = path.join(
  SOURCE_ROOT,
  "features/model-picker",
);
const TRAINING_SOURCE_ROOT = path.join(SOURCE_ROOT, "features/training");
const HUB_SOURCE_FILES = sourceFiles(HUB_SOURCE_ROOT);
const HUB_SOURCES = HUB_SOURCE_FILES.map((file) =>
  readFileSync(file, "utf8"),
).join("\n");
const HUB_RENDER_SOURCES = `${HUB_SOURCES}\n${readHubSource("hub.css")}`;
const HUB_PAGE = readHubSource("hub-page.tsx");
const DOWNLOAD_CARDS = [
  "catalog/download-section.tsx",
  "catalog/gguf-download-card.tsx",
  "catalog/local-on-device-card.tsx",
  "catalog/safetensors-download-card.tsx",
]
  .map(readHubSource)
  .join("\n");
const DELETE_DIALOG_OWNERS = [
  readHubSource("catalog/safetensors-download-card.tsx"),
  readHubSource("catalog/gguf-download-card.tsx"),
  readHubSource("catalog/local-on-device-card.tsx"),
  readHubSource("catalog/model-row-menu.tsx"),
];

test("the Hub does not own Chat or Training workflows", () => {
  assert.doesNotMatch(HUB_SOURCES, CHAT_IMPORT_RE);
  assert.doesNotMatch(HUB_SOURCES, TRAINING_IMPORT_RE);
  assert.doesNotMatch(HUB_SOURCES, INFERENCE_API_RE);
  assert.doesNotMatch(HUB_SOURCES, HUB_RUNTIME_OWNER_RE);
  assert.doesNotMatch(HUB_SOURCES, CHAT_HEAVY_BARREL_RE);
  assert.doesNotMatch(HUB_PAGE, CHAT_ROUTE_RE);
  assert.doesNotMatch(HUB_PAGE, TRAINING_ROUTE_RE);
  assert.doesNotMatch(HUB_SOURCES, TRAINING_ACTION_RE);
});

test("runtime remains wire metadata rather than Hub-owned state", () => {
  const inventoryApi = readHubSource("inventory/api.ts");
  const references = HUB_SOURCE_FILES.filter((file) =>
    readFileSync(file, "utf8").includes("ModelInventoryRuntime"),
  )
    .map((file) =>
      path.relative(HUB_SOURCE_ROOT, file).split(path.sep).join("/"),
    )
    .sort();
  assert.deepEqual(references, [
    "index.ts",
    "inventory/api.ts",
    "inventory/index.ts",
  ]);
  const runtimeFields = Array.from(inventoryApi.matchAll(RUNTIME_FIELD_RE));
  assert.ok(runtimeFields.length > 0);
  for (const [, optional, type] of runtimeFields) {
    assert.equal(optional, "?");
    const runtimeType = type.trim();
    assert.match(runtimeType, MODEL_INVENTORY_RUNTIME_TYPE_RE);
  }
});

test("the complete Hub dependency graph does not enter the Chat feature", () => {
  const chatDependencies = Array.from(
    dependencyClosure(HUB_SOURCE_FILES),
  ).filter((file) => file.includes(path.join("features", "chat")));
  assert.deepEqual(chatDependencies, []);
});

test("the complete Hub dependency graph does not enter the model picker", () => {
  const modelPickerDependencies = Array.from(
    dependencyClosure(HUB_SOURCE_FILES),
  ).filter((file) => file.startsWith(`${MODEL_PICKER_SOURCE_ROOT}${path.sep}`));
  assert.deepEqual(modelPickerDependencies, []);
});

test("the complete Hub dependency graph does not enter Training", () => {
  const trainingDependencies = Array.from(
    dependencyClosure(HUB_SOURCE_FILES),
  ).filter((file) => file.startsWith(`${TRAINING_SOURCE_ROOT}${path.sep}`));
  assert.deepEqual(trainingDependencies, []);
});

test("Hub does not expose runtime actions or state", () => {
  assert.doesNotMatch(HUB_RENDER_SOURCES, RUN_ACTION_CLASS_RE);
  assert.doesNotMatch(HUB_RENDER_SOURCES, RUNTIME_LABEL_RE);
  assert.doesNotMatch(HUB_SOURCES, RUNTIME_HANDLER_RE);
  assert.doesNotMatch(DOWNLOAD_CARDS, CARD_LOAD_HANDLER_RE);
});

test("Hub keeps settled device compatibility guidance without owning inference state", () => {
  assert.match(HUB_PAGE, SETTLED_GGUF_ONLY_DEVICE_RE);
  assert.match(
    readHubSource("catalog/model-inspector.tsx"),
    GGUF_ONLY_DEVICE_LABEL_RE,
  );
});

test("every model delete dialog reads the backend delete preflight", () => {
  // What the preview decides is covered behaviourally in delete-impact-state.test.ts; only
  // the wiring needs source inspection, because a dialog that drops one of these props
  // silently falls back to its default and stops reporting the preflight at all.
  for (const source of DELETE_DIALOG_OWNERS) {
    assert.match(source, /blocked=\{deleteImpactState\.blocked\}/);
    assert.match(source, /checking=\{deleteImpactState\.checking\}/);
    assert.match(source, /unavailable=\{deleteImpactState\.unavailable\}/);
  }
});
