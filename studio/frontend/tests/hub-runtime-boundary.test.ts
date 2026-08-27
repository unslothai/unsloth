// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const HUB_ROOT = path.join(HERE, "..", "src/features/hub");

function sourceFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(target);
    return /\.(?:ts|tsx)$/.test(entry.name) ? [target] : [];
  });
}

const HUB_SOURCE_FILES = sourceFiles(HUB_ROOT);
const HUB_SOURCES = HUB_SOURCE_FILES.map((file) =>
  readFileSync(file, "utf8"),
).join("\n");
const HUB_RENDER_SOURCES = `${HUB_SOURCES}\n${readFileSync(
  path.join(HUB_ROOT, "hub.css"),
  "utf8",
)}`;

test("obsolete Hub action and settings modules stay removed", () => {
  const removedModules = [
    "catalog/hub-model-settings-view.tsx",
    "inventory/settings-identity.ts",
    "lib/hub-feature-flags.ts",
  ];

  for (const modulePath of removedModules) {
    assert.equal(
      existsSync(path.join(HUB_ROOT, modulePath)),
      false,
      modulePath,
    );
  }
});

test("the Hub no longer owns model runtime actions or model configuration", () => {
  const removedIdentifiers = [
    "HubModelSettingsView",
    "useChatModelRuntime",
    "ModelConfigPage",
    "PerModelConfig",
    "modelConfigIdentity",
    "settingsGgufVariantForRow",
    "openSelectedModelSettings",
    "onOpenModelSettings",
    "CardSettingsButton",
    "settingsTarget",
    "runSelectedModel",
    "openNewChat",
    "loadingPhase",
    "runtimeCapabilities",
    "SelectedResourceRef",
  ];

  for (const identifier of removedIdentifiers) {
    assert.equal(HUB_SOURCES.includes(identifier), false, identifier);
  }
});

test("residency remains a cache mutation safety input", () => {
  const hubPage = readFileSync(path.join(HUB_ROOT, "hub-page.tsx"), "utf8");
  const adoptionSource = readFileSync(
    path.join(HUB_ROOT, "lib/adopt-inference-status.ts"),
    "utf8",
  );
  const ggufCard = readFileSync(
    path.join(HUB_ROOT, "catalog/gguf-download-card.tsx"),
    "utf8",
  );
  const localCard = readFileSync(
    path.join(HUB_ROOT, "catalog/local-on-device-card.tsx"),
    "utf8",
  );
  const safetensorsCard = readFileSync(
    path.join(HUB_ROOT, "catalog/safetensors-download-card.tsx"),
    "utf8",
  );

  assert.match(hubPage, /getInferenceStatus\(\)/);
  assert.match(hubPage, /adoptResidentModelStatus\(/);
  assert.match(
    hubPage,
    /Promise\.all\(\[getInferenceStatus\(\), readIdleUnloadArmed\(\)\]\)\s*\.then\(\(\[status, idleUnloadArmed\]\) => \{/,
  );
  assert.match(
    hubPage,
    /adoptResidentModelStatus\([\s\S]*?modelLoading: store\.modelLoading,\s*idleUnloadArmed: preserveIdleUnloaded && idleUnloadArmed,/,
  );
  assert.match(hubPage, /\.catch\(\(\) => idleUnloadArmed\.current\)/);
  assert.match(
    hubPage,
    /applyStatus: \(previous\) => \{\s*applyActiveModelStatusToStore\(status, \{/,
  );
  assert.match(
    hubPage,
    /checkpointId: isSpeechOnlyStatus\(status\)\s*\? null\s*: resolveInferenceCheckpointId\(status\),/,
  );
  assert.match(hubPage, /speechOnly: isSpeechOnlyStatus\(status\),/);
  assert.doesNotMatch(hubPage, /setCheckpoint\(status\.active_model/);
  assert.match(
    adoptionSource,
    /state\.idleUnloadArmed && !status\.speechOnly/,
  );
  assert.match(
    hubPage,
    /subscribeResidentStatusRefresh\(\s*refreshResidentModelStatus/,
  );
  assert.match(
    hubPage,
    /subscribeModelLifecycle\(\(\{ runtime \}\) => \{\s*if \(runtime === "chat" \|\| runtime === "stt"\) return;\s*void refreshResidentModelStatus\(\{ preserveIdleUnloaded: false \}\);\s*\}\)/,
  );
  assert.match(
    hubPage,
    /idleUnloadArmed: preserveIdleUnloaded && idleUnloadArmed/,
  );
  assert.ok((hubPage.match(/residentModelIdMatches\(/g) ?? []).length >= 2);
  assert.ok((hubPage.match(/selectedModel\.loadId/g) ?? []).length >= 2);
  assert.match(hubPage, /runtime=\{inspectorRuntime\}/);

  const selectedView = readFileSync(
    path.join(HUB_ROOT, "hooks/use-selected-model-view.ts"),
    "utf8",
  );
  assert.match(selectedView, /loadId: selectedCachedRow\.loadId/);
  assert.match(
    selectedView,
    /loadId: selectedCachedRow\?\.loadId \?\? selectedLocalRow\?\.loadId \?\? null/,
  );

  assert.match(
    ggufCard,
    /mutationBlocked=\{\s*isLoadingThisModel \|\|\s*\(isActive && item\.key === activeVariantKey\)/,
  );
  assert.match(
    ggufCard,
    /canDelete=\{[\s\S]*?!selectedIsActive[\s\S]*?!isLoadingThisModel[\s\S]*?\}/,
  );
  assert.match(
    ggufCard,
    /const showUpdateAction =[\s\S]*?!selectedIsActive[\s\S]*?!isLoadingThisModel[\s\S]*?;/,
  );
  assert.match(
    localCard,
    /const canUpdate =[\s\S]*?!isActive &&[\s\S]*?!isLoading &&[\s\S]*?updateAvailable;/,
  );
  assert.match(
    localCard,
    /const canDelete =[\s\S]*?!isActive &&[\s\S]*?!isLoading[\s\S]*?;/,
  );
  assert.match(
    safetensorsCard,
    /const canDelete =[\s\S]*?!isActive &&[\s\S]*?!isLoadingThisModel;/,
  );
});

test("the Hub does not expose model run, eject, training, or loaded actions", () => {
  assert.doesNotMatch(HUB_RENDER_SOURCES, /hub-run-action-btn/);
  assert.doesNotMatch(
    HUB_RENDER_SOURCES,
    />\s*(?:Run|Eject|Loaded|Use in Chat|New Chat|Train)\s*</,
  );
  assert.doesNotMatch(
    HUB_SOURCES,
    /\bonUseInChat\b|\bonEject\b|\bonTrain\b|\bejectModel\b|\bselectModel\b|\brunId\b/,
  );

  const filesWithOnLoad = HUB_SOURCE_FILES.filter((file) =>
    /\bonLoad(?:Local)?\b/.test(readFileSync(file, "utf8")),
  ).map((file) => path.relative(HUB_ROOT, file).split(path.sep).join("/"));

  // This is an image lifecycle callback, not a model runtime action.
  assert.deepEqual(filesWithOnLoad.sort(), ["catalog/owner-avatar.tsx"]);
});

test("the shared model row menu no longer offers Hub model settings", () => {
  const rowMenu = readFileSync(
    path.join(
      HERE,
      "..",
      "src/features/model-picker/components/model-selector/model-row-menu.tsx",
    ),
    "utf8",
  );

  assert.doesNotMatch(rowMenu, /ModelRowMenuSettings|Settings02Icon/);
  assert.doesNotMatch(rowMenu, />\s*Settings\s*</);
});
