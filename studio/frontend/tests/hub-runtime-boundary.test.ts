// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = path.join(HERE, "..");
const HUB_ROOT = path.join(FRONTEND_ROOT, "src/features/hub");

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
    /adoptingExistingServerModel:\s*previous\.checkpoint === null \|\| previous\.checkpoint === "",/,
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
    /mutationBlocked=\{\s*runPending \|\|\s*isLoadingThisModel \|\|\s*\(isActive && item\.key === activeVariantKey\)/,
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
    /const canDelete =[\s\S]*?!isActive &&[\s\S]*?!isLoadingThisModel &&[\s\S]*?!runPending;/,
  );
});

test("the Hub hands Run to shared configuration without owning runtime actions", () => {
  const hubPage = readFileSync(path.join(HUB_ROOT, "hub-page.tsx"), "utf8");
  const modelInspector = readFileSync(
    path.join(HUB_ROOT, "catalog/model-inspector.tsx"),
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

  assert.match(HUB_RENDER_SOURCES, /hub-run-action-btn/);
  assert.match(HUB_RENDER_SOURCES, />\s*Run\s*</);
  assert.match(
    hubPage,
    /clearNewChatDraft\(\);[\s\S]*?setActiveThreadId\(null\);[\s\S]*?setActiveProjectId\(null\);[\s\S]*?setIncognito\(false\);[\s\S]*?requestModelConfigHandoff\(request\)/,
  );
  assert.match(hubPage, /requestModelConfigHandoff\(request\)/);
  assert.match(hubPage, /to: "\/chat", search: \{ new: requestId \}/);
  assert.match(
    hubPage,
    /const controller = runConfigOpenCoordinator\.begin\(\);[\s\S]*?await waitForRunConfigRefresh\(\s*refreshResidentModelStatus\(\),\s*controller\.signal,\s*\);\s*if \(controller\.signal\.aborted\) return;/,
  );
  assert.match(hubPage, /const RUN_CONFIG_REFRESH_TIMEOUT_MS = 5_000;/);
  assert.match(
    hubPage,
    /const handleCloseDetail = useCallback\(\(\) => \{\s*runConfigOpenCoordinator\.cancel\(\);\s*setRunConfigOpening\(null\);/,
  );
  assert.match(HUB_RENDER_SOURCES, /Opening…/);
  assert.match(HUB_RENDER_SOURCES, /aria-busy=\{loading\}/);
  assert.match(ggufCard, /open=\{runPending \? false : open\}/);
  assert.match(ggufCard, /disabled=\{runPending\}/);
  assert.match(localCard, /open=\{runPending \? false : variantOpen\}/);
  assert.match(
    localCard,
    /disabled=\{currentVariantState\.loading \|\| runPending\}/,
  );
  assert.match(
    localCard,
    /label=\{`Configure and run \$\{displayName\.trim\(\) \|\| "this model"\}`\}/,
  );
  assert.doesNotMatch(
    localCard,
    /label=\{`Configure and run \$\{repoId \?\? modelId\}`\}/,
  );
  assert.match(
    safetensorsCard,
    /const showRunAction =[^;]*?!isLoadingThisModel[^;]*?;/s,
  );
  assert.doesNotMatch(
    safetensorsCard,
    /const showRunAction =[^;]*?repoPeerActive[^;]*?;/s,
  );
  assert.match(
    ggufCard,
    /const showRunAction =[^;]*?!isLoadingThisModel[^;]*?;/s,
  );
  assert.match(
    localCard,
    /const showRunAction =[^;]*?!isLoading[^;]*?;/s,
  );
  assert.match(
    modelInspector,
    /s\.isChatOnly\(\) && !s\.capabilitiesUnknown\(\)/,
  );
  assert.match(
    modelInspector,
    /nonGgufRuntimeAvailable:\s*!chatOnlyMeasured &&\s*unslothSupport\.status !== "unsupported"/,
  );
  assert.match(
    modelInspector,
    /const openRunConfig = isHubModelRunEligible\(/,
  );
  assert.match(
    ggufCard,
    /const showRunAction =[^;]*?!downloadingThisVariant[^;]*?;/s,
  );
  assert.match(
    modelInspector,
    /<ModelStatusChips[\s\S]*?chatOnly=\{chatOnly\}/,
  );
  assert.doesNotMatch(
    HUB_RENDER_SOURCES,
    />\s*(?:Eject|Loaded|Use in Chat|New Chat|Train)\s*</,
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

test("the Run handoff opens configuration immediately and remains nonce-scoped", () => {
  const chatPage = readFileSync(
    path.join(FRONTEND_ROOT, "src/features/chat/chat-page.tsx"),
    "utf8",
  );
  const modelSelector = readFileSync(
    path.join(
      FRONTEND_ROOT,
      "src/features/model-picker/components/model-selector.tsx",
    ),
    "utf8",
  );

  assert.match(chatPage, /modelConfigHandoffForDestination\(state\.request,/);
  assert.match(
    chatPage,
    /active,\s*newChatId: search\.new,\s*threadId: search\.thread,\s*compareId: search\.compare,\s*projectId: search\.project,/,
  );
  assert.match(chatPage, /modelSelectorOpen \|\| modelConfigRequest !== null/);
  assert.match(modelSelector, /requestedConfigTarget \?\? configTarget/);
  assert.match(
    modelSelector,
    /aria-label=\{\s*visibleConfigTarget\s*\? `Run settings for \$\{visibleConfigTarget\.displayName\}`\s*:\s*undefined\s*\}/,
  );
  assert.match(
    modelSelector,
    /modelConfigInstanceKey\(\s*visibleConfigTarget\.configId \?\? visibleConfigTarget\.id,\s*visibleConfigTarget\.ggufVariant,\s*visibleLoadedConfig,/,
  );
  assert.match(
    modelSelector,
    /setAdoptedConfigRequestId\(configRequest\.requestId\);\s*setConfigTarget\(requestedConfigTarget\)/,
  );
  assert.match(
    modelSelector,
    /onConfigRequestAdopted\?\.\(configRequest\.requestId\)/,
  );
  assert.match(
    chatPage,
    /setSettingsOpen\(false\);\s*setModelSelectorLocked\(false\);\s*setModelSelectorOpen\(true\);\s*clearModelConfigHandoff\(requestId\)/,
  );
  assert.match(
    chatPage,
    /open=\{active && modelConfigRequest === null && settingsOpen\}/,
  );
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
