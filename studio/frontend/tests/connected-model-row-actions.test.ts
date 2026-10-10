// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const pickers = readSrc(
  "features/model-picker/components/model-selector/pickers.tsx",
);
const marks = readSrc(
  "features/model-picker/components/model-selector/connected-model-meta.ts",
);
const connectedPins = readSrc(
  "features/model-picker/components/model-selector/pinned-connected-models.ts",
);
const onDevicePins = readSrc(
  "features/model-picker/components/model-selector/pinned-models.ts",
);
const selector = readSrc("features/model-picker/components/model-selector.tsx");
const settingsStore = readSrc(
  "features/settings/stores/settings-dialog-store.ts",
);
const connectionsTab = readSrc("features/settings/tabs/connections-tab.tsx");
const providersDialog = readSrc("features/chat/chat-providers-dialog.tsx");

test("a connected row draws its badges through ModelRow", () => {
  assert.match(
    pickers,
    /const renderConnectedModelRow = \(\s*model: ExternalModelOption,/,
  );
  assert.match(
    pickers,
    /capabilities=\{marks\.capabilities\}\s*\n\s*showVision=\{marks\.vision\}/,
  );
  assert.doesNotMatch(pickers, /reserveLeadingSlot/);
  assert.doesNotMatch(pickers, /leadingBadge=\{\s*<ApiProviderLogo/);
  assert.match(
    pickers,
    /pinnedConnectedRows\.map\(\(model\) =>\s*renderPinnedDragRow\(\s*pinnedConnectedDrag,\s*model\.id,\s*renderConnectedModelRow\(model, true\)/,
  );
  assert.match(
    pickers,
    /group\.models\.map\(\(model\) =>\s*renderConnectedModelRow\(model, !headed\)/,
  );
});

test("capabilities are keyed by the provider's own model id", () => {
  // `model.id` is the external:: address and `model.name` is rewritten by OpenRouter.
  assert.match(
    pickers,
    /parseExternalModelId\(model\.id\)\?\.modelId \?\? model\.name/,
  );
  assert.match(
    pickers,
    /connectedModelMarks\(\{\s*providerType: model\.providerType,\s*modelId: providerModelId,\s*baseUrl,\s*apiType: externalApiTypeById\.get\(model\.providerId\),/,
  );
  assert.match(
    pickers,
    /useSyncExternalStore\(\s*subscribeModelCatalog,\s*modelCatalogVersion,?\s*\)/,
  );
});

test("modality comes from the resolvers the app already has", () => {
  assert.match(
    marks,
    /providerModelSupportsVision\(providerType, modelId\) === true/,
  );
  assert.match(marks, /providerSupportsBuiltinImageGeneration\(/);
  assert.match(marks, /videoGen: false,/);
  assert.doesNotMatch(marks, /byName/);
  assert.doesNotMatch(marks, /detectCapabilities/);
  assert.match(
    marks,
    /imageGen: providerSupportsBuiltinImageGeneration\(\s*providerType,\s*modelId,\s*baseUrl,\s*apiType,\s*\),/,
  );
  // Audio is withheld: the attachment adapter only knows loaded local models.
  assert.match(marks, /audio: false,/);
  assert.doesNotMatch(marks, /includes\("audio"\)/);
  const audioAdapter = readSrc("features/chat/audio-attachment-adapter.ts");
  assert.match(
    audioAdapter,
    /const activeModel = state\.models\.find\(\(m\) => m\.id === checkpoint\);/,
  );
  assert.match(audioAdapter, /if \(modelLoaded && !activeModel\?\.hasAudioInput\) \{/);
  assert.match(
    readSrc("features/chat/lib/attached-media-gate.ts"),
    /if \(audio && !activeModel\?\.hasAudioInput\) \{/,
  );
  assert.match(
    pickers,
    /type ConnectedModalityFilter = "all" \| "vision" \| "imageGen";/,
  );
  assert.doesNotMatch(
    marks,
    /vision: providerModelSupportsVision\([^)]*\) \?\?/,
  );
});

test("connected pins are a separate list from the On Device ones", () => {
  assert.match(connectedPins, /"unsloth_pinned_connected_models"/);
  assert.match(onDevicePins, /"unsloth_pinned_models"/);
  // External ids contain "::", which the On Device pin store reads as a quant separator.
  assert.match(onDevicePins, /const sep = key\.indexOf\("::"\)/);
  assert.doesNotMatch(connectedPins, /"unsloth_pinned_models"/);
  assert.doesNotMatch(pickers, /togglePinned\(model\.id\)/);
  assert.doesNotMatch(pickers, /pinKey\(model\.id\)/);
  assert.match(pickers, /onToggle: \(\) => togglePinnedConnected\(model\.id\)/);
  // Toggles and drag commits rebase on the stored list, since writes replace it whole.
  // Behaviour is tested in pinned-connected-models-reorder.test.ts.
  assert.match(
    connectedPins,
    /const next = rebaseOnStored\(state\.pinned\);\s*writePinned\(next\);/,
  );
});

test("a pin moves the row out of its provider group", () => {
  assert.match(
    pickers,
    /for \(const model of connectedMatches\) \{\s*if \(pinnedConnectedSet\.has\(model\.id\)\) continue;/,
  );
  assert.match(
    pickers,
    /connectedMatches\s*\.filter\(\(model\) => pinnedConnectedSet\.has\(model\.id\)\)/,
  );
});

test("connected rows join the keyboard order", () => {
  assert.match(
    pickers,
    /if \(section === "connected"\) \{\s*if \(!pinnedConnectedCollapsed\) \{\s*keys\.push\(\s*\.\.\.pinnedConnectedRows\.map/,
  );
  assert.match(
    pickers,
    /if \(collapsedConnectedGroups\.has\(group\.providerId\)\) continue;/,
  );
});

test("nothing on a connected row opens local run settings", () => {
  assert.doesNotMatch(
    pickers,
    /onConfigure\(model\.id, \{\s*source: "external"/,
  );
  assert.match(
    selector,
    /function handleConfigureConnection\(providerId: string\) \{\s*setOpen\(false\);\s*useSettingsDialogStore\.getState\(\)\.openConnectionSettings\(providerId\);/,
  );
});

test("the connection request is one-shot and tab-scoped", () => {
  assert.match(
    settingsStore,
    /connectionRequested:\s*\n?\s*tab === "connections" \? state\.connectionRequested : null/,
  );
  assert.match(
    settingsStore,
    /closeDialog: \(\) =>\s*set\(\{[^}]*connectionRequested: null,/,
  );
  assert.match(connectionsTab, /openProviderId=\{connectionRequested\}/);
  assert.match(
    connectionsTab,
    /onOpenProviderConsumed=\{consumeConnectionRequest\}/,
  );
});

test("the deep link waits for the provider and fires once", () => {
  assert.match(providersDialog, /\[providersReady, setProvidersReady\] = useState\(false\)/);
  assert.match(providersDialog, /onProvidersChange\(syncedProviders\);\s*setProvidersReady\(true\);/);
  assert.match(providersDialog, /if \(!providersReady\) return;/);
  assert.match(
    providersDialog,
    /toast\.error\(`Failed to load connections: \$\{message\}`\);\s*\}\s*(\/\/[^\n]*\n\s*)*if \(isMounted\) setProvidersReady\(true\);/,
  );
  assert.match(providersDialog, /\[openProviderId, providers, providersReady, onOpenProviderConsumed\]/);
  assert.match(providersDialog, /if \(!provider\) return;/);
  assert.match(
    providersDialog,
    /openedProviderRef\.current = openProviderId;\s*void editProvider\(provider\);\s*onOpenProviderConsumed\?\.\(\);/,
  );
  assert.doesNotMatch(
    providersDialog,
    /autoOpenedAddFormRef\.current = false;/,
  );
});

test("the row menu carries the connected actions", () => {
  for (const label of ["Model info", "Copy model ID"]) {
    assert.match(pickers, new RegExp(`label: "${label}"`));
  }
  // navigator.clipboard is undefined in the desktop shell and over plain HTTP.
  assert.match(
    pickers,
    /if \(await copyToClipboard\(providerModelId\)\) \{\s*toast\.success/,
  );
  assert.doesNotMatch(pickers, /navigator\.clipboard\s*\n?\s*\.?writeText/);
  assert.doesNotMatch(pickers, /Use by default in new chats/);
  assert.doesNotMatch(pickers, /Hide from this list/);
  assert.doesNotMatch(pickers, /label: "Connection settings",/);
  assert.doesNotMatch(
    pickers,
    /onSelect: \(\) => onConfigureConnection\(model\.providerId\),/,
  );
});

test("the name sort flattens the list without dropping a model", () => {
  assert.match(pickers, /if \(connectedSort !== "name"\) return groups;/);
  assert.match(
    pickers,
    /for \(const model of connectedMatches\) \{\s*if \(pinnedConnectedSet\.has\(model\.id\)\) continue;/,
  );
  assert.doesNotMatch(pickers, /headless && onConfigureConnection/);
});

test("the connection's own settings hang off the group heading", () => {
  assert.match(
    pickers,
    /<div className="-mr-2 flex shrink-0 items-center -space-x-0\.5">\s*\{onConfigure \?/,
  );
  assert.match(
    pickers,
    /HEADING_ACTION_CLASS,\s*"opacity-0 group-hover\/heading:opacity-100/,
  );
  assert.match(pickers, /className=\{HEADING_ACTION_CLASS\}\s*>\s*\{collapsed \?/);
  assert.match(
    pickers,
    /const HEADING_ACTION_CLASS =\s*"flex size-5 shrink-0 items-center justify-center rounded-md/,
  );
  assert.match(
    pickers,
    /aria-label=\{configureLabel \?\? "Connection settings"\}/,
  );
  assert.match(
    pickers,
    /onConfigure=\{\s*onConfigureConnection\s*\? \(\) => onConfigureConnection\(group\.providerId\)/,
  );
  assert.match(
    pickers,
    /configureLabel=\{`\$\{group\.providerName\} connection settings`\}/,
  );
  assert.match(
    pickers,
    /label="Pinned"\s*collapsed=\{pinnedConnectedCollapsed\}\s*onToggle=\{\(\) =>\s*setPinnedConnectedCollapsed/,
  );
});

test("the gear opens the model's own settings", () => {
  assert.match(
    pickers,
    /tooltip="Model settings"\s*onConfigure=\{\(\) =>\s*setSettingsModel\(\{/,
  );
  assert.doesNotMatch(pickers, /label: "Model settings/);
});

test("the connected sort and modality filter drive the list", () => {
  assert.match(pickers, /CONNECTED_SORT_OPTIONS/);
  assert.match(pickers, /CONNECTED_MODALITY_OPTIONS/);
  assert.match(pickers, /if \(connectedSort !== "name"\) return groups;/);
});

test("per-model prompt and cap reuse the memory Chat already keeps", () => {
  const settingsDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-settings-dialog.tsx",
  );
  assert.match(settingsDialog, /setRememberedParamsForModel\(checkpointId, \{/);
  assert.doesNotMatch(settingsDialog, /localStorage/);
  // Patch only touched fields; null means untouched.
  assert.match(settingsDialog, /const systemPrompt = promptDraft \?\? remembered\?\.systemPrompt \?\? "";/);
  assert.match(
    settingsDialog,
    /const liveEffortDraft =\s*effortDraft !== null && \(effortDraft === FOLLOW_CHAT \|\| offered\(effortDraft\)\)\s*\? effortDraft\s*: null;/,
  );
  assert.match(
    settingsDialog,
    /const effort =\s*liveEffortDraft \?\?\s*\(pinnedEffort && offered\(pinnedEffort\) \? pinnedEffort : FOLLOW_CHAT\);/,
  );
  assert.doesNotMatch(
    readSrc(
      "features/model-picker/components/model-selector/connected-model-info-dialog.tsx",
    ),
    /pinnedEffort/,
  );
  assert.match(
    settingsDialog,
    /\.\.\.\(promptDraft !== null \? \{ systemPrompt: promptDraft \} : \{\}\),/,
  );
  assert.match(
    settingsDialog,
    /if \(liveEffortDraft !== null\) \{\s*const pinned = liveEffortDraft === FOLLOW_CHAT \? null : liveEffortDraft;\s*setModelReasoningEffort\(checkpointId, pinned\);/,
  );
  const effortStore = readSrc(
    "features/model-picker/components/model-selector/model-reasoning-effort.ts",
  );
  assert.match(
    effortStore,
    /window\.addEventListener\("storage", \(event\) => \{/,
  );
  const chatPage = readSrc("features/chat/chat-page.tsx");
  assert.match(
    chatPage,
    /const activePinnedEffort = useModelReasoningEffortStore\(\s*\(state\) => state\.effortByModel\[inferenceParams\.checkpoint\],/,
  );
  assert.match(
    chatPage,
    /if \(appliedPinnedEffort\.current === activePinnedEffort\) return;/,
  );
  assert.match(
    chatPage,
    /reconcilePinnedReasoningEffort\(\{\s*checkpoint: inferenceParams\.checkpoint,\s*caps,\s*providerType: provider\?\.providerType,\s*apiType: provider\?\.apiType,\s*\}\);\s*\}, \[activePinnedEffort,/,
  );
  assert.match(
    chatPage,
    /reasoningFieldsAfterCatalogRefresh\(useChatRuntimeStore\.getState\(\), caps\),\s*\);[\s\S]{0,220}?reconcilePinnedReasoningEffort\(\{/,
  );
  assert.match(readSrc("features/chat/stores/chat-runtime-store.ts"), /if \(!pinned && !pinHoldsLiveEffort\(\)\) return;/);
  assert.match(
    chatPage,
    /current:\s*!pinnedEffort && pinHoldsLiveEffort\(\)\s*\? \(takeEffortDisplacedByPin\(\) \?\? store\.reasoningEffort\)\s*: store\.reasoningEffort,/,
  );
  assert.match(
    chatPage,
    /current:\s*!pinnedEffort && pinHoldsLiveEffort\(\)\s*\? \(takeEffortDisplacedByPin\(\) \?\? state\.reasoningEffort\)\s*: state\.reasoningEffort,/,
  );
  assert.match(
    readSrc("features/chat/stores/chat-runtime-store.ts"),
    /export function pinHoldsLiveEffort\(\): boolean \{\s*return \(\s*pinOwnsLiveReasoningEffort\(useChatRuntimeStore\.getState\(\)\) \|\|\s*effortDisplacedByPin !== null\s*\);/,
  );
  assert.match(
    readSrc("features/chat/stores/chat-runtime-store.ts"),
    /if \(key === "reasoningEffort" && pinOwnsLiveReasoningEffort\(state\)\) \{[\s\S]{0,500}?effortDisplacedByPin = value as ReasoningEffort;/,
  );
  const localLoad = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(
    localLoad,
    /const existingReasoningEffort =\s*\(pinHoldsLiveEffort\(\) \? takeEffortDisplacedByPin\(\) : null\) \?\?\s*useChatRuntimeStore\.getState\(\)\.reasoningEffort;/,
  );
  const statusApply = readSrc("features/chat/lib/apply-inference-status-to-store.ts");
  assert.match(
    statusApply,
    /const effortToClamp =\s*\(pinHoldsLiveEffort\(\) \? takeEffortDisplacedByPin\(\) : null\) \?\?\s*prevState\.reasoningEffort;/,
  );
  assert.match(
    statusApply,
    /\? clampReasoningEffortToLevels\(effortToClamp, reasoningEffortLevels\)\s*: clampLocalReasoningEffort\(effortToClamp\);/,
  );
  assert.doesNotMatch(chatPage, /takeEffortDisplacedByPin\(\) : null/);
  // parseInt stops at "e", so 1e5 read as 1.
  assert.match(
    settingsDialog,
    /const typedCap = Math\.round\(Number\(maxTokens\.trim\(\)\)\);/,
  );
  assert.doesNotMatch(settingsDialog, /Number\.parseInt\(/);
  assert.match(
    settingsDialog,
    /const efforts = externalReasoningTakesEffort\(reasoning\)\s*\? reasoning\.reasoningEffortLevels\.filter\(\(level\) => level !== "none"\)\s*: \[\];/,
  );
  const capabilities = readSrc("features/chat/provider-capabilities.ts");
  assert.match(
    capabilities,
    /export function externalReasoningTakesEffort\([\s\S]{0,120}?return caps\.supportsReasoning && caps\.reasoningStyle === "reasoning_effort";/,
  );
  assert.match(
    capabilities,
    /if \(!externalReasoningTakesEffort\(caps\)\) return current;/,
  );
  const infoDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-info-dialog.tsx",
  );
  assert.match(
    infoDialog,
    /const takesEffort = externalReasoningTakesEffort\(reasoning\);/,
  );
  assert.match(
    infoDialog,
    /const effortLevels = takesEffort\s*\? reasoning\.reasoningEffortLevels\.filter\(\(level\) => level !== "none"\)\s*: \[\];/,
  );
  assert.match(
    settingsDialog,
    /capDraft !== null && Number\.isFinite\(typedCap\) && typedCap > 0\s*\? \{ maxTokens: clampCap\(typedCap\) \}\s*: \{\}/,
  );
  assert.match(settingsDialog, /"Remember settings per model" is off/);
  assert.doesNotMatch(settingsDialog, /placeholder=/);
  // Merges are per key, so a blank cap could not clear the stored value.
  assert.match(
    settingsDialog,
    /capDraft \?\? String\(clampCap\(remembered\?\.maxTokens \?\? chatMaxTokens\)\)/,
  );
  assert.doesNotMatch(settingsDialog, /blank follows the connection/);
});

test("the cap offers the bounds every request is clamped to", () => {
  const settingsDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-settings-dialog.tsx",
  );
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const sheet = readSrc("features/chat/chat-settings-sheet.tsx");
  assert.match(sheet, /getExternalMaxOutputTokens\(/);
  assert.match(adapter, /getExternalMinOutputTokens\(externalProvider\?\.providerType\),/);
  assert.match(
    settingsDialog,
    /const minCap = getExternalMinOutputTokens\(providerType\);/,
  );
  assert.match(
    settingsDialog,
    /const maxCap = getExternalMaxOutputTokens\(\s*providerType,\s*modelId,\s*connectionMaxOutputTokens,\s*\)/,
  );
  assert.match(
    settingsDialog,
    /const clampCap = \(value: number\) =>\s*Math\.min\(Math\.max\(value, minCap\), maxCap\);/,
  );
  assert.match(
    pickers,
    /provider\.maxOutputTokens \?\? null,/,
  );
  assert.match(
    settingsDialog,
    /useSyncExternalStore\(subscribeModelCatalog, modelCatalogVersion\);/,
  );
});

test("live effort edits use the shared runtime action", () => {
  const dialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-settings-dialog.tsx",
  );
  assert.match(dialog, /reconcilePinnedReasoningEffort\(\{/);
  assert.doesNotMatch(dialog, /useChatRuntimeStore\.setState/);
  assert.doesNotMatch(dialog, /state\.setReasoningEffort/);
});

test("a row with no heading over it still names its connection", () => {
  assert.match(pickers, /renderConnectedModelRow\(model, true\)/);
  assert.match(pickers, /renderConnectedModelRow\(model, !headed\)/);
  assert.match(pickers, /<span className="block text-ui-10 mt-1">\s*\{model\.providerName\}/);
  assert.doesNotMatch(pickers, /leadingSlot/);
  assert.match(
    pickers,
    /const leading = formatDot \? <FormatTag \{\.\.\.formatDot\} \/> : null;/,
  );
  assert.match(
    pickers,
    /<TooltipContent\s*side="right"\s*className="tooltip-compact max-w-\[calc\(15rem\*var\(--ui-space-scale,1\)\)\] break-words"/,
  );
});

test("the store write merges per key and reaches the live params", () => {
  const runtime = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(
    runtime,
    /saveSettingsPatch\(\{ inferenceParamsByModel: \{ \[modelId\]: patch \} \}\)/,
  );
  assert.match(runtime, /const live = state\.params\.checkpoint === modelId;/);
  assert.match(
    runtime,
    /\/\/ \/api\/chat\/settings request was out\.\s*saveSettingsPatch\(/,
  );
  // Held as a patch: the wholesale overlay would drop the model's other settings.
  assert.match(
    runtime,
    /if \(!state\.settingsHydrated\) \{\s*modelParamEditsBeforeHydration\.set\(modelId, \{/,
  );
  assert.match(
    runtime,
    /for \(const \[modelId, patch\] of modelParamEditsBeforeHydration\) \{\s*hydrated\[modelId\] = \{ \.\.\.hydrated\[modelId\], \.\.\.patch \};/,
  );
  assert.match(
    runtime,
    /locallyRememberedModels\.add\(modelId\);\s*\}\s*modelParamEditsBeforeHydration\.clear\(\);/,
  );
  assert.match(
    runtime,
    /if \(liveChanged\) getChangedInferenceParams\(liveParams, state\.params\);/,
  );
  assert.match(
    runtime,
    /const liveParams = live\s*\? restoreThreadScopedParams\(\{ \.\.\.state\.params, \.\.\.patch \}\)\s*: null;/,
  );
  assert.match(
    runtime,
    /const liveChanged =\s*liveParams !== null &&\s*shouldAdvanceQueuedSettingsEpoch\(state\.params, liveParams\);/,
  );
  assert.match(
    runtime,
    /liveChanged\s*\? \{\s*params: liveParams,\s*queuedSettingsEpoch: state\.queuedSettingsEpoch \+ 1,/,
  );
});

test("a pinned reasoning effort wins, unless the catalogue withdrew it", () => {
  const effort = readSrc(
    "features/model-picker/components/model-selector/model-reasoning-effort.ts",
  );
  assert.match(
    effort,
    /if \(!effort \|\| \(allowed && !allowed\.includes\(effort\)\)\) return null;/,
  );
  const caps = readSrc("features/chat/provider-capabilities.ts");
  assert.match(
    caps,
    /if \(pinned && levels\.includes\(pinned as ReasoningEffortLevel\)\) \{\s*return pinned as ReasoningEffortLevel;/,
  );
  assert.match(
    caps,
    /if \(caps\.defaultEffort && levels\.includes\(caps\.defaultEffort\)\) \{/,
  );
  const chatPage = readSrc("features/chat/chat-page.tsx");
  assert.match(
    chatPage,
    /externalReasoningTakesEffort\(reasoningCaps\)\s*\? pinnedReasoningEffort\(value, effortLevels\)/,
  );
  assert.match(
    chatPage,
    /externalReasoningTakesEffort\(reasoningCaps\)\s*\? pinnedReasoningEffort\(inferenceParams\.checkpoint, effortLevels\)/,
  );
  assert.match(
    chatPage,
    /if \(pinnedEffort && nextReasoningEffort !== store\.reasoningEffort\) \{\s*noteEffortDisplacedByPin\(store\.reasoningEffort\);/,
  );
  assert.match(
    chatPage,
    /if \(pinnedEffort && nextReasoningEffort !== state\.reasoningEffort\) \{\s*noteEffortDisplacedByPin\(state\.reasoningEffort\);/,
  );
  assert.match(
    chatPage,
    /\}, \[externalProvidersForChat, inferenceParams\.checkpoint, settingsHydrated\]\);/,
  );
  assert.doesNotMatch(chatPage, /const anthropicTopEffort =/);
});

test("the info box is a dialog, so it is wide enough and closable", () => {
  const infoDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-info-dialog.tsx",
  );
  assert.match(infoDialog, /<DialogContent className="sm:max-w-xl">/);
  assert.doesNotMatch(infoDialog, /AlertDialog/);
  assert.match(infoDialog, /className="break-all font-mono"/);
  assert.doesNotMatch(infoDialog, /own settings/);
  assert.doesNotMatch(infoDialog, /paramsByModel/);
  assert.doesNotMatch(infoDialog, /model-reasoning-effort/);
  assert.doesNotMatch(infoDialog, /checkpointId/);
  assert.doesNotMatch(pickers, /checkpointId=\{infoModel\.model\.id\}/);
});

test("a row's name starts where its heading's label does", () => {
  // ml-4 + pl-3.5 must sum to 30px to align with the heading label.
  assert.match(
    pickers,
    /cn\(downloadedRowShellClassName\(isSelected\), "ml-4"\)/,
  );
  assert.match(
    pickers,
    /className=\{cn\(downloadedRowButtonClassName, "pl-3\.5"\)\}/,
  );
  assert.doesNotMatch(pickers, /reserveLeadingSlot=\{true\}/);
  assert.match(
    pickers,
    /group\/heading flex items-center justify-between gap-1 px-2\.5 pb-1 pt-3/,
  );
  assert.match(
    pickers,
    /<span className="flex min-w-0 items-center gap-1\.5 text-ui-10 font-semibold/,
  );
  const listLabel = /className=\{cn\(\s*"flex items-center justify-between gap-1 px-2\.5 pb-1",\s*divider \? "mt-3 border-t border-border pt-3" : "pt-3",/;
  assert.match(pickers, listLabel);
  assert.match(
    pickers,
    /<span className="flex items-center gap-1\.5 text-ui-10 font-semibold/,
  );
});

test("reasoning is read through the resolver the composer uses", () => {
  const infoDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-info-dialog.tsx",
  );
  const settingsDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-settings-dialog.tsx",
  );
  // Not the catalogue directly: `openai_codex` is not a models.dev namespace.
  for (const source of [infoDialog, settingsDialog]) {
    assert.match(
      source,
      /getExternalReasoningCapabilities\(providerType, modelId, \{\s*isReasoningProvider,\s*reasoningConfig,\s*baseUrl,\s*apiType,\s*\}\)/,
    );
    assert.match(source, /\(level\) => level !== "none"/);
  }
  assert.doesNotMatch(settingsDialog, /entry\?\.reasoning \? entry\.efforts/);
  assert.match(pickers, /provider\.isReasoningModel === true,/);
  assert.match(pickers, /apiType: externalApiTypeById\.get\(model\.providerId\)/);
  assert.match(pickers, /apiType=\{settingsModel\.apiType\}/);
  assert.match(pickers, /apiType=\{infoModel\.apiType\}/);
});

test("a Codex connection resolves against OpenAI's catalogue", async () => {
  const { resolveModelCatalogEntry } = await import("../src/features/chat/model-catalog.ts");
  const viaOpenAI = resolveModelCatalogEntry("openai", "gpt-4-turbo");
  assert.ok(viaOpenAI !== null, "the fixture model must be in the bundled snapshot");
  assert.deepEqual(resolveModelCatalogEntry("openai_codex", "gpt-4-turbo"), viaOpenAI);
});

test("the published context window reaches the row and the info box", () => {
  const catalog = readSrc("features/chat/model-catalog.ts");
  const snapshot = readSrc("features/chat/model-catalog-snapshot.ts");
  const infoDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-info-dialog.tsx",
  );
  assert.match(snapshot, /context\?: number;/);
  assert.match(
    catalog,
    /if \(entry\.contextLength != null \|\| !snapshot\) return entry;/,
  );
  assert.match(
    catalog,
    /\.find\(\(context\) => typeof context === "number" && context > 0\)/,
  );
  assert.doesNotMatch(pickers, /ctx`\]/);
  assert.doesNotMatch(marks, /formatContextLength/);
  assert.doesNotMatch(marks, /contextLength/);
  assert.match(infoDialog, /<Field label="Context window">/);
  assert.match(infoDialog, /tokens\(entry\.contextLength\)/);
});

test("a served catalogue cannot take away a context window it has no field for", async () => {
  const { resolveModelCatalogEntry, setModelsDevCatalog } = await import(
    "../src/features/chat/model-catalog.ts"
  );
  const bundled = resolveModelCatalogEntry("openai", "gpt-4-turbo")?.contextLength;
  assert.ok(
    typeof bundled === "number" && bundled > 0,
    "the bundled snapshot must publish a window for this model, else the rest proves nothing",
  );

  setModelsDevCatalog({
    fetched_at: Date.now(),
    providers: { openai: { "gpt-4-turbo": { input: ["text"], context: 999_999 } } },
  } as never);
  assert.equal(resolveModelCatalogEntry("openai", "gpt-4-turbo")?.contextLength, 999_999);

  // A backend payload without context must not erase the bundled window.
  setModelsDevCatalog({
    fetched_at: Date.now(),
    providers: { openai: { "gpt-4-turbo": { input: ["text"] } } },
  } as never);
  assert.equal(resolveModelCatalogEntry("openai", "gpt-4-turbo")?.contextLength, bundled);

  setModelsDevCatalog({ fetched_at: Date.now(), providers: {} } as never);
});

test("connection saves write back the live store, not the render snapshot", () => {
  const liveWrites = providersDialog.match(
    /onProvidersChange\(\s*\[?\s*(\.\.\.)?useExternalProvidersStore\.getState\(\)\.providers\.(map|filter)\(/g,
  );
  assert.equal(liveWrites?.length, 3);
  assert.match(providersDialog, /models: keepSavedModels \? undefined : modelsToSave,/);
  assert.match(providersDialog, /availableModels: keepSavedModels \? undefined : availableModelsToSave,/);
  assert.match(providersDialog, /updated\.models\?\.length \? updated\.models : existing\.models/);
  assert.match(providersDialog, /updated\.available_models\?\.length\s*\?\s*updated\.available_models\s*:\s*existing\.availableModels/);
});
