// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Connected rows used to be a name and nothing else, while every On Device row carried modality
// badges, a pin and a gear. These pin the parts that closed that gap, and the one boundary it
// must not be closed across: the two pin lists stay apart.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, readText } from "./helpers/kit.ts";

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
  // The same component the On Device rows use, so the glyph set cannot drift into a second one.
  assert.match(
    pickers,
    /const renderConnectedModelRow = \(\s*model: ExternalModelOption,/,
  );
  assert.match(
    pickers,
    /capabilities=\{marks\.capabilities\}\s*\n\s*showVision=\{marks\.vision\}/,
  );
  // The format-dot slot stays reserved though nothing goes in it: that is what starts the names
  // on the same line as every other list's.
  assert.match(pickers, /reserveLeadingSlot=\{true\}/);
  // The connection is named by the group heading, so it is not repeated on each of its rows.
  assert.doesNotMatch(pickers, /leadingBadge=\{\s*<ApiProviderLogo/);
  // Both groups render through it. Only the pinned one drags: the groups below are sorted, so a
  // drop there could not be honoured.
  assert.match(
    pickers,
    /pinnedConnectedRows\.map\(\(model\) =>\s*renderConnectedModelRow\(model, true, true\)/,
  );
  assert.match(
    pickers,
    /group\.models\.map\(\(model\) =>\s*renderConnectedModelRow\(model, false, !headed\)/,
  );
});

test("capabilities are keyed by the provider's own model id", () => {
  // `model.id` is the external:: address and `model.name` a label OpenRouter rewrites, so a
  // lookup on either misses every catalogue entry.
  assert.match(
    pickers,
    /parseExternalModelId\(model\.id\)\?\.modelId \?\? model\.name/,
  );
  assert.match(
    pickers,
    /connectedModelMarks\(\{\s*providerType: model\.providerType,\s*modelId: providerModelId,\s*baseUrl,/,
  );
  // A catalogue lands after first paint, so the marks have to be re-read when it does.
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
  assert.match(marks, /resolveModelCatalogEntry\(providerType, modelId\);/);
  assert.match(marks, /const modalities = entry\?\.inputModalities \?\? null;/);
  assert.match(marks, /providerSupportsBuiltinImageGeneration\(/);
  // Unknown is not a promise: only an explicit true draws the eye.
  assert.doesNotMatch(
    marks,
    /vision: providerModelSupportsVision\([^)]*\) \?\?/,
  );
});

test("connected pins are a separate list from the On Device ones", () => {
  assert.match(connectedPins, /"unsloth_pinned_connected_models"/);
  assert.match(onDevicePins, /"unsloth_pinned_models"/);
  // An external id carries "::", which this reads as the repo/quant separator, so a pin filed in
  // the On Device store comes back out as a phantom pinned quant on that tab.
  assert.match(onDevicePins, /const sep = key\.indexOf\("::"\)/);
  assert.doesNotMatch(connectedPins, /"unsloth_pinned_models"/);
  // So a connected row reaches for its own toggle and never the On Device one.
  assert.doesNotMatch(pickers, /togglePinned\(model\.id\)/);
  assert.doesNotMatch(pickers, /pinKey\(model\.id\)/);
  assert.match(pickers, /onToggle: \(\) => togglePinnedConnected\(model\.id\)/);
});

test("a pin moves the row out of its provider group", () => {
  // Copying it instead would list the model twice under one search.
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
  // A folded group's rows leave the order too, or the arrows walk through rows off screen.
  assert.match(
    pickers,
    /if \(collapsedConnectedGroups\.has\(group\.providerId\)\) continue;/,
  );
});

test("nothing on a connected row opens local run settings", () => {
  // ModelConfigPage holds KV cache dtype and GPU layers, which a remote model has none of.
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
  // Closing the dialog drops it, or it replays on the next visit.
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
  // The panel mounts before its first sync lands, so an id it cannot find yet is not a miss.
  assert.match(providersDialog, /if \(!provider\) return;/);
  assert.match(
    providersDialog,
    /openedProviderRef\.current = openProviderId;\s*void editProvider\(provider\);\s*onOpenProviderConsumed\?\.\(\);/,
  );
  // The empty-list auto-open latch stays set, as tests/connections-empty-opens-form.test.ts asks.
  assert.doesNotMatch(
    providersDialog,
    /autoOpenedAddFormRef\.current = false;/,
  );
});

test("the row menu carries the connected actions", () => {
  for (const label of ["Model info", "Copy model ID"]) {
    assert.match(pickers, new RegExp(`label: "${label}"`));
  }
  // Through the shared helper, which handles the desktop shell and falls back to execCommand.
  // navigator.clipboard is undefined there and over plain HTTP on a LAN address, and reading
  // .writeText off it throws where no catch on the promise can report it.
  assert.match(
    pickers,
    /if \(await copyToClipboard\(providerModelId\)\) \{\s*toast\.success/,
  );
  assert.doesNotMatch(pickers, /navigator\.clipboard\s*\n?\s*\.?writeText/);
  // The picked model already carries into a new chat, so a default-for-new-chats pin said nothing.
  assert.doesNotMatch(pickers, /Use by default in new chats/);
  // Unticking a model belongs to the connection form, which owns that list.
  assert.doesNotMatch(pickers, /Hide from this list/);
  // The connection is the heading's, not the row's: a row under one would offer the same
  // destination twice. Only a row with no heading over it carries the entry, and the group
  // heading's own gear is what every other row reaches it through.
  assert.match(
    pickers,
    /\.\.\.\(headless && onConfigureConnection\s*\? \[\s*\{\s*key: "connection",\s*label: "Connection settings",/,
  );
  assert.match(
    pickers,
    /onSelect: \(\) => onConfigureConnection\(model\.providerId\),/,
  );
});

test("a connection stays reachable when its heading is gone", () => {
  // The name sort drops every heading, and a connection whose every model is pinned loses its
  // group entirely, so the heading gear cannot be the only way in.
  assert.match(pickers, /if \(connectedSort !== "name"\) return groups;/);
  assert.match(
    pickers,
    /for \(const model of connectedMatches\) \{\s*if \(pinnedConnectedSet\.has\(model\.id\)\) continue;/,
  );
  assert.match(pickers, /headless && onConfigureConnection/);
});

test("the connection's own settings hang off the group heading", () => {
  // Beside the fold chevron, in the heading's own right-hand cluster.
  assert.match(
    pickers,
    /<div className="-mr-2 flex shrink-0 items-center -space-x-0\.5">\s*\{onConfigure \?/,
  );
  // Hover-only, as a row's gear is, while the chevron beside it stays on screen.
  assert.match(
    pickers,
    /HEADING_ACTION_CLASS,\s*"opacity-0 group-hover\/heading:opacity-100/,
  );
  assert.match(pickers, /className=\{HEADING_ACTION_CLASS\}\s*>\s*\{collapsed \?/);
  // The same box and glyph the row gutter's buttons use, so the columns line up.
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
  // Named, since a picker can hold several connections.
  assert.match(
    pickers,
    /configureLabel=\{`\$\{group\.providerName\} connection settings`\}/,
  );
  // The Pinned group spans connections, so there is no single one for its heading to open.
  assert.match(
    pickers,
    /label="Pinned"\s*collapsed=\{pinnedConnectedCollapsed\}\s*onToggle=\{\(\) =>\s*setPinnedConnectedCollapsed/,
  );
});

test("the gear opens the model's own settings", () => {
  // Not a menu entry as well: the gutter already holds a settings button.
  assert.match(
    pickers,
    /tooltip="Model settings"\s*onConfigure=\{\(\) =>\s*setSettingsModel\(\{/,
  );
  assert.doesNotMatch(pickers, /label: "Model settings/);
});

test("the connected sort and modality filter drive the list", () => {
  assert.match(pickers, /CONNECTED_SORT_OPTIONS/);
  assert.match(pickers, /CONNECTED_MODALITY_OPTIONS/);
  // Sorting by name flattens the provider groups rather than leaving headings of one row each.
  assert.match(pickers, /if \(connectedSort !== "name"\) return groups;/);
});

test("per-model prompt and cap reuse the memory Chat already keeps", () => {
  const settingsDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-settings-dialog.tsx",
  );
  // Not a second store: the editor writes into paramsByModel, which rememberParamsPerModel has
  // been restoring per checkpoint all along.
  assert.match(settingsDialog, /setRememberedParamsForModel\(checkpointId, \{/);
  assert.doesNotMatch(settingsDialog, /localStorage/);
  // Only the fields the user touched: this is a patch, and an untouched one would put whatever
  // the dialog happened to be showing over the stored value. null is untouched, so an untouched
  // field keeps reading the store even when hydration lands while the dialog is open.
  assert.match(settingsDialog, /const systemPrompt = promptDraft \?\? remembered\?\.systemPrompt \?\? "";/);
  assert.match(
    settingsDialog,
    /const effort = effortDraft \?\? pinnedEffort \?\? FOLLOW_CHAT;/,
  );
  assert.match(
    settingsDialog,
    /\.\.\.\(promptDraft !== null \? \{ systemPrompt: promptDraft \} : \{\}\),/,
  );
  // An untouched effort writes nothing: another tab can change the pin while this is open, and a
  // save of an unrelated field would put the value the select opened with back over it, and reset
  // the composer's own Think level while it was at it.
  assert.match(
    settingsDialog,
    /if \(effortDraft !== null\) \{\s*const pinned = effortDraft === FOLLOW_CHAT \? null : effortDraft;\s*setModelReasoningEffort\(checkpointId, pinned\);/,
  );
  // Which is the listener that makes that reachable.
  const effortStore = readSrc(
    "features/model-picker/components/model-selector/model-reasoning-effort.ts",
  );
  assert.match(
    effortStore,
    /window\.addEventListener\("storage", \(event\) => \{/,
  );
  // And the chat subscribes to the active model's pin for the same reason: the normalization
  // reads it through a getState helper, so a cross-tab change reached the dialog but not the
  // composer. Its own effect, since rerunning that whole block would reset the pills too.
  const chatPage = readSrc("features/chat/chat-page.tsx");
  assert.match(
    chatPage,
    /const activePinnedEffort = useModelReasoningEffortStore\(\s*\(state\) => state\.effortByModel\[inferenceParams\.checkpoint\],/,
  );
  assert.match(
    chatPage,
    /if \(appliedPinnedEffort\.current === activePinnedEffort\) return;/,
  );
  // And it advances the queued-settings epoch, or a prompt waiting on startup dispatches with
  // the level it captured before the change.
  assert.match(
    chatPage,
    /useChatRuntimeStore\.setState\(\(state\) => \(\{\s*reasoningEffort: next,\s*queuedSettingsEpoch: state\.queuedSettingsEpoch \+ 1,/,
  );
  // One reconciler for every trigger that can dislodge the pin, the catalogue included: a refresh
  // decides which levels a stored pin is legal against, so it can make an ignored pin the valid
  // one, and the pin effect's own guard sees no change in the stored string.
  assert.match(
    chatPage,
    /reconcilePinnedReasoningEffort\(\{\s*checkpoint: inferenceParams\.checkpoint,\s*caps,\s*providerType: provider\?\.providerType,\s*\}\);\s*\}, \[activePinnedEffort,/,
  );
  assert.match(
    chatPage,
    /reasoningFieldsAfterCatalogRefresh\(useChatRuntimeStore\.getState\(\), caps\),\s*\);[\s\S]{0,220}?reconcilePinnedReasoningEffort\(\{/,
  );
  // Whether the level came from a pin is asked of the store, never inferred from the trigger:
  // absence is a clear for the pin effect but not for a refresh, and it is a clear the user can
  // also cause from the composer, where resolving afresh would undo the level just chosen.
  assert.match(chatPage, /if \(!pinned && !pinHoldsLiveEffort\(\)\) return;/);
  // Switching to an unpinned model resolves from the chat's own level, not from the outgoing
  // model's pin, which is what the live value holds.
  assert.match(
    chatPage,
    /current:\s*!pinnedEffort && pinHoldsLiveEffort\(\)\s*\? \(takeEffortDisplacedByPin\(\) \?\? store\.reasoningEffort\)\s*: store\.reasoningEffort,/,
  );
  assert.match(
    chatPage,
    /current:\s*!pinnedEffort && pinHoldsLiveEffort\(\)\s*\? \(takeEffortDisplacedByPin\(\) \?\? state\.reasoningEffort\)\s*: state\.reasoningEffort,/,
  );
  // A pin in force owns the live level outright, whether or not it displaced anything when it
  // was applied: pinning the level already in the store recorded nothing, and the thread opened
  // next had its own level held back behind that pin with no record to hand back.
  assert.match(
    readSrc("features/chat/stores/chat-runtime-store.ts"),
    /export function pinHoldsLiveEffort\(\): boolean \{\s*return \(\s*pinOwnsLiveReasoningEffort\(useChatRuntimeStore\.getState\(\)\) \|\|\s*effortDisplacedByPin !== null\s*\);/,
  );
  // And the chat's own level is recorded as the snapshot is held back, so the pin can be handed
  // back the level of the chat that is open now rather than the one before it.
  assert.match(
    readSrc("features/chat/stores/chat-runtime-store.ts"),
    /if \(key === "reasoningEffort" && pinOwnsLiveReasoningEffort\(state\)\) \{[\s\S]{0,500}?effortDisplacedByPin = value as ReasoningEffort;/,
  );
  // Leaving a pinned model for a local one hands the chat's level to the clamp, which narrows it
  // to the local model's ladder and would otherwise carry the pin's level onto that model. Taken
  // where the model has actually become resident, never where one was picked: picking an uncached
  // model only queues a download, and the load can still abort at a lease, a confirmation or a
  // token prompt without the model ever changing.
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
  // Both clamps read the one value, so neither can keep clamping the pin's.
  assert.match(
    statusApply,
    /\? clampReasoningEffortToLevels\(effortToClamp, reasoningEffortLevels\)\s*: clampLocalReasoningEffort\(effortToClamp\);/,
  );
  // And the pick itself takes nothing, so an abort leaves the pin applied with nothing to undo.
  assert.doesNotMatch(chatPage, /takeEffortDisplacedByPin\(\) : null/);
  // A number field accepts scientific notation and parseInt stops at the "e", so 1e5 was read as
  // 1 and saved clamped to the provider minimum.
  assert.match(
    settingsDialog,
    /const typedCap = Math\.round\(Number\(maxTokens\.trim\(\)\)\);/,
  );
  assert.doesNotMatch(settingsDialog, /Number\.parseInt\(/);
  // Blank or junk is an absence, not a zero the request would then send as the cap.
  assert.match(
    settingsDialog,
    /capDraft !== null && Number\.isFinite\(typedCap\) && typedCap > 0\s*\? \{ maxTokens: clampCap\(typedCap\) \}\s*: \{\}/,
  );
  // And it says so when the setting that restores them is switched off.
  assert.match(settingsDialog, /"Remember settings per model" is off/);
  // No example prompt: a placeholder reads as a value that is already set.
  assert.doesNotMatch(settingsDialog, /placeholder=/);
  // The cap field always holds a number. A blank would have to mean "forget the cap", and
  // neither merge can express that: the patch merges per key and the server deep-merges the
  // settings row, so an omitted key keeps the old value and the clear goes nowhere.
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
  // The same two helpers the chat's own Max Tokens control bounds itself with, and the adapter
  // clamps every outbound request to, so a stored cap cannot differ from the one actually sent.
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
  // Clamped on save, not just hinted by the input's attributes.
  assert.match(
    settingsDialog,
    /const clampCap = \(value: number\) =>\s*Math\.min\(Math\.max\(value, minCap\), maxCap\);/,
  );
  // The connection's own cap lowers the model's documented one, so it has to reach the editor.
  assert.match(
    pickers,
    /provider\.maxOutputTokens \?\? null,/,
  );
  // And an OpenRouter cap arrives with the live catalogue, after this has rendered.
  assert.match(
    settingsDialog,
    /useSyncExternalStore\(subscribeModelCatalog, modelCatalogVersion\);/,
  );
});

test("editing the live model's effort reaches the chat now", () => {
  const settingsDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-settings-dialog.tsx",
  );
  const chatPage = readSrc("features/chat/chat-page.tsx");
  // The pin is consumed on a switch, and a switch to the model already loaded returns first, so
  // without this an edit would not apply until the user switched away and back.
  assert.match(
    chatPage,
    /if \(isSameLoadedModel && !meta\?\.forceReload\) \{\s*return;/,
  );
  assert.match(
    settingsDialog,
    /state\.params\.checkpoint === checkpointId,/,
  );
  // Through the same resolver the switch uses, so clearing the pin recomputes the default
  // instead of leaving the cleared level in force, and a level the model rejects cannot get in.
  assert.match(
    settingsDialog,
    /useChatRuntimeStore\.setState\(\(state\) => \(\{\s*reasoningEffort: next,\s*queuedSettingsEpoch: state\.queuedSettingsEpoch \+ 1,/,
  );
  // Clearing resolves from the level the pin displaced, not the live one: the live one IS the
  // pin, and a ladder with no catalogue default and no "medium" falls back to clamping it, so
  // "Follow chat" would have kept the level just cleared.
  assert.match(
    settingsDialog,
    /current: pinned \? chatEffort : \(takeEffortDisplacedByPin\(\) \?\? chatEffort\),/,
  );
  assert.match(
    settingsDialog,
    /if \(pinned && next !== chatEffort\) noteEffortDisplacedByPin\(chatEffort\);/,
  );
  // The record lives in the chat store, so the user's own Think change invalidates it.
  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(store, /export function noteEffortDisplacedByPin\(/);
  assert.match(store, /effortDisplacedByPin \?\?= current;/);
  // Nothing before hydration: the live level is still the store's own default there, and with
  // ??= that first record would outlast the chat's real level landing a moment later.
  assert.match(
    store,
    /if \(!useChatRuntimeStore\.getState\(\)\.settingsHydrated\) return;\s*effortDisplacedByPin \?\?= current;/,
  );
  assert.match(
    store,
    /own level, so whatever a pin displaced before is history\.\s*effortDisplacedByPin = null;/,
  );
  // And the pin goes with it. Every caller is the user picking a level for the model on screen,
  // which is what its pin claims to decide, so leaving the pin ran this level now and put the
  // pin's back on the next switch, with the row naming a level the composer did not show.
  assert.match(
    store,
    /if \(checkpoint && pinned !== undefined && pinned !== reasoningEffort\) \{\s*setModelReasoningEffort\(checkpoint, null\);/,
  );
  // reasoningEffort is a thread-scoped key, so a snapshot taken while a pin is in force would
  // store the model's level as the chat's own and clearing the pin could not undo it.
  assert.match(
    store,
    /!explicitlyEditedThreadFields\.has\("reasoningEffort"\) &&\s*pinOwnsLiveReasoningEffort\(useChatRuntimeStore\.getState\(\)\)\s*\) \{\s*const onRecord = reasoningEffortOnRecord\(\);\s*if \(onRecord !== undefined\) settings\.reasoningEffort = onRecord;/,
  );
  // Same for the capture that becomes every snapshot-less chat's default.
  assert.match(
    store,
    /if \(pinOwnsLiveReasoningEffort\(state\)\) \{\s*const onRecord = reasoningEffortOnRecord\(\);\s*if \(onRecord === undefined\) delete captured\.reasoningEffort;\s*else captured\.reasoningEffort = onRecord;/,
  );
  // And a thread snapshot being applied records its level without putting it over the pin, which
  // outranks the chat for as long as its model is selected.
  assert.match(
    store,
    /applied\[key\] = value;[\s\S]{0,400}?if \(key === "reasoningEffort" && pinOwnsLiveReasoningEffort\(state\)\) \{/,
  );
  // With the snapshot pin-free, clearing resolves from the chat's own recorded level first: that
  // one survives a reload and a thread switch, which the in-memory record does not.
  assert.match(
    store,
    /export function takeEffortDisplacedByPin\(\)[\s\S]{0,320}?return \(\s*activeThreadScopedSettings\?\.reasoningEffort \?\?\s*globalThreadScopedDefaults\?\.reasoningEffort \?\?\s*displaced\s*\);/,
  );
  // setState, not the action: setReasoningEffort persists the value through
  // setScalarSettingVersion as the chat-wide preference, which is the setting a per-model pin
  // exists to override. The pin itself is stored by setModelReasoningEffort.
  assert.doesNotMatch(settingsDialog, /state\.setReasoningEffort/);
  const runtime = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(
    runtime,
    /const writeGlobal = \(\) => \{\s*scalarSettingMutationVersions\[key\] \+= 1;\s*saveSettingsPatch\(\{ \[key\]: value \}\);/,
  );
});

test("a row with no heading over it still names its connection", () => {
  // Pinned rows and the name-sorted flat list have no provider heading, and two connections can
  // serve one model id, so those rows carry the logo the heading would have carried.
  assert.match(pickers, /renderConnectedModelRow\(model, true, true\)/);
  assert.match(pickers, /renderConnectedModelRow\(model, false, !headed\)/);
  assert.match(
    pickers,
    /leadingSlot=\{\s*headless \? \(\s*<ApiProviderLogo/,
  );
  // In the slot the format dot would have used, so the names stay on one line.
  assert.match(
    pickers,
    /const leading = formatDot \? <FormatTag \{\.\.\.formatDot\} \/> : \(leadingSlot \?\? null\);/,
  );
  // And the connection's name on hover, which is the only thing that tells two of one provider
  // type apart.
  assert.match(pickers, /<span className="block text-ui-10 mt-1">\s*\{model\.providerName\}/);
});

test("the store write merges per key and reaches the live params", () => {
  const runtime = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(
    runtime,
    /saveSettingsPatch\(\{ inferenceParamsByModel: \{ \[modelId\]: patch \} \}\)/,
  );
  // Editing the model that is loaded has to land now, not on the next switch back.
  assert.match(runtime, /const live = state\.params\.checkpoint === modelId;/);
  // A save while the initial settings request is still out has to survive it, so the write is not
  // gated on hydration: the patch can only set the keys it names and the server merges per key.
  assert.match(
    runtime,
    /\/\/ \/api\/chat\/settings request was out\.\s*saveSettingsPatch\(/,
  );
  // Held as a patch, not a row. locallyRememberedModels is overlaid wholesale, which would
  // replace the server's entry with one naming only the keys the dialog touched and drop that
  // model's temperature, top-p and seed.
  assert.match(
    runtime,
    /if \(!state\.settingsHydrated\) \{\s*modelParamEditsBeforeHydration\.set\(modelId, \{/,
  );
  assert.match(
    runtime,
    /for \(const \[modelId, patch\] of modelParamEditsBeforeHydration\) \{\s*hydrated\[modelId\] = \{ \.\.\.hydrated\[modelId\], \.\.\.patch \};/,
  );
  // Complete once merged, so a later response takes the wholesale overlay.
  assert.match(
    runtime,
    /locallyRememberedModels\.add\(modelId\);\s*\}\s*modelParamEditsBeforeHydration\.clear\(\);/,
  );
  // And the live keys are fenced the way a slider edit fences its own, or the response in flight
  // puts the global set back over them.
  assert.match(
    runtime,
    /if \(liveChanged\) getChangedInferenceParams\(liveParams, state\.params\);/,
  );
  // systemPrompt is one of the chat's own keys, so the live overlay goes through the restore a
  // switch uses: the open chat outranks the model it is running, and writing past that would
  // send the model's prompt for the rest of the chat and store it as the chat's own next.
  assert.match(
    runtime,
    /const liveParams = live\s*\? restoreThreadScopedParams\(\{ \.\.\.state\.params, \.\.\.patch \}\)\s*: null;/,
  );
  // The epoch as setParams advances it: a queued prompt or a paste still reading its file
  // captured the old prompt and cap, and nothing else would tell it they had moved.
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
  // Asking for a level the provider rejects fails the request, so an unavailable pin falls back.
  assert.match(
    effort,
    /if \(allowed && allowed\.length > 0 && !allowed\.includes\(effort\)\) return null;/,
  );
  const caps = readSrc("features/chat/provider-capabilities.ts");
  // Ahead of the catalogue and the per-provider defaults: the user set this one deliberately.
  assert.match(
    caps,
    /if \(pinned && levels\.includes\(pinned as ReasoningEffortLevel\)\) \{\s*return pinned as ReasoningEffortLevel;/,
  );
  assert.match(
    caps,
    /if \(caps\.defaultEffort && levels\.includes\(caps\.defaultEffort\)\) \{/,
  );
  // One resolver for every caller, so a reload and a resync cannot answer differently from a
  // switch. The pin used to be read by the switch alone.
  const chatPage = readSrc("features/chat/chat-page.tsx");
  assert.match(
    chatPage,
    /const pinnedEffort = pinnedReasoningEffort\(value, effortLevels\);/,
  );
  assert.match(
    chatPage,
    /const pinnedEffort = pinnedReasoningEffort\(\s*inferenceParams\.checkpoint,\s*effortLevels,\s*\);/,
  );
  // Both record the level the pin takes the place of, so clearing it can put that back.
  assert.match(
    chatPage,
    /if \(pinnedEffort && nextReasoningEffort !== store\.reasoningEffort\) \{\s*noteEffortDisplacedByPin\(store\.reasoningEffort\);/,
  );
  assert.match(
    chatPage,
    /if \(pinnedEffort && nextReasoningEffort !== state\.reasoningEffort\) \{\s*noteEffortDisplacedByPin\(state\.reasoningEffort\);/,
  );
  // The normalization effect reruns on reload and on every provider resync, and it is the one
  // that used to put the provider default back over the pin.
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
  // AlertDialog caps at max-w-md and asks a question; this reports, and needs Dialog's close X.
  assert.match(infoDialog, /<DialogContent className="sm:max-w-xl">/);
  assert.doesNotMatch(infoDialog, /AlertDialog/);
  // An id has no spaces to wrap at, so break-words would let it run past the edge.
  assert.match(infoDialog, /className="break-all font-mono"/);
});

test("the row pill starts where its heading does", () => {
  // The heading is px-2.5; without the same inset the pill began 10px to its left.
  assert.match(
    pickers,
    /cn\(downloadedRowShellClassName\(isSelected\), "ml-2\.5"\)/,
  );
});

test("reasoning is read through the resolver the composer uses", () => {
  const infoDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-info-dialog.tsx",
  );
  const settingsDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-settings-dialog.tsx",
  );
  // Reading the catalogue directly reported every Codex model as unpublished while the composer
  // had it thinking: `openai_codex` is not a namespace models.dev publishes.
  for (const source of [infoDialog, settingsDialog]) {
    assert.match(
      source,
      /getExternalReasoningCapabilities\(providerType, modelId, \{\s*isReasoningProvider,\s*baseUrl,\s*\}\)/,
    );
    // "none" is the off switch, not a level on offer.
    assert.match(source, /\(level\) => level !== "none"/);
  }
  assert.doesNotMatch(settingsDialog, /entry\?\.reasoning \? entry\.efforts/);
  // And a self-hosted endpoint's only reasoning signal is the flag on its connection.
  assert.match(pickers, /provider\.isReasoningModel === true,/);
});

test("a Codex connection resolves against OpenAI's catalogue", () => {
  const catalog = readSrc("features/chat/model-catalog.ts");
  assert.match(
    catalog,
    /CATALOG_NAMESPACE_ALIASES: Record<string, string> = \{\s*openai_codex: "openai",\s*\}/,
  );
  // At the snapshot lookup, so every caller of it inherits the alias.
  assert.match(
    catalog,
    /\| undefined \{\s*providerType = catalogNamespace\(providerType\);/,
  );
});

test("the published context window reaches the row and the info box", () => {
  const catalog = readSrc("features/chat/model-catalog.ts");
  const snapshot = readSrc("features/chat/model-catalog-snapshot.ts");
  const infoDialog = readSrc(
    "features/model-picker/components/model-selector/connected-model-info-dialog.tsx",
  );
  assert.match(snapshot, /context\?: number;/);
  // models.dev publishes it on the snapshot only, so a live entry backfills from there rather
  // than reporting a model that has one as having none.
  assert.match(
    catalog,
    /if \(entry\.contextLength != null \|\| !snapshot\) return entry;/,
  );
  assert.match(
    catalog,
    /\.find\(\(context\) => typeof context === "number" && context > 0\)/,
  );
  // Rounded on the row, exact in the info box.
  assert.match(marks, /export function formatContextLength/);
  assert.match(
    pickers,
    /\[`\$\{formatContextLength\(marks\.contextLength\)\} ctx`\]/,
  );
  assert.match(infoDialog, /<Field label="Context window">/);
  assert.match(infoDialog, /tokens\(entry\.contextLength\)/);
});

test("a served catalogue cannot take away a context window it has no field for", () => {
  const catalog = readSrc("features/chat/model-catalog.ts");
  // The served payload replaces the bundled entry per model, so before the backend carried the
  // field every model models.dev covers read as publishing no window at all.
  assert.match(catalog, /const context = entry\.context \?\? bundled\[id\]\?\.context;/);
  assert.match(
    catalog,
    /merged\[id\] = context == null \? entry : \{ \.\.\.entry, context \};/,
  );
  // And the backend now sends it, so a fresh payload needs no rescue.
  const trimmer = readText(
    "../../backend/core/inference/provider_model_capabilities.py",
  );
  assert.match(trimmer, /context = limit\.get\("context"\) if isinstance\(limit, dict\) else None/);
  assert.match(trimmer, /entry\["context"\] = context/);
});
