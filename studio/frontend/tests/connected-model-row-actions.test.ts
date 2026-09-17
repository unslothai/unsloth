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
    /pinnedConnectedRows\.map\(\(model\) =>\s*renderConnectedModelRow\(model, true\)/,
  );
  assert.match(
    pickers,
    /group\.models\.map\(\(model\) =>\s*renderConnectedModelRow\(model\)/,
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
  // The picked model already carries into a new chat, so a default-for-new-chats pin said nothing.
  assert.doesNotMatch(pickers, /Use by default in new chats/);
  // Unticking a model belongs to the connection form, which owns that list.
  assert.doesNotMatch(pickers, /Hide from this list/);
  // Nor is the connection: it is one setting shared by every row under the heading.
  assert.doesNotMatch(pickers, /label: "Connection settings"/);
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
  // Blank is an absence, not a zero the request would then send as the cap.
  assert.match(
    settingsDialog,
    /Number\.isFinite\(cap\) && cap > 0 \? \{ maxTokens: cap \} : \{\}/,
  );
  // And it says so when the setting that restores them is switched off.
  assert.match(settingsDialog, /"Remember settings per model" is off/);
  // No example prompt: a placeholder reads as a value that is already set.
  assert.doesNotMatch(settingsDialog, /placeholder=/);
});

test("the store write merges per key and reaches the live params", () => {
  const runtime = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(
    runtime,
    /saveSettingsPatch\(\{ inferenceParamsByModel: \{ \[modelId\]: patch \} \}\)/,
  );
  // Editing the model that is loaded has to land now, not on the next switch back.
  assert.match(runtime, /const live = state\.params\.checkpoint === modelId;/);
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
  const chatPage = readSrc("features/chat/chat-page.tsx");
  assert.match(
    chatPage,
    /const pinnedEffort = pinnedReasoningEffort\(value, effortLevels\);/,
  );
  // Ahead of the catalogue and per-provider defaults: the user set this one deliberately.
  assert.match(
    chatPage,
    /pinnedEffort as typeof catalogDefaultEffort\) \?\?\s*catalogDefaultEffort/,
  );
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
