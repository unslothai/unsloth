// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  CONNECTED_TAB,
  DEVICE_TAB,
  FAVORITES_TAB,
  OFF_LEVEL,
  type PickerRow,
  RECOMMENDED_TAB,
  chooseQuant,
  dedupeRows,
  downloadedQuants,
  effortLabel,
  filterRows,
  findChosenRow,
  isChatTask,
  localSection,
  orderBySection,
  pinCoversId,
  repoSection,
  sliderLevels,
  sliderValue,
} from "../src/components/assistant-ui/composer-model-picker/picker-model.ts";
import { readSrc } from "./helpers/kit.ts";

const row = (over: Partial<PickerRow>): PickerRow => ({
  key: over.id ?? "k",
  id: "unsloth/Qwen3-4B-GGUF",
  label: "Qwen3-4B-GGUF",
  tab: DEVICE_TAB,
  kind: "gguf",
  ...over,
});
const none = () => false;

test("the chosen row matches by id or by a checkpoint alias, on device before Recommended", () => {
  const rows = [
    row({
      key: "hub",
      id: "unsloth/gemma-4-E2B-it-GGUF",
      tab: RECOMMENDED_TAB,
      kind: "hub",
    }),
    row({
      key: "dev",
      id: "unsloth/gemma-4-E2B-it-GGUF",
      aliases: ["/cache/snap"],
    }),
    row({
      key: "ext",
      id: "external::p1::gpt-5",
      tab: CONNECTED_TAB,
      kind: "external",
    }),
  ];
  assert.equal(findChosenRow(rows, "UNSLOTH/gemma-4-E2B-it-GGUF")?.key, "dev");
  assert.equal(findChosenRow(rows, "/cache/snap")?.key, "dev");
  assert.equal(findChosenRow(rows, "external::p1::gpt-5")?.key, "ext");
  assert.equal(
    findChosenRow([rows[0]], "unsloth/gemma-4-E2B-it-GGUF")?.key,
    "hub",
  );
  assert.equal(findChosenRow(rows, null), undefined);
});

test("a tab lists its own rows; Favorites lists each favorite once; search narrows", () => {
  const rows = [
    row({ key: "dev-1", id: "a/Qwen3-4B", label: "Qwen3-4B" }),
    row({
      key: "hub-1",
      id: "a/Qwen3-4B",
      label: "Qwen3-4B",
      tab: RECOMMENDED_TAB,
      kind: "hub",
    }),
    row({ key: "dev-2", id: "a/Llama-3", label: "Llama-3" }),
    row({
      key: "ext-1",
      id: "external::p::gpt",
      label: "gpt-5",
      tab: CONNECTED_TAB,
      kind: "external",
      providerName: "OpenAI",
    }),
  ];
  assert.deepEqual(
    filterRows(rows, DEVICE_TAB, "", none).map((r) => r.key),
    ["dev-1", "dev-2"],
  );
  assert.deepEqual(
    filterRows(rows, CONNECTED_TAB, "openai", none).map((r) => r.key),
    ["ext-1"],
  );
  const favorites = new Set(["a/Qwen3-4B", "external::p::gpt"]);
  assert.deepEqual(
    filterRows(rows, FAVORITES_TAB, "", (r) => favorites.has(r.id)).map(
      (r) => r.key,
    ),
    ["dev-1", "ext-1"],
  );
  assert.deepEqual(
    filterRows(rows, DEVICE_TAB, "llama", none).map((r) => r.key),
    ["dev-2"],
  );
});

test("an on-device model also lists under its maker's tab", () => {
  const rows = [
    row({
      key: "g",
      id: "unsloth/gemma-4-E2B-it-GGUF",
      groups: ["maker:google"],
    }),
    row({ key: "q", id: "unsloth/Qwen3.8-27B-GGUF", groups: ["maker:qwen"] }),
    row({ key: "s", id: "hubertsiuzdak/snac_24khz" }),
  ];
  assert.deepEqual(
    filterRows(rows, "maker:google", "", none).map((r) => r.key),
    ["g"],
  );
  assert.deepEqual(
    filterRows(rows, DEVICE_TAB, "", none).map((r) => r.key),
    ["g", "q", "s"],
  );
});

test("favorites read the header picker's keys: a repo or any of its quants", () => {
  const pinned = ["unsloth/gemma-4-E2B-it-GGUF::UD-Q4_K_XL", "org/plain-model"];
  assert.equal(pinCoversId(pinned, "unsloth/gemma-4-E2B-it-GGUF"), true);
  assert.equal(pinCoversId(pinned, "org/plain-model"), true);
  assert.equal(pinCoversId(pinned, "unsloth/gemma-4-E2B"), false);
});

test("rows dedupe by id, first wins", () => {
  const out = dedupeRows([
    row({ key: "x", id: "A/b" }),
    row({ key: "y", id: "a/B" }),
  ]);
  assert.deepEqual(
    out.map((r) => r.key),
    ["x"],
  );
});

test("only chat tasks list on device; an untagged repo counts as chat", () => {
  assert.equal(isChatTask(null), true);
  assert.equal(isChatTask("image-text-to-text"), true);
  assert.equal(isChatTask("text-to-image"), false);
  assert.equal(isChatTask("text-to-speech"), false);
});

test("a GGUF pick loads a whole quant: preferred, then default, then the first", () => {
  const variants = [
    { quant: "Q8_0", downloaded: true },
    { quant: "Q4_K_M", downloaded: true },
    { quant: "Q2_K", downloaded: true, partial: true },
    { quant: "BF16", downloaded: false },
  ];
  const onDisk = downloadedQuants(variants);
  assert.deepEqual(
    onDisk.map((v) => v.quant),
    ["Q8_0", "Q4_K_M"],
  );
  assert.equal(chooseQuant(onDisk, ["q4_k_m", "Q8_0"])?.quant, "Q4_K_M");
  assert.equal(chooseQuant(onDisk, [null, "BF16"])?.quant, "Q8_0");
  assert.equal(chooseQuant([], ["Q8_0"]), undefined);
});

test("the effort ladder leads with Off when thinking can be turned off", () => {
  assert.deepEqual(
    sliderLevels(["none", "low", "medium", "high"], true).map((l) => l.value),
    [OFF_LEVEL, "low", "medium", "high"],
  );
  assert.deepEqual(
    sliderLevels(["none", "low"], false).map((l) => l.value),
    ["none", "low"],
  );
  assert.equal(sliderValue(false, "high", true), OFF_LEVEL);
  assert.equal(sliderValue(true, "none", true), OFF_LEVEL);
  assert.equal(sliderValue(true, "high", true), "high");
  assert.equal(effortLabel("xhigh"), "Extra High");
  assert.equal(effortLabel("xhigh", "claude-opus-4-6-20260101"), "Max");
});

test("thinking without effort levels is a switch, as Preserve thinking is", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /const showThinkingSwitch = reasoning\.supportsReasoning && !showEffort;/,
  );
  assert.match(picker, /<SwitchRow\s+label="Thinking"/);
  assert.match(picker, /<SwitchRow\s+label="Preserve thinking"/);
  assert.match(picker, /<EffortSlider\s+title="Effort"/);
  const slider = readSrc(
    "components/assistant-ui/composer-model-picker/effort-slider.tsx",
  );
  assert.doesNotMatch(slider, /<canvas/);
  const css = readSrc("index.css");
  assert.match(
    css,
    /\.composer-effort-fill \{\s*background-color: var\(--control-accent\);/,
  );
});

test("a loaded row ejects from its dot, its menu and the collapsed view", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /<LoadedDot label=\{row\.label\} onEject=\{\(\) => eject\(row\)\} \/>/,
  );
  assert.match(picker, /key: "eject"/);
  assert.match(picker, /aria-label=\{`Eject \$\{chosenName\}`\}/);
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /eject: handleEject,/);
});

test("Run settings open inside the composer picker, not the header picker", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /<ModelLoadSettingsAction/);
  assert.match(picker, /<ModelRowMenu/);
  assert.match(
    picker,
    /openConfig\(modelConfigTarget\(row\.id, meta, row\.label\)\)/,
  );
  assert.match(picker, /<ModelConfigPage/);
  assert.match(picker, /forceReload: true/);
  // The Hub's Run lands here too, now that chat has no header picker.
  assert.match(picker, /source\?\.configRequest/);
  assert.match(
    picker,
    /onConfigRequestAdopted\?\.\(configRequest\.requestId\)/,
  );
  const page = readSrc("features/chat/chat-page.tsx");
  assert.doesNotMatch(page, /openRunSettings/);
  assert.match(page, /configRequest: modelConfigRequest,/);
  assert.match(
    page,
    /open: active && \(modelSelectorOpen \|\| modelConfigRequest !== null\)/,
  );
});

test("On Device groups rows like the header picker, NPU and custom folders included", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /useNpuCatalog\(/);
  assert.match(picker, /kind: "npu",\s+section: "npu"/);
  assert.match(picker, /useCustomFolders\(/);
  assert.match(
    picker,
    /section: localSection\(lora\.source, lora\.baseModel\)/,
  );
  const folders = readSrc(
    "components/assistant-ui/composer-model-picker/custom-folders.tsx",
  );
  assert.match(folders, /addScanFolderToStore/);
  assert.match(folders, /removeScanFolderFromStore/);
  assert.match(folders, /<FolderBrowser/);
});

test("the composer renders the model chip where the Thinking pill was", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /<ComposerModelPicker side=\{menuSide\} \/>/);
  assert.doesNotMatch(thread, /const ReasoningToggle\b/);
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(
    page,
    /<ComposerModelPickerContext\.Provider value=\{composerPickerSource\}>/,
  );
  assert.match(page, /onPick: handleCheckpointChange,/);
});

test("favorites are one list in both pickers, drawn as stars", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /usePinnedModelsStore\(/);
  assert.match(picker, /usePinnedConnectedModelsStore\(/);
  const full = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.doesNotMatch(full, /\bPinIcon\b/);
  assert.doesNotMatch(full, /pinLabel: "Pin"/);
  const menu = readSrc(
    "features/model-picker/components/model-selector/model-row-menu.tsx",
  );
  assert.match(menu, /pin\.pinned \? StarOffIcon : StarIcon/);
});

test("the picker opens with its list unless the user collapsed it", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.doesNotMatch(picker, /<Popover[^>]*\bmodal\b/);
  assert.match(picker, /localStorage\.getItem\(EXPANDED_KEY\) !== "false"/);
  assert.match(picker, /useState\(readExpanded\)/);
  assert.match(picker, /onClick=\{\(\) => setExpanded\(false\)\}/);
});

test("On Device sections follow the header picker's buckets and order", () => {
  assert.equal(repoSection("unsloth/gemma-4-E2B-it-GGUF"), "unsloth");
  assert.equal(repoSection("Qwen/Qwen3-8B"), "other");
  assert.equal(localSection("training", undefined), "fineTuned");
  assert.equal(localSection("exported", undefined), "fineTuned");
  assert.equal(localSection("local", "LM Studio"), "lmstudio");
  assert.equal(localSection("local", "Local models"), "modelsDir");
  assert.equal(localSection("local", "Ollama"), "custom");
  assert.equal(localSection("local", "Custom Folders"), "custom");
  const ordered = orderBySection([
    { id: "a", section: "custom" as const },
    { id: "b", section: "unsloth" as const },
    { id: "c", section: "npu" as const },
    { id: "d", section: "unsloth" as const },
    { id: "e", section: "recommended" as const },
  ]);
  assert.deepEqual(
    ordered.map((row) => row.id),
    ["b", "d", "c", "a", "e"],
  );
});

test("a quant still resolving never overrides a newer pick", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /await rowQuants\(target\);\s+if \(pickSeq\.current !== seq\) \{\s+return;/,
  );
  assert.match(
    picker,
    /if \(list\.length === 1\) \{\s+if \(pickSeq\.current === seq\)/,
  );
  assert.match(picker, /const seq = supersede\(\);/);
});

test("composer search keeps Mac-native repos, as the header picker did", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /isMac &&\s+isRecommendableFormat\(r\.id, r\.isGguf, true, r\.quantMethod\)/,
  );
});

test("on-disk quant lists are re-read when the picker opens or inventory changes", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /\[open, inventory\.cachedGguf, loraModels\]/);
  assert.match(
    picker,
    /quantsReadAt\.current\.get\(row\.key\) !== quantsEpoch/,
  );
});

test("resident rows key, pick and eject by their serving checkpoint", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /const servedAs = .*=> m\.checkpoint \?\? m\.id;/);
  assert.match(picker, /key: `loaded:\$\{checkpoint\}`,\s+id: checkpoint,/);
  assert.match(picker, /loadedId: servedAs\(hit\)/);
});

test("connected models keep their Model settings dialog", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /<ConnectedModelSettingsDialog/);
  assert.match(picker, /onConfigure=\{\(\) => openConnectedSettings\(row\)\}/);
});

test("quant lookups send the HF token and keep partial quants in download offers", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /const token = useHfTokenStore\.getState\(\)\.token \|\| undefined;\s+const res = await listGgufVariants\(target\.id, token,/,
  );
  assert.match(picker, /\(v\) => target\.remote \|\| v\.partial !== true/);
});

test("On Device sort and format filter follow the header picker's options", async () => {
  const { matchesFormat, sortDeviceRows } = await import(
    "../src/components/assistant-ui/composer-model-picker/picker-model.ts"
  );
  const row = (
    label: string,
    format: "gguf" | "mlx" | "checkpoint" | undefined,
    sizeBytes: number,
    updatedAt: number,
  ) => ({
    key: label,
    id: label,
    label,
    tab: DEVICE_TAB,
    kind: "gguf" as const,
    format,
    sizeBytes,
    updatedAt,
  });
  const rows = [
    row("b", "gguf", 5, 1),
    row("a", "mlx", 9, 3),
    row("c", "checkpoint", 1, 2),
  ];
  assert.deepEqual(
    sortDeviceRows(rows, "name", () => -1).map((r) => r.label),
    ["a", "b", "c"],
  );
  assert.deepEqual(
    sortDeviceRows(rows, "size", () => -1).map((r) => r.label),
    ["a", "b", "c"],
  );
  assert.deepEqual(
    sortDeviceRows(rows, "downloaded", () => -1).map((r) => r.label),
    ["a", "c", "b"],
  );
  assert.deepEqual(
    sortDeviceRows(rows, "recent", (r) => (r.label === "c" ? 10 : -1)).map(
      (r) => r.label,
    ),
    ["c", "a", "b"],
  );
  assert.deepEqual(
    rows.filter((r) => matchesFormat(r, "safetensors")).map((r) => r.label),
    ["c"],
  );
  assert.equal(matchesFormat({ kind: "npu", format: undefined }, "npu"), true);
  assert.equal(matchesFormat({ kind: "gguf", format: "gguf" }, "mlx"), false);
});

test("the composer picker carries the header picker's remaining row actions", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // Own models delete and reveal.
  assert.match(picker, /deleteFineTunedModel\(/);
  assert.match(picker, /revealFineTunedModel\(/);
  assert.match(picker, /DELETABLE_LOCAL_SOURCES\.has\(local\.source\)/);
  // Interrupted downloads stay listed and resume.
  assert.match(picker, /isDownloaded: !row\.partial/);
  assert.match(picker, /remote: row\.partial === true/);
  // GGUF updates go through the download manager.
  assert.match(picker, /downloadManager\.requestStart\(/);
  assert.match(picker, /<TransportConflictDialog/);
  // Eject all, connected info and copy, the Hub hint and a removed connection's name.
  assert.match(picker, /source\?\.ejectAll\(\)/);
  assert.match(picker, /<ConnectedModelInfoDialog/);
  assert.match(picker, /Copy model ID/);
  assert.match(picker, /<HubFailureHint/);
  assert.match(picker, /missingExternalModel\(/);
  // MLX repos list their variants and pick as checkpoints.
  assert.match(picker, /mlx: res\?\.variant_format === "mlx"/);
  // Load times are stamped here now that Chat has no header picker.
  assert.match(
    picker,
    /useModelLoadTimes\(loadStampId\(checkpoint, source\?\.loaded\)\)/,
  );
});

test("on-device row menus open with the model's details", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /<ModelFacts facts=\{modelFacts\(row, onDisk\)\} \/>/);
  for (const label of ["Format", "Size", "Parameters", "From"]) {
    assert.match(picker, new RegExp(`\\["${label}"`));
  }
  const menu = readSrc(
    "features/model-picker/components/model-selector/model-row-menu.tsx",
  );
  assert.match(menu, /details \? "w-60" : "w-48"/);
});

test("the details menu shows modalities with the Model Hub's coloured icons", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /<CapabilityGlyph capability=\{c\} \/>/);
  assert.match(picker, /detectHubCapabilities\(undefined, task, id\)/);
});

test("an MLX variant's gear keeps its format, and exported GGUFs delete per quant", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /ggufMeta\(row, variant, onDisk, isGguf\)/);
  assert.match(picker, /variant\.downloaded === true,\s+!offer\.mlx,/);
  assert.match(picker, /onDisk && \(row\.gguf \|\| exportedGguf\)/);
  assert.match(
    picker,
    /source: "exported",\s+exportType: "gguf",\s+ggufVariant: variant\.quant,/,
  );
});
