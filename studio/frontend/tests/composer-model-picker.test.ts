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
  isTestFixtureRepo,
  localSection,
  orderBySection,
  pickerAlignOffset,
  pickerDropOffset,
  pickerSide,
  pinCoversId,
  repoSection,
  sectionsHaveRows,
  sliderLevels,
  sliderValue,
  sortConnectedRows,
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
    /const showThinkingSwitch =\s*!pane && reasoning\.supportsReasoning && !showEffort;/,
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

test("a loaded row ejects from its menu and the collapsed view, never its dot", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // A Compare pane has nothing to eject.
  assert.match(picker, /const canEject = !pane \|\| pane\.loaded === true;/);
  // The dot is only a mark: hovering it never turns it into Eject.
  assert.match(picker, /<LoadedDot \/>/);
  assert.doesNotMatch(picker, /Loaded\. Click to eject/);
  assert.match(picker, /\.\.\.\(row\.loaded && canEject/);
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
  assert.match(
    page,
    /configRequest: modelConfigRequest \?\? ownedConfigRequest,/,
  );
  assert.match(
    page,
    /open:\s+active &&\s+view\.mode !== "compare" &&\s+\(modelSelectorOpen \|\| modelConfigRequest !== null\)/,
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
  // The arrow flips each key from its first direction.
  assert.deepEqual(
    sortDeviceRows(rows, "name", () => -1, true).map((r) => r.label),
    ["c", "b", "a"],
  );
  assert.deepEqual(
    sortDeviceRows(rows, "size", () => -1, false).map((r) => r.label),
    ["c", "b", "a"],
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
  assert.match(picker, /<ModelFacts\s+facts=\{modelFacts\(/);
  for (const label of ["Quant", "Disk space", "Parameters", "Supports"]) {
    assert.match(picker, new RegExp(`\\["${label}"`));
  }
  // The format rides on the quant ("UD-Q4_K_XL GGUF"); the section is not repeated.
  assert.match(picker, /\["Quant", withFormat\(onDisk\[0\]\.quant\)\]/);
  assert.doesNotMatch(picker, /\["From"/);
  const menu = readSrc(
    "features/model-picker/components/model-selector/model-row-menu.tsx",
  );
  assert.match(menu, /details \? "w-60" : "w-48"/);
});

test("the details menu shows modalities with the Model Hub's coloured icons", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /<CapabilityGlyph\s+capability=\{c\}\s+className=\{glyphScale\(c\.key, c\.label\)\}/,
  );
  assert.match(picker, /detectHubCapabilities\(undefined, task, id\)/);
});

test("an MLX variant's gear keeps its format, and exported GGUFs delete per quant", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /ggufMeta\(row, variant, onDisk, isGguf, hasVision\)/);
  assert.match(picker, /variant\.downloaded === true,\s+!offer\.mlx,/);
  assert.match(picker, /onDisk && \(row\.gguf \|\| exportedGguf\)/);
  assert.match(
    picker,
    /source: "exported",\s+exportType: "gguf",\s+ggufVariant: variant\.quant,/,
  );
});

test("the details menu lists every modality, preferring the backend's flags", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /known\?\.hasVideoInput/);
  assert.match(picker, /known\?\.speaks/);
  assert.match(picker, /known\?\.hasAudioInput \|\| known\?\.isAudio/);
  assert.match(picker, /"diffusion",\s+"embedding",/);
});

test("a search that finds only a custom-folder model is not reported empty", () => {
  const cedar = row({
    id: "/models/Cedar-7B",
    kind: "local",
    section: "custom",
  });
  const sections = ["unsloth", "fineTuned", "custom"] as const;
  assert.equal(sectionsHaveRows([...sections], [cedar]), true);
  assert.equal(sectionsHaveRows([...sections], []), false);
  // A row outside the listed sections does not count.
  assert.equal(sectionsHaveRows(["unsloth"], [cedar]), false);
});

test("the composer picker follows Settings > Chat > Model selection", () => {
  const src = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  for (const setting of [
    "expandQuantizations",
    "showAllQuantizations",
    "fitOnDeviceOnly",
    "showMemoryBar",
  ]) {
    assert.match(
      src,
      new RegExp(`(?:\\bs|getState\\(\\))\\.${setting}\\b`),
      `${setting} is not read`,
    );
  }
});

test("a downloaded MLX repo still resolves its precision folder", () => {
  const src = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // Discover rows list MLX variants whether or not they are on disk.
  assert.match(
    src,
    /row\.kind === "hub" && \(row\.hub\?\.isGguf \|\| row\.format === "mlx"\)/,
  );
  // On Device rows resolve one too, and a plain snapshot falls back to loading whole.
  assert.match(src, /row\.kind === "cached" && row\.format === "mlx"/);
  assert.match(src, /row\.kind === "cached" && !\(mlx && quant\)/);
});

test("cached MLX snapshots load whole and favorites show quant offers", () => {
  const src = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // The offer's whole-checkpoint fallback is hub-only, so a cached snapshot must not open one.
  assert.match(
    src,
    /row\.kind !== "cached" &&\s+\(offerAll \|\| \(target\.remote/,
  );
  // The Favorites wrapper renders an open offer as the plain list does.
  const fav = src.slice(src.indexOf("{...drag.rowProps(favoriteKey)}"));
  assert.match(fav.slice(0, 900), /\{quantList\}/);
  assert.match(src, /const quantList = offerOpen \? \(\s+renderOffer\(row\)/);
});

test("Discover keeps rows the Hub search matched by maker alias", () => {
  const gemma = row({
    id: "unsloth/gemma-4-E2B-it-GGUF",
    label: "gemma-4-E2B-it-GGUF",
    tab: RECOMMENDED_TAB,
    kind: "hub",
  });
  // A plain text filter drops it: "google" is in neither the id nor the label.
  assert.deepEqual(filterRows([gemma], RECOMMENDED_TAB, "google", none), []);
  assert.deepEqual(
    filterRows(
      [gemma],
      RECOMMENDED_TAB,
      "google",
      none,
      (r) => r.kind === "hub",
    ),
    [gemma],
  );
});

test("the open picker is one height on every tab, loaded model or not", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /\(isExpanded \|\| configTarget\) &&\s+"h-\[calc\(460px\*var\(--ui-space-scale,1\)\)\]"/,
  );
  assert.match(picker, /composer-model-list flex min-h-0 flex-1 flex-col/);
  // The model in use keeps a hover look, a hair darker than a hovered row.
  assert.match(
    picker,
    /rowCurrent \? CURRENT_ROW_BG : rowActive && "bg-sidebar-accent"/,
  );
});

test("a panel opening down rides up over the composer when the room below is short", () => {
  // Room below: the usual 8px gap.
  assert.equal(pickerDropOffset(400, 300, 900), 8);
  // 100px short: up by that much, over the composer.
  assert.equal(pickerDropOffset(500, 492, 900), -100);
  // Taller than the window: pinned 8px from its top.
  assert.equal(pickerDropOffset(500, 2000, 900), -492);
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /sideOffset=\{dropsDown \? dropOffset : PICKER_GAP\}/);
});

test("the picker's side is fixed per opening by the trigger, not the panel", () => {
  const vh = 900;
  // New chat (prefers below): a centred composer opens down, one near the foot opens up.
  assert.equal(pickerSide("bottom", { top: 500, bottom: 530 }, vh), "bottom");
  assert.equal(pickerSide("bottom", { top: 760, bottom: 790 }, vh), "top");
  // A docked composer opens up, unless the window leaves it under 180px above.
  assert.equal(pickerSide("top", { top: 840, bottom: 870 }, vh), "top");
  assert.equal(pickerSide("top", { top: 120, bottom: 150 }, vh), "bottom");
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // Radix's flip would re-judge the side whenever a tab or search changes the height.
  assert.match(picker, /side=\{openSide\}\s+avoidCollisions=\{false\}/);
});

test("the picker list glides to its new height instead of jumping", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // The element, not the ref: the panel mounts after the picker renders.
  assert.match(picker, /useAnimatedListHeight\(listEl\);/);
  assert.match(picker, /setListEl\(node\);/);
  assert.match(picker, /ref=\{attachList\}/);
  const glide = readSrc(
    "components/assistant-ui/composer-model-picker/animated-height.ts",
  );
  // Before paint, from the height on screen, and never under reduced motion.
  assert.match(glide, /new MutationObserver\(glide\)/);
  // The observer fires after the rows change, so at rest it starts from the remembered height.
  assert.match(
    glide,
    /const from = el\.style\.height \? el\.getBoundingClientRect\(\)\.height : shown;/,
  );
  assert.match(glide, /prefers-reduced-motion: reduce/);
});

test("each sort direction has words for its tooltip", async () => {
  const { DEVICE_SORT_DESC, DEVICE_SORT_ORDER } = await import(
    "../src/components/assistant-ui/composer-model-picker/picker-model.ts"
  );
  // The first direction of each key matches what its rows show before any flip.
  assert.equal(DEVICE_SORT_ORDER.recent.desc, "Newest first");
  assert.equal(DEVICE_SORT_ORDER.size.desc, "Largest first");
  assert.equal(DEVICE_SORT_ORDER.name.asc, "A to Z");
  assert.deepEqual(DEVICE_SORT_DESC, { recent: true, size: true, name: false });
});

test("the composer's Run settings keep Always load and the save status in their own row", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.doesNotMatch(picker, /dense=\{true\}/);
  const page = readSrc(
    "features/model-picker/components/model-config-page.tsx",
  );
  assert.doesNotMatch(page, /alwaysLoad=\{/);
  assert.match(page, /aria-label="Always load with these settings"/);
  const css = readSrc("index.css");
  // The footer sits at the panel's foot when the settings are short.
  assert.match(
    css,
    /\.composer-run-settings \[data-pinned-footer\] \{\s+margin-top: auto;/,
  );
});

test("a finished Discover search with no eligible results stays empty", async () => {
  const { discoverUsesFallback } = await import(
    "../src/components/assistant-ui/composer-model-picker/picker-model.ts"
  );
  // Browsing, or a search still waiting on results, shows the chat model list.
  assert.equal(discoverUsesFallback("", 0), true);
  assert.equal(discoverUsesFallback("", 12), true);
  assert.equal(discoverUsesFallback("whisper", 0), true);
  // Results came back but the filters rejected them all: no fallback, so "No models found".
  assert.equal(discoverUsesFallback("whisper", 7), false);
});

test("MLX repos marked only by their card resolve as MLX, on device and on Discover", async () => {
  const { hasMlxCardTag, isMlxId } = await import(
    "../src/features/hub/lib/format-filters.ts"
  );
  assert.equal(
    isMlxId("someone/Llama-4bit", hasMlxCardTag({ tags: ["mlx"] })),
    true,
  );
  assert.equal(
    isMlxId("someone/Llama-4bit", hasMlxCardTag({ libraryName: "mlx" })),
    true,
  );
  assert.equal(
    isMlxId("someone/Llama-4bit", hasMlxCardTag({ tags: [] })),
    false,
  );
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /hasMlxCardTag\(\{ tags: repo\.tags, libraryName: repo\.library_name \}\)/,
  );
  assert.match(
    picker,
    /isMlxId\(m\.id, mlxTaggedHub\.has\(m\.id\.toLowerCase\(\)\)\)/,
  );
  // The composer's own name-only rule is gone.
  assert.doesNotMatch(picker, /const isMlxId = /);
});

test("a per-quant delete reconciles quant favorites and settings keep the vision verdict", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  const del = picker.slice(
    picker.indexOf("const deleteQuantFor"),
    picker.indexOf("const openConnectedSettings"),
  );
  assert.match(del, /reconcileGgufPinsAfterDelete\(\s*repo\.repo_id,/);
  assert.match(picker, /ggufMeta\(row, variant, onDisk, isGguf, hasVision\)/);
  assert.match(
    picker,
    /!offer\.mlx,\s*offer\.hasVision \?\? row\.gguf\?\.has_vision,\s*\)/,
  );
  assert.match(picker, /!mlx,\s*hasVision \?\? row\.gguf\?\.has_vision,\s*\)/);
});

test("the search ignores IME confirmation keys, and Chat keeps an adopted handoff until the picker closes", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /event\.nativeEvent\.isComposing \|\| event\.keyCode === 229/,
  );
  const chat = readSrc("features/chat/chat-page.tsx");
  assert.match(
    chat,
    /configRequest: modelConfigRequest \?\? ownedConfigRequest,/,
  );
  assert.match(
    chat,
    /if \(handoffOwnsPicker && !modelSelectorOpen\) setOwnedConfigRequest\(null\);/,
  );
});

test("Compare panes pick through the composer picker, with no reasoning or eject", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  const header = page.slice(
    page.indexOf("function GeneralCompareHeader"),
    page.indexOf("function generalCompareLabels"),
  );
  assert.match(
    header,
    /<ComposerModelPicker\s+side="bottom"\s+align=\{side === "left" \? "start" : "end"\}/,
  );
  assert.match(
    header,
    /selection: \{\s*id: value,\s*ggufVariant: selectedGgufVariant \?\? null,\s*config: selectedConfig \?\? null,\s*\}/,
  );
  assert.doesNotMatch(header, /<ModelSelector/);
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /const checkpoint = pane \? pane\.id : runtimeCheckpoint;/,
  );
  assert.match(
    picker,
    /const reasoningChip = pane \? null : reasoning\.chipLabel;/,
  );
  assert.match(picker, /const chosenLoaded = pane\s*\? pane\.loaded === true/);
  // Reopened Run settings start from the pane's own config, not the saved one.
  assert.match(
    picker,
    /initialConfig=\{\s*pane\?\.config &&\s*modelConfigTargetIsResident\(\{\s*target: configTarget,\s*selectedId: pane\.id,/,
  );
});

test("a plain Compare pick runs with the model's saved settings", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  const header = page.slice(
    page.indexOf("function GeneralCompareHeader"),
    page.indexOf("function generalCompareLabels"),
  );
  assert.match(
    header,
    /meta\.config \|\| meta\.source === "external"\s*\? null\s*: resolveInitialConfig\(id, meta\.ggufVariant\);/,
  );
  assert.match(
    header,
    /remembered\?\.remembered \? \{ \.\.\.meta, config: remembered\.config \} : meta,/,
  );
});

test("Compare and the API page always show the picker's dropdown chevron", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.equal(
    picker.match(
      /\{\(chosenName \|\| modelLoading \|\| pending\) && !pane \? null : \(\s*<HugeiconsIcon\s*icon=\{ChevronDownStandardIcon\}/g,
    )?.length,
    2,
  );
});

test("a folder row the scan calls GGUF lists its quants", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /!lora\.isDirectGguf &&\s*\(lora\.isGguf === true \|\|/);
  assert.match(picker, /lora\.isDirectGguf \|\|\s*lora\.isGguf \|\|/);
});

test("the picker slides back inside a narrow window instead of hanging off its edge", () => {
  // A phone: the chip ends at 295px, so a 343px end-aligned panel would start at -48px.
  assert.equal(
    pickerAlignOffset("end", { left: 267, right: 295 }, 404, 375),
    -56,
  );
  // Compare's left pane opens rightward and would overrun the right edge.
  assert.equal(
    pickerAlignOffset("start", { left: 60, right: 200 }, 404, 420),
    -36,
  );
  // With room, the panel stays on its trigger.
  assert.equal(
    pickerAlignOffset("end", { left: 900, right: 1050 }, 404, 1280),
    0,
  );
  assert.equal(
    pickerAlignOffset("start", { left: 300, right: 450 }, 404, 1280),
    0,
  );
  assert.equal(pickerAlignOffset("end", null, 404, 375), 0);
});

test("the Connected tab lists every connection's models, favorites then newest first", () => {
  const ext = (id: string, providerId: string, providerName: string) =>
    row({
      key: id,
      id,
      label: id,
      tab: `ext:${providerId}`,
      groups: [CONNECTED_TAB],
      kind: "external",
      providerName,
    });
  const rows = [
    ext("gpt-5", "p2", "OpenAI"),
    ext("claude-opus", "p1", "Anthropic"),
    ext("claude-haiku", "p1", "Anthropic"),
    row({ key: "device", id: "unsloth/Qwen3-4B-GGUF" }),
  ];
  const connected = filterRows(rows, CONNECTED_TAB, "", none);
  // Without versions, the provider's order stands.
  assert.deepEqual(
    sortConnectedRows(connected).map((r) => r.key),
    ["claude-opus", "claude-haiku", "gpt-5"],
  );
  // Newest version first, favorites ahead of it.
  const gpt = [
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5.6-sol",
    "gpt-6-astra",
    "gpt-6-sol",
  ].map((id) => ext(id, "p2", "OpenAI"));
  assert.deepEqual(
    sortConnectedRows(gpt).map((r) => r.key),
    [
      "gpt-6-astra",
      "gpt-6-sol",
      "gpt-5.6-sol",
      "gpt-5.5",
      "gpt-5.4",
      "gpt-5.4-mini",
    ],
  );
  assert.deepEqual(
    sortConnectedRows(gpt, (r) => r.key === "gpt-5.4")
      .map((r) => r.key)
      .slice(0, 2),
    ["gpt-5.4", "gpt-6-astra"],
  );
  // Each connection keeps its own tab too.
  assert.deepEqual(
    filterRows(rows, "ext:p2", "", none).map((r) => r.key),
    ["gpt-5"],
  );
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /value: CONNECTED_TAB,\s+label: "Connected models"/);
  // Right after On Device, ahead of the maker tabs.
  assert.match(picker, /value: CONNECTED_TAB,[\s\S]{0,200}?\.\.\.makerTabs,/);
  assert.match(picker, /Only show models that fit/);
  // A maker gets a tab only with 3+ distinct models.
  assert.match(picker, /const MAKER_TAB_MIN_MODELS = 3;/);
  assert.match(
    picker,
    /entry\.ids\.add\(row\.id\.toLowerCase\(\)\)[\s\S]{0,200}?if \(ids\.size < MAKER_TAB_MIN_MODELS\)/,
  );
});

test("a Discover GGUF lists every quant, so a cached partial is never picked alone", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /offerAll \|\| row\.onDevice \|\| row\.kind === "hub"\s+\? \{ \.\.\.target, remote: true, everyQuant: true \}/,
  );
});

test("maker and connection tabs reorder by drag; a connection sharing a logo wears a cloud", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // The pinned four never move; only movable tabs drag, and the order is saved.
  assert.match(picker, /const pinned = out\.filter\(\(t\) => !t\.movable\);/);
  assert.match(picker, /draggable=\{tab\.movable\}/);
  assert.match(
    picker,
    /localStorage\.setItem\(TAB_ORDER_KEY, JSON\.stringify\(order\)\)/,
  );
  assert.match(
    picker,
    /t\.value\.startsWith\(EXT_PREFIX\) &&\s+t\.logo &&\s+\(logos\.get\(t\.logo\) \?\? 0\) > 1/,
  );
});

test("an interrupted quant counts as a variant, so the row names no quant", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /const manyQuants = onDisk\.length \+ partials\.length > 1;/,
  );
  assert.match(picker, /const soleQuant = manyQuants \? null : wholeQuant;/);
  // A click folds both open, as for two whole quants.
  assert.match(picker, /if \(variantCount\(row\) > 1\) \{\s+setFoldedQuants/);
});

test("an audio pick opens the Audio page instead of loading into Chat", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // Every pick passes the router before it reaches the page's handler.
  assert.match(
    picker,
    /if \(!routeMediaPick\.current\(id, meta\)\) \{\s+rawSource\.onPick\(id, meta\);/,
  );
  assert.match(
    picker,
    /audioPickSearch\(id, \{ \.\.\.meta, task: pickedTask, audioType \}\)/,
  );
  // Compare and the API page load what they pick, so they list none.
  assert.match(picker, /const listsAudio = !pane;/);
});

test("a partial quant beside loadable ones is listed with its delete menu", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /partials: listed\.filter\(\(v\) => v\.partial === true\)/,
  );
  assert.match(picker, /renderPartialQuants\(row, partials\)/);
  assert.match(
    picker,
    /variant\.partial === true && row\.gguf && row\.tab === DEVICE_TAB/,
  );
});

test("Discover filters from one menu, in On Device's style", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // One format filter for both tabs, as in the header picker.
  assert.equal((picker.match(/\{formatGroup\}/g) ?? []).length, 2);
  // A ticked circle, as Run settings' Always load.
  assert.match(picker, /setFitOnDeviceOnly\(!fitOnDeviceOnly\);/);
  // The menu takes the Model hub button's place on Discover.
  assert.match(
    picker,
    /viewed !== DEVICE_TAB &&\s+viewed !== RECOMMENDED_TAB &&\s+viewed !== CONNECTED_TAB \?/,
  );
});

test("Discover leaves out llama.cpp's stories260K test fixtures", () => {
  for (const id of [
    "ggml-org/stories260K",
    "ggml-org/test-model-stories260K-infill",
    "someone/stories260k-et3",
  ]) {
    assert.equal(isTestFixtureRepo(id), true, id);
  }
  assert.equal(isTestFixtureRepo("unsloth/Qwen3.5-4B-GGUF"), false);
  assert.equal(isTestFixtureRepo("unsloth/stories-model"), false);
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /!isHiddenModelId\(r\.id\) &&\s*!isTestFixtureRepo\(r\.id\)/,
  );
});

test("Enter on a row's own button is not a pick", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /if \(event\.key === "Enter"\) \{\s*if \(event\.target === event\.currentTarget\) \{\s*pick\(row\);/,
  );
});

test("trained native-audio adapters stay out of the picker unless merged; on-disk ones list", async () => {
  const { nativeAudioCheckpointIsLoadable } = await import(
    "../src/features/model-picker/components/model-selector/audio-picker-policy.ts"
  );
  assert.equal(
    nativeAudioCheckpointIsLoadable("moss_tts_local", "adapter"),
    false,
  );
  assert.equal(
    nativeAudioCheckpointIsLoadable("moss_tts_local", "merged"),
    true,
  );
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /\(lora\.audioType && !listsAudio\) \|\|\s*\(lora\.source !== "local" &&\s*!nativeAudioCheckpointIsLoadable\(lora\.audioType, lora\.exportType\)\)/,
  );
});

test("Only models that fit judges MLX and safetensors results by the checkpoint rule", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /searchRowFitsDevice\(\s*\{\s*id: r\.id,/);
  assert.match(
    picker,
    /const checkpointFits = hubCheckpointFits\.get\(row\.id\.toLowerCase\(\)\);\s*if \(checkpointFits !== undefined\) \{\s*return checkpointFits;/,
  );
});

test("a standalone resident GGUF opens Run settings under its bare path", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /isStandaloneGgufPath\(row\.id\)\s*\? \{ \.\.\.residentMeta\(row\), ggufVariant: undefined, isGguf: true \}\s*: residentMeta\(row\);/,
  );
});

test("an empty Connected tab offers Add connection, ChatGPT and Anthropic", async () => {
  const { newConnectionRequest, newConnectionRequestType } = await import(
    "../src/features/chat/external-providers.ts"
  );
  assert.equal(
    newConnectionRequestType(newConnectionRequest("openai_codex")),
    "openai_codex",
  );
  assert.equal(newConnectionRequestType(newConnectionRequest()), "");
  assert.equal(newConnectionRequestType("provider-123"), null);
  const { DEFAULT_PROVIDERS } = await import(
    "../src/components/assistant-ui/composer-model-picker/picker-model.ts"
  );
  assert.deepEqual(
    DEFAULT_PROVIDERS.map((p) => [p.label, p.connect]),
    [
      ["Connect ChatGPT", "openai_codex"],
      ["Connect Anthropic", "anthropic"],
    ],
  );
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // Connected is always a tab; no provider tab shows until it is connected.
  assert.match(
    picker,
    /\.\.\.\(connectionsOn \|\| \(externalModels\?\.length \?\? 0\) > 0/,
  );
  assert.doesNotMatch(picker, /SETUP_PREFIX/);
  assert.match(
    picker,
    /label: "Add connection",\s*primary: true,\s*row: 0,\s*run: \(\) => connectFrom\(\)/,
  );
  // Beside it, for owners: the Agents settings that connect coding agents to a local model.
  assert.match(
    picker,
    /\.\.\.\(isOwner\s*\? \[\s*\{\s*label: "Connect coding agent",/,
  );
  assert.match(picker, /openDialog\("agents"\)/);
  assert.match(
    picker,
    /icon: providerLogo\(provider\.type\),\s*row: 1 as const,\s*run: \(\) => connectFrom\(provider\.connect\)/,
  );
  assert.match(picker, /variant=\{action\.primary \? "dark" : "outline"\}/);
  // Picker actions use the shared Button.
  assert.equal(
    picker.match(/<Button\s+variant="destructive"\s+size="xs"/g)?.length,
    2,
  );
  assert.match(picker, /<Button\s+variant="ghost"\s+size="xs"/);
  assert.doesNotMatch(picker, /EJECT_PILL_HOVER/);
  const api = readSrc(
    "features/api-monitor/components/api-model-load-controls.tsx",
  );
  assert.match(api, /connections: false,/);
});

test("an empty On Device tab links to the Model hub's Discover models", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /title: "No models on this device",\s*hint: "Download models from Model hub",\s*actions: \[\s*\{\s*label: "Discover models",\s*icon: \(\s*<HugeiconsIcon\s*icon=\{DashboardCircleIcon\}/,
  );
  assert.match(
    picker,
    /navigate\(\{ to: "\/hub", search: \{ tab: "discover" \} \}\)/,
  );
});

test("on-device quant reads stay bounded and stop once the effect is torn down", async () => {
  const { readBounded } = await import(
    "../src/components/assistant-ui/composer-model-picker/picker-model.ts"
  );
  let inFlight = 0;
  let peak = 0;
  const seen: number[] = [];
  await readBounded(
    Array.from({ length: 40 }, (_, i) => i),
    6,
    async (i) => {
      inFlight += 1;
      peak = Math.max(peak, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 1));
      inFlight -= 1;
      return i;
    },
    (i) => seen.push(i),
    () => true,
  );
  assert.equal(peak, 6);
  assert.equal(seen.length, 40);

  let live = true;
  let started = 0;
  await readBounded(
    Array.from({ length: 40 }, (_, i) => i),
    6,
    async (i) => {
      started += 1;
      await new Promise((resolve) => setTimeout(resolve, 1));
      live = false;
      return i;
    },
    () => undefined,
    () => live,
  );
  assert.equal(started, 6);
});

test("Eject on the chosen model takes the host's guarded unload; other rows name their model", async () => {
  const { ejectTargetId } = await import(
    "../src/components/assistant-ui/composer-model-picker/picker-model.ts"
  );
  assert.equal(
    ejectTargetId(
      { id: "unsloth/a-GGUF", loadedId: "/cache/a.gguf" },
      "unsloth/a-GGUF",
    ),
    undefined,
  );
  assert.equal(
    ejectTargetId(
      { id: "unsloth/b-GGUF", loadedId: "/cache/b.gguf" },
      "unsloth/a-GGUF",
    ),
    "/cache/b.gguf",
  );
  assert.equal(ejectTargetId({ id: "unsloth/b" }, undefined), "unsloth/b");
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.doesNotMatch(picker, /source\?\.eject\(checkpoint/);
});

test("local audio checkpoints route to the Audio page by path and HF name", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /for \(const m of inventory\.localModels\) \{\s*put\(\[m\.id, m\.model_id\], m\.task, m\.model_id\);/,
  );
});

test("a new search or tab restarts the keyboard highlight before the active row is seated", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  const reset = picker.search(
    /useEffect\(\(\) => \{\s*setHighlight\(false\);\s*\}, \[query, viewed, setHighlight\]\);/,
  );
  const seat = picker.indexOf(
    "// Rows paging in under the pointer must not move the highlight.",
  );
  assert.ok(reset >= 0, "no highlight reset on query / tab change");
  assert.ok(
    reset < seat,
    "the reset must run before the effect that seats `active`",
  );
});

test("connected-model dialogs close when Chat goes off-route", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /if \(!pageActive\) \{\s*setConnectedSettings\(null\);\s*\}/,
  );
  assert.match(
    picker,
    /\{connectedSettings && pageActive\s*\? renderConnectedSettings\(/,
  );
});

test("an audio pick hands the Audio page the inventory's audio type", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /const audioType = meta\.audioType \?\? gguf\?\.audio_type \?\? cached\?\.audio_type;/,
  );
  assert.match(
    picker,
    /audioPickSearch\(id, \{ \.\.\.meta, task: pickedTask, audioType \}\)/,
  );
});
