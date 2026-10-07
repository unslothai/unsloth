// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { atDefaultUiScale } from "./helpers/kit.ts";

function read(path: string): string {
  // Lengths are compared in px, so read them at the default UI font size.
  return atDefaultUiScale(
    readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8"),
  );
}

const PICKERS = read(
  "../src/features/model-picker/components/model-selector/pickers.tsx",
);
const MODELS_TABLE = read("../src/features/hub/catalog/models-table.tsx");
const HUB_CARD = read("../src/features/hub/catalog/gguf-download-card.tsx");
const HUB_PAGE = read("../src/features/hub/hub-page.tsx");
const CATALOG = read(
  "../src/features/model-picker/components/model-selector/model-catalog.ts",
);
const RECOMMENDED = read(
  "../src/features/model-picker/components/model-selector/recommended-fit.ts",
);
const GPU_INFO = read("../src/hooks/use-gpu-info.ts");
const INSPECTOR = read("../src/features/hub/catalog/model-inspector.tsx");
const SIDEBAR = read("../src/components/app-sidebar.tsx");
const CSS = read("../src/index.css");

test("a downloaded row is marked the way the Hub marks one", () => {
  const start = PICKERS.indexOf("function DownloadedBadge()");
  const badge = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
  assert.ok(!badge.includes("Download01Icon"), "no download glyph");
  assert.match(badge, /size-\[5px\] rounded-full bg-status-success/);
  assert.match(badge, /aria-label="On device"/);
  assert.ok(
    MODELS_TABLE.includes("bg-status-success"),
    "and the Hub still uses that dot, so the two agree",
  );
});

test("the download glyph is gone from the picker entirely", () => {
  assert.ok(!PICKERS.includes("Download01Icon"));
});

test("the scoped badge column reserves the wider on-device marker", () => {
  // Video fits a 26px pill, 4px gap and 14px marker; narrower would shift columns.
  assert.ok(PICKERS.includes('badgeMid: "min-w-min min-[560px]:w-[44px]"'));
});

test("the unscoped badge column is sized per list, not to the union of both", () => {
  assert.ok(PICKERS.includes('badgeDevice: "min-w-min min-[560px]:w-[26px]"'));
  assert.ok(PICKERS.includes('badgeWide: "min-w-min min-[560px]:w-[44px]"'));
  // The partial mark sits with the name so it does not widen one row's badge slot.
  const gguf = PICKERS.slice(PICKERS.indexOf("const renderDownloadedGgufRow"));
  const row = gguf.slice(0, gguf.indexOf("\n  };"));
  assert.ok(row.includes('alignMeta="device"'));
  assert.ok(row.includes("showVision={c.has_vision"));
  assert.ok(row.includes("partial={isPartialRepo}"));
  assert.match(
    PICKERS,
    /\{alignMeta === "device" && partial \? \(\n\s*<span className="ml-\[max\(6px,6px\)\] flex shrink-0 items-center self-center">\n\s*<PartialBadge resumable=\{partialResumable\} \/>/,
  );
  assert.match(
    PICKERS,
    /\{showVision && <VisionBadge \/>\}\n\s*\{partial && alignMeta !== "device" \? \(\n\s*<PartialBadge resumable=\{partialResumable\} \/>/,
  );
  assert.match(
    PICKERS,
    /alignMeta === "device"\n\s*\? META_COLUMN\.badgeDevice\n\s*: META_COLUMN\.badgeWide/,
  );
});

test("a row's leading dot starts where its section label does", () => {
  // Dot centred in a 14px target: 10 - (14 - 5) / 2 = 5.5px aligns it with labels.
  assert.match(PICKERS, /py-1\.5 pl-\[5\.5px\] pr-2 text-left text-sm/);
  const label = PICKERS.slice(
    PICKERS.indexOf("flex items-center justify-between gap-1 px-2.5"),
  );
  assert.ok(label.startsWith("flex items-center justify-between gap-1 px-2.5"));
  assert.ok(
    PICKERS.includes(
      'className="flex size-[14px] shrink-0 items-center justify-center"',
    ),
  );
});

test("the parameter and size columns are sized to the ink they hold", () => {
  // formatBytes writes no space, so the widest size ("128GB") is 29.5px.
  assert.ok(PICKERS.includes('size: "min-w-min min-[560px]:w-[3.2em]"'));
  assert.match(
    PICKERS,
    /No space: "145MB" reads as one value beside the quant chip\./,
  );
});

test("the parameter column is fixed, so the quant column cannot drift row to row", () => {
  // 4.4em holds the widest param label at text-ui-10 so rows stay aligned.
  assert.ok(PICKERS.includes('param: "min-w-min min-[560px]:w-[4.4em]"'));
  assert.ok(PICKERS.includes('paramWide: "min-w-min min-[560px]:w-[5.2em]"'));
});

test("the parameter chip leads its column, so the modality gap is the cluster's own", () => {
  assert.match(
    PICKERS,
    /alignMeta === "hub"\n\s*\? cn\("justify-end", META_COLUMN\.paramWide\)\n\s*: cn\("justify-start", META_COLUMN\.param\)/,
  );
  assert.equal(PICKERS.split('className="pr-1"').length - 1, 3, "the custom folder rows");
});



test("the quant chip is flush right in its slot, so the chips read as one column", () => {
  assert.ok(PICKERS.includes('quant: "min-[560px]:max-w-[7.2em]"'));
  assert.match(PICKERS, /"flex shrink-0 items-center justify-end text-ui-9"/);
});

test("the quant chip rides in the meta cluster, not on the end of the name", () => {
  // One items-center rule in the meta cluster aligns chips structurally.
  const meta = PICKERS.slice(
    PICKERS.indexOf('"ml-auto flex shrink-0 items-center"'),
  );
  const quantSlot = meta.indexOf("META_COLUMN.quant");
  const badgeSlot = meta.indexOf("badgeColumn");
  assert.ok(quantSlot > 0, "the quant slot sits inside the meta cluster");
  assert.ok(quantSlot < badgeSlot, "and leads the badge column");
  assert.ok(
    !PICKERS.includes("justify-end self-center text-ui-9"),
    "no leftover baseline compensation",
  );
});

test("every chip in the row band pins the same height", () => {
  // Fixed chip height, since line box height scales with --ui-font-scale.
  for (const chip of ["QuantChip", "VisionBadge", "CapabilityIcons", "ParamChip"]) {
    const start = PICKERS.indexOf(`function ${chip}(`);
    assert.ok(start > 0, `${chip} exists`);
    const body = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
    assert.ok(body.includes("h-[18px]"), `${chip} pins the band height`);
  }
  // Read the class list, not the body: a comment there mentions py-px.
  const paramStart = PICKERS.indexOf("function ParamChip(");
  const param = PICKERS.slice(paramStart, PICKERS.indexOf("\n}", paramStart));
  const paramClasses = /className="([^"]*)"/.exec(param)?.[1] ?? "";
  assert.ok(paramClasses.length > 0, "ParamChip has a class list");
  assert.ok(!paramClasses.includes("py-px"), "no leftover vertical padding");
  assert.ok(
    paramClasses.includes("items-center"),
    "label centres in the fixed box",
  );
});

test("an over budget row dims instead of putting a pill on every line", () => {
  assert.ok(PICKERS.includes("group/row flex w-full flex-col items-stretch"));
  assert.ok(
    PICKERS.includes(
      '"opacity-0 transition-opacity group-hover/row:opacity-100 group-focus-visible/row:opacity-100"',
    ),
  );
  assert.ok(
    PICKERS.includes(
      '"opacity-60 transition-opacity group-hover/row:opacity-100 group-focus-visible/row:opacity-100"',
    ),
  );
  assert.ok(PICKERS.includes('vram: "min-w-min min-[560px]:w-[18px]"'));
});

test("one fit badge, so a colour or reveal change cannot miss a list", () => {
  assert.equal(
    PICKERS.split("const VRAM_VERDICT").length - 1,
    1,
    "one verdict table",
  );
  assert.ok(!PICKERS.includes("!text-red-700"), "no red fit badge left");
  assert.ok(!PICKERS.includes(">\n        OOM\n"), "no OOM text pill left");
  assert.match(
    PICKERS,
    /<VramBadge\n\s*status=\{\n\s*diffusionRefuses\(fit, diffusionLoad, hostPooledMemory\)/,
  );
  assert.equal(
    PICKERS.split("<VramBadge status={vramStatus} revealOnHover={!selected} />")
      .length - 1,
    2,
    "both model row slots reveal on hover",
  );
  assert.match(PICKERS, /exceeds &&\n\s*!selected &&/);
});

test("chat and the Hub answer the fit question with one formula", () => {
  // The loader admits at the saved budget or 0.97 over weights plus KV, not 0.7.
  assert.ok(
    CATALOG.includes("return classifyGgufFitForDevice(sizeBytes, budget);"),
    "the catalog classifier delegates",
  );
  assert.ok(
    RECOMMENDED.includes('from "../../../../lib/gguf-fit.ts"'),
    "and so does the Recommended fit filter",
  );
  assert.ok(
    !RECOMMENDED.includes("* 0.7"),
    "recommended-fit has no 0.7 budget left",
  );
  assert.ok(!PICKERS.includes("* 0.7"), "pickers has no 0.7 budget left");
  // Media GGUFs use the diffusion planner's lower budget; boundaries tested in model-catalog.check.ts.
  assert.ok(CATALOG.includes("export function classifyMediaGgufFit("));
  assert.match(
    PICKERS,
    /if \(diffusionLoad\) \{\n\s*return classifyMediaGgufFit\(/,
  );
  // Audio GGUFs run under llama.cpp or whisper, so only diffusion tasks use the media rule.
  assert.ok(PICKERS.includes("const DIFFUSION_TASKS: ReadonlySet<string>"));
  assert.match(PICKERS, /\.\.\.IMAGE_GEN_TASKS,\n\s*\.\.\.VIDEO_GEN_TASKS,/);
  assert.ok(
    !PICKERS.includes("mediaLoad: taskScoped,"),
    "no task-wide media rule",
  );
  assert.ok(RECOMMENDED.includes("mediaLoad: opts.diffusionLoad"));
  // Diffusion loads are torch, so rule and device source follow the runtime, not file format.
  assert.ok(
    RECOMMENDED.includes(
      "opts.diffusionLoad || !opts.isGguf ? opts.gpu : opts.inferenceGpu",
    ),
  );
  assert.ok(
    PICKERS.includes("diffusionLoad || !r.isGguf ? rowGpu : rowInferenceGpu"),
  );
  assert.ok(
    PICKERS.includes(
      "const expanderBudgetGpu = diffusionLoad ? gpu : inferenceGpu;",
    ),
  );
  assert.ok(
    !PICKERS.includes("inferenceGpu.systemRamAvailableGb"),
    "no expander takes the GGUF backend's RAM directly",
  );
  assert.ok(
    !PICKERS.includes("inferenceGpu.memoryTotalGb"),
    "no row takes the GGUF backend's capacity directly",
  );
  assert.ok(
    !PICKERS.includes("budgetKnown={inferenceGpu.budgetKnown}"),
    "and none takes its probe state either",
  );
  assert.equal(
    PICKERS.split("gpuGb={expanderGpuGb}").length - 1,
    9,
    "seven expanders and the two quant rows beside them",
  );
  assert.ok(
    HUB_CARD.includes(
      "const showFitInfo = !mediaRuntime && (Boolean(gpuGb) || Boolean(systemRamGb));",
    ),
  );
  assert.match(
    INSPECTOR,
    /showMemoryBar=\{!runsOnMediaRuntime\}\n\s*mediaPage=\{mediaPage\}/,
  );
  assert.match(PICKERS, /diffusionLoad\n\s*\? classifyMediaGgufFit\(/);
  assert.ok(PICKERS.includes("mediaLoad: diffusionLoad,"));
  assert.equal(
    PICKERS.split("diffusionLoad={diffusionLoad}").length - 1,
    7,
    "every task-scoped expander",
  );
  // On a host pool drop the RAM tier, or the APU window is counted with the RAM it maps.
  assert.ok(PICKERS.includes("function mediaRamBudgetGb("));
  assert.ok(
    PICKERS.includes("return hostPooled ? 0 : systemRamGb;"),
    "no RAM budget beside a host pool",
  );
  assert.ok(
    RECOMMENDED.includes("hostPooledMemory ? 0 : (systemRamGb ?? 0)"),
    "the gate drops it too",
  );
  assert.ok(PICKERS.includes("useVramBudgetFraction()"));
  assert.ok(PICKERS.includes("budgetFraction,"));
  assert.ok(HUB_PAGE.includes("useVramBudgetFraction()"));
  assert.ok(RECOMMENDED.includes("budgetFraction?: number;"));
  // Both surfaces must count cards, or badges diverge above the default budget on multi-GPU.
  assert.ok(HUB_PAGE.includes("gpuCount: source.deviceCount"));
  assert.ok(HUB_PAGE.includes("gpuCount: inferenceGpu.deviceCount"));
  assert.equal(
    HUB_PAGE.split("rowFitsDevice(row.result)").length - 1,
    2,
    "both Hub fit gates",
  );
  assert.ok(
    HUB_PAGE.includes("mediaRow || !result.isGguf ? gpu : inferenceGpu,"),
  );
  assert.ok(HUB_PAGE.includes("mediaLoad: mediaRow,"));
  assert.ok(
    HUB_PAGE.includes("studioPageForTask(result.pipelineTag) !== undefined"),
  );
  // The count narrows with the capacity so both describe the same inventory.
  assert.ok(RECOMMENDED.includes("deviceCount: 1,"), "scoped to one device");
  assert.ok(PICKERS.includes("gpuCount: rowInferenceGpu.deviceCount"));
  assert.ok(
    RECOMMENDED.includes("gpuCount: source.deviceCount ?? opts.gpuCount"),
  );
  assert.ok(
    !PICKERS.includes("gpuCount={inferenceGpu.deviceCount}"),
    "never the unscoped host count",
  );
  assert.equal(
    PICKERS.split("gpuCount={expanderGpuCount}").length - 1,
    7,
    "every task-scoped expander counts the scoped inventory",
  );
  assert.equal(
    PICKERS.split("gpuCount={gpu.deviceCount}").length - 1,
    1,
    "the exported GGUF list counts its own host",
  );
  assert.equal(
    PICKERS.split("<GgufVariantExpander").length - 1,
    8,
    "and that is every expander there is",
  );
  // Pass raw devices to gpuSharedHostMemoryGb; pre-folding undercounted multi-socket hosts.
  assert.ok(
    GPU_INFO.includes("gpuSharedHostMemoryGb(devices)"),
    "the RAM tier folds unified in, on the raw devices",
  );
  assert.ok(
    !GPU_INFO.includes("shared_memory: sharesHostMemory({"),
    "and never pre-folds them on the way in",
  );
  assert.ok(HUB_CARD.includes("gpuCount?: number;"));
  assert.ok(RECOMMENDED.includes("budgetFraction: opts.budgetFraction,"));
});

test("each fit verdict is an info mark that explains itself", () => {
  assert.ok(PICKERS.includes("icon={InformationCircleIcon}"));
  assert.ok(PICKERS.includes("marginal: MIGHT_FIT"));
  assert.ok(PICKERS.includes("partial: OFFLOADS"));
  // llama-server never refuses on size (--fit offloads), so all over-budget GGUFs say offloads.
  assert.ok(PICKERS.includes("oom: OFFLOADS"));
  // A torch pipeline has no --fit, so that one keeps a refusal.
  assert.ok(PICKERS.includes("exceeds: WONT_FIT"));
  assert.ok(
    PICKERS.includes(
      'hint: "Needs more memory than this device has. This model will not load."',
    ),
  );
  assert.ok(!PICKERS.includes("Larger than your VRAM and system RAM together"));
  // `tight` comes from checkVramFit's torch estimate that still fits the card.
  assert.ok(PICKERS.includes("tight: DEVICE_TIGHT"));
  assert.ok(
    PICKERS.includes(
      'hint: "Uses nearly all your VRAM, with little headroom for anything else."',
    ),
  );
  assert.ok(
    PICKERS.includes(
      'hint: "Model may not fit but still works with offloading. Expect slower inference."',
    ),
  );
  assert.ok(
    HUB_CARD.includes(
      '"Model may not fit but still works with offloading. Expect slower inference."',
    ),
  );
  // Marginal always takes --fit even on an idle card, so it must not promise a resident load.
  const mightFitHint =
    "Larger than your VRAM Budget allows, so part of it offloads even on an idle GPU. It is still smaller than the card, so raising the budget can keep it resident.";
  assert.ok(PICKERS.includes(mightFitHint));
  assert.ok(HUB_CARD.includes(mightFitHint));
  assert.ok(
    !PICKERS.includes("If other apps are using VRAM"),
    "no conditional offload copy",
  );
  assert.ok(!HUB_CARD.includes("If other apps are using VRAM"));
  assert.ok(!PICKERS.includes('label: "Might fit"'));
  assert.ok(!HUB_CARD.includes('label: "Might fit"'));
  assert.ok(!PICKERS.includes("Loading can fail while other apps"));
  assert.ok(!HUB_CARD.includes("Within the last GB of VRAM headroom"));
  assert.ok(!HUB_CARD.includes('label: "Won\'t fit"'));
  assert.equal(
    HUB_CARD.split(
      '"Model may not fit but still works with offloading. Expect slower inference."',
    ).length - 1,
    2,
    "partial and oom both",
  );

  assert.ok(PICKERS.includes('label: "Over budget"'));
  assert.ok(PICKERS.includes('label: "Does not fit"'));
  // Torch inference raises on any CPU/disk offload (raise_if_offloaded), so exceeds won't load.
  assert.ok(
    PICKERS.includes(
      'hint: "Needs more memory than this device has. This model will not load."',
    ),
  );
  assert.ok(PICKERS.includes("aria-label={verdict.label}"));
  assert.ok(PICKERS.includes("`Needs ~${vramEst}GB memory (GPU: ${gpuGb}GB)`"));
  assert.ok(
    !PICKERS.includes("GB VRAM (GPU:"),
    "no VRAM wording on an overage",
  );
  assert.ok(PICKERS.includes("GB VRAM (tight fit on"));
  assert.ok(
    PICKERS.includes(
      'return status === "partial" || status === "ram" || status === "oom" || status === "exceeds";',
    ) || /function isOverBudget[\s\S]{0,320}"exceeds"/.test(PICKERS),
    "dimming covers the orange verdicts only",
  );
});

test("a GGUF row takes the GGUF verdict, not the torch refusal", () => {
  assert.ok(
    PICKERS.includes("status: ggufRowFit(sizeBytes, rowInferenceGpu),"),
  );
  assert.ok(!PICKERS.includes("exceedsSize"));
  assert.match(
    PICKERS,
    /const ggufRowFit = \([\s\S]{0,220}\): GgufFitClass \| VramFitStatus \| null =>/,
  );
  // Every surviving producer of "exceeds" is a curated torch pipeline, which has no --fit.
  const producers = PICKERS.split("\n").filter(
    (line) => line.includes('"exceeds"') && line.includes("status:"),
  );
  assert.equal(producers.length, 2, "curated rows only");
  for (const line of producers) {
    assert.match(line, /curatedFit\.fits \? null : "exceeds"/);
  }
});

test("a diffusion model too big for a shared pool is refused, not offloaded", () => {
  // diffusion_memory.py refuses on unified memory, where offload frees nothing.
  assert.ok(PICKERS.includes("function diffusionRefuses("));
  assert.ok(
    PICKERS.includes('return fit === "oom" && diffusionLoad && hostPooled;'),
  );
  // hardware.py sets shared_memory only on Windows, so fold in unified_memory per device.
  assert.ok(!PICKERS.includes("gpu.sharedMemory"), "not the aggregate flag");
  assert.ok(
    GPU_INFO.includes("loadDeviceSharesHostMemory: sharesHostMemory({"),
    "the flag is the load device's, folded",
  );
  assert.match(
    PICKERS,
    /diffusionRefuses\(fit, diffusionLoad, hostPooledMemory\)\n\s*\? "exceeds"/,
  );
  assert.ok(
    PICKERS.includes(
      "diffusionRefuses(fit, diffusionLoad, gpu.loadDeviceSharesHostMemory)",
    ),
  );
  assert.ok(
    PICKERS.includes(
      'hint: "Needs more memory than this device has. This model will not load."',
    ),
  );
  assert.equal(
    PICKERS.split("hostPooledMemory={gpu.loadDeviceSharesHostMemory}").length -
      1,
    7,
    "every task-scoped expander learns the pool kind",
  );
});

test("the row tooltip reports the figure the verdict was reached with", () => {
  // The media rule scores raw size, so it keeps the raw number.
  assert.ok(PICKERS.includes("requiredGgufMemoryGb(sizeBytes)"));
  assert.match(
    PICKERS,
    /diffusionLoad\n\s*\? sizeBytes \/ 1024 \*\* 3\n\s*: requiredGgufMemoryGb\(sizeBytes\)/,
  );
});

test("aligned meta slots spend their slack on the name", () => {
  assert.ok(PICKERS.includes('"flex shrink-0 items-center gap-1 text-ui-10"'));
  assert.ok(PICKERS.includes('alignMeta === "device" ? "justify-start" : "justify-end"'));
});

test("On Device keeps at least 6px between the name, quant, modality and parameter marks", () => {
  assert.ok(PICKERS.includes('const DEVICE_META_GAP = "gap-[max(6px,6px)]";'));
  assert.ok(PICKERS.includes('alignMeta === "device" ? DEVICE_META_GAP : "gap-1"'));
  assert.ok(PICKERS.includes('alignMeta === "device" ? DEVICE_META_GAP : aligned ? "gap-1" : "gap-1.5"'));
});

test("every select-model surface shares that one badge", () => {
  const copies = [
    "../src/features/images/images-page.tsx",
    "../src/features/video/video-page.tsx",
    "../src/features/audio/audio-page.tsx",
  ];
  for (const path of copies) {
    const src = read(path);
    assert.ok(
      src.includes("@/features/model-picker/components/model-selector"),
      `${path} uses the shared selector`,
    );
    assert.ok(
      !src.includes("DownloadedBadge"),
      `${path} has no badge of its own`,
    );
  }
});

test("list header actions end where a hovered row's action does", () => {
  // Row actions sit inside a pill inset by unrailedRowPadding (9px, 8px under titlebar).
  assert.match(
    CSS,
    /\.sidebar-row-action \{\n\t\t@apply absolute top-0 bottom-0 right-0[^;]*pr-0\.75 /,
  );
  const label = CSS.slice(CSS.indexOf(".sidebar-sticky-label {"));
  // pl: unrailedRowPadding + a row's pl-3, so labels start where row content does.
  assert.match(label.slice(0, 500), /pl-\[18px\] pr-\[9px\] /);

  assert.ok(
    CSS.includes(
      ".sidebar-sticky-label.sidebar-sticky-label-desktop {\n\t\tpadding-left: 17px;\n\t\tpadding-right: 8px;",
    ),
  );

  assert.match(
    SIDEBAR,
    /const unrailedRowPadding = usesDesktopTitlebar \? "px-\[5px\]" : "px-1\.5";/,
  );
  assert.ok(
    SIDEBAR.includes(
      'const headerInset = usesDesktopTitlebar\n    ? "sidebar-sticky-label-desktop"\n    : null;',
    ),
  );
});

test("every list header takes the same alignment", () => {
  const shared = (
    SIDEBAR.match(
      /"sidebar-sticky-label sidebar-sticky-label-following group\/sidebar-header gap-1",\n\s*headerInset,/g,
    ) ?? []
  ).length;
  assert.equal(shared, 4, "Pinned, custom sections, Projects and Recents");
  assert.ok(!SIDEBAR.includes("translate-x-[2px]"));
});

test("capability glyph tags are the vision badge's pill, each in its own colour", () => {
  const body = (name: string) => {
    const start = PICKERS.indexOf(`function ${name}(`);
    return PICKERS.slice(start, PICKERS.indexOf("\n}", start));
  };
  for (const name of ["VisionBadge", "CapabilityIcons"]) {
    assert.match(body(name), /h-\[18px\] shrink-0 items-center justify-center rounded-md border border-border px-1\.5/);
  }
  assert.ok(!body("CapabilityIcons").includes("text-muted-foreground"), "no grey glyphs left");
  const list = PICKERS.slice(PICKERS.indexOf("const CAPABILITY_BADGES"), PICKERS.indexOf("const CapabilityScope"));
  const tones = [...list.matchAll(/tone: "([^"]+)"/g)].map((m) => m[1]);
  assert.equal(tones.length, 3, "every capability names a tone");
  assert.equal(new Set(tones).size, 3, "no two tags share a colour");
  assert.ok(tones.every((t) => !t.includes("indigo")), "none reuses the vision indigo");
});
