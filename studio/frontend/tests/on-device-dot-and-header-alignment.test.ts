// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
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
  // A download arrow read as "click to fetch this" on the one row that needs
  // no fetching. The Hub already had the right answer.
  assert.ok(!badge.includes("Download01Icon"), "no download glyph");
  assert.match(badge, /size-\[5px\] rounded-full bg-status-success/);
  assert.match(badge, /aria-label="On device"/);
  assert.ok(
    MODELS_TABLE.includes("bg-status-success"),
    "and the Hub still uses that dot, so the two agree",
  );
});

test("the download glyph is gone from the picker entirely", () => {
  // Left behind, it would still be imported for nothing.
  assert.ok(!PICKERS.includes("Download01Icon"));
});

test("the scoped badge column reserves the wider on-device marker", () => {
  // Video can show one 18px capability, a 4px gap and the 14px marker. If the
  // fixed width remains 34px, min-w-min expands only those rows and shifts all
  // metadata columns after the badge slot.
  assert.ok(PICKERS.includes('badgeMid: "min-w-min min-[560px]:w-[36px]"'));
});

test("the unscoped badge column is sized per list, not to the union of both", () => {
  // Sized to the widest set each list can draw, not to both lists at once. On Device is the vision
  // badge (26px), a gap-1 (4px) and the partial mark (14px): a GGUF repo can show vision and be
  // half-downloaded at once, and at 26px that row alone grew and carried its quant chip 18px left
  // of every other row -- the exact drift the fixed columns exist to stop.
  assert.ok(PICKERS.includes('badgeDevice: "min-w-min min-[560px]:w-[44px]"'));
  assert.ok(PICKERS.includes('badgeWide: "min-w-min min-[560px]:w-[36px]"'));
  // Both marks really can land on one On Device row, which is what makes 44 the right number.
  const gguf = PICKERS.slice(PICKERS.indexOf("const renderDownloadedGgufRow"));
  const row = gguf.slice(0, gguf.indexOf("\n  };"));
  assert.ok(row.includes('alignMeta="device"'));
  assert.ok(row.includes("showVision={c.has_vision"));
  assert.ok(row.includes("partial={isPartialRepo}"));
  // And they sit in that one slot rather than overlapping.
  assert.match(
    PICKERS,
    /\{showVision && <VisionBadge \/>\}\n\s*\{partial \? <PartialBadge resumable=\{partialResumable\} \/> : null\}/,
  );
  assert.match(
    PICKERS,
    /alignMeta === "device"\n\s*\? META_COLUMN\.badgeDevice\n\s*: META_COLUMN\.badgeWide/,
  );
});

test("a row's leading dot starts where its section label does", () => {
  // Section labels sit at px-2.5. The dot is centred in a 14px hover target, so its slot starts
  // at 10 - (14 - 5) / 2 = 5.5px for the dot to land on 10px. px-2 put it at 12.5px.
  assert.match(PICKERS, /py-1\.5 pl-\[5\.5px\] pr-2 text-left text-sm/);
  const label = PICKERS.slice(
    PICKERS.indexOf("flex items-center justify-between gap-1 px-2.5"),
  );
  assert.ok(label.startsWith("flex items-center justify-between gap-1 px-2.5"));
  // The 14px hover target is what makes 5.5 the right number; shrinking it would move the dot.
  assert.ok(
    PICKERS.includes(
      'className="flex size-[14px] shrink-0 items-center justify-center"',
    ),
  );
});

test("the parameter and size columns are sized to the ink they hold", () => {
  // The widest size formatBytes writes ("128GB", no space) is 29.5px, not the ~40px a spaced
  // "536 MB" would need.
  assert.ok(PICKERS.includes('size: "min-w-min min-[560px]:w-[3.2em]"'));
  // The no-space format is what makes 3.2em enough; a spaced size would need ~4.2em again.
  assert.match(
    PICKERS,
    /No space: "145MB" reads as one value beside the quant chip\./,
  );
});

test("the parameter column is fixed, so the quant column cannot drift row to row", () => {
  // It is the last variable width to the right of the name group, and the name group is flex-1:
  // hugging the chip handed a "1B" row -- and more so a row with no param at all -- the leftover,
  // which carried that row's quant chip further right. 4.4em holds the widest label these lists
  // draw at text-ui-10 ("235B" is 38.4px, a 5-char "0.35B" 40.9px), so nothing routine trips
  // min-w-min and shifts the row back out of line.
  assert.ok(PICKERS.includes('param: "min-w-min min-[560px]:w-[4.4em]"'));
  // Hub keeps its own column: its labels run to "2779.5B".
  assert.ok(PICKERS.includes('paramWide: "min-w-min min-[560px]:w-[5.2em]"'));
});

test("the parameter chip leads its column, so the modality gap is the cluster's own", () => {
  // Trailing the chip spends the column's leftover in FRONT of it, where it reads as part of the
  // gap to the modality mark and grows with the label: 6.9px after a "217B", 19px after a "1B".
  // Leading it leaves that gap as gap-1 -- the same 4px the quant chip keeps to the same mark.
  assert.match(
    PICKERS,
    /alignMeta === "hub"\n\s*\? cn\("justify-end", META_COLUMN\.paramWide\)\n\s*: cn\("justify-start", META_COLUMN\.param\)/,
  );
});

test("the quant chip is flush right in its slot, so the chips read as one column", () => {
  // The slot is sized for the longest quant, so left-aligning ended a "Q8_0" and a "UD-Q4_K_XL"
  // at different x even once the slot itself stopped moving.
  assert.ok(PICKERS.includes('quant: "min-[560px]:w-[7.2em]"'));
  assert.match(PICKERS, /"flex shrink-0 items-center justify-end text-ui-9"/);
});

test("the quant chip rides in the meta cluster, not on the end of the name", () => {
  // The name group is items-baseline and sized by the name's own line box; a chip centred against
  // THAT agrees with the rest of the row only while the two boxes happen to share a centre. In the
  // meta cluster one items-center rule lines the chip up with the vision mark, the parameter chip,
  // the size and the row's buttons, so the agreement is structural.
  const meta = PICKERS.slice(
    PICKERS.indexOf('"ml-auto flex shrink-0 items-center"'),
  );
  const quantSlot = meta.indexOf("META_COLUMN.quant");
  const badgeSlot = meta.indexOf("badgeColumn");
  assert.ok(quantSlot > 0, "the quant slot sits inside the meta cluster");
  assert.ok(quantSlot < badgeSlot, "and leads the badge column");
  // self-center was what compensated for the baseline box; in an items-center row it is noise.
  assert.ok(
    !PICKERS.includes("justify-end self-center text-ui-9"),
    "no leftover baseline compensation",
  );
});

test("every chip in the row band pins the same height", () => {
  // ParamChip sized itself from its line box, the one height here that scales with
  // --ui-font-scale: at 1.0 it stood 1px prouder than the quant and vision chips beside it and at
  // 0.8125 it sat 1.8px shorter, so the row was only level at the scale where the two crossed.
  for (const chip of ["QuantChip", "VisionBadge", "ParamChip"]) {
    const start = PICKERS.indexOf(`function ${chip}(`);
    assert.ok(start > 0, `${chip} exists`);
    const body = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
    assert.ok(body.includes("h-[18px]"), `${chip} pins the band height`);
  }
  // The height is what centres the label now, so the padding that used to set it is gone. Read
  // the class list, not the body: the comment above it names py-px as the thing it replaced.
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
  // Recommended is mostly over budget on a normal GPU, so a pill per row was a wall of colour.
  // The row dims and the pill is painted only while that row is hovered or focused.
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
  // The mark is hidden, not removed, and its slot keeps a width, so revealing it cannot reflow.
  assert.ok(PICKERS.includes('vram: "min-w-min min-[560px]:w-[18px]"'));
});

test("one fit badge, so a colour or reveal change cannot miss a list", () => {
  // The quantization rows had their own copy and stayed red when the pill went orange.
  assert.equal(
    PICKERS.split("const VRAM_VERDICT").length - 1,
    1,
    "one verdict table",
  );
  assert.ok(!PICKERS.includes("!text-red-700"), "no red fit badge left");
  assert.ok(!PICKERS.includes(">\n        OOM\n"), "no OOM text pill left");
  // Variant rows hand the classifier's verdict straight to the badge, with no second mapping
  // between them to drift.
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
  // A selected row is exempt from dimming, so hiding its mark too left it with no verdict at all
  // until hovered, which is nothing at rest and nothing on touch.
  assert.match(PICKERS, /exceeds &&\n\s*!selected &&/);
});

test("chat and the Hub answer the fit question with one formula", () => {
  // The picker carried its own rule: 0.7 of GPU plus 0.7 of RAM against the raw file size, with a
  // comment claiming it matched _select_gpus. The loader admits against the saved VRAM Budget or
  // 0.97, over weights PLUS estimated KV, so 0.7 matched neither the loader nor the Hub badge.
  assert.ok(
    CATALOG.includes("return classifyGgufFitForDevice(sizeBytes, budget);"),
    "the catalog classifier delegates",
  );
  assert.ok(
    RECOMMENDED.includes('from "../../../../lib/gguf-fit.ts"'),
    "and so does the Recommended fit filter",
  );
  // No copy of the old budget survives in either GGUF path.
  assert.ok(
    !RECOMMENDED.includes("* 0.7"),
    "recommended-fit has no 0.7 budget left",
  );
  assert.ok(!PICKERS.includes("* 0.7"), "pickers has no 0.7 budget left");
  // model-catalog keeps one, and only for the media pickers: a GGUF offered on Images or Video is
  // placed by the diffusion backend, whose budget is far below llama.cpp's. On a 64 GiB Mac that
  // planner allows about 43.5 GiB, this rule 44.8, the llama.cpp rule 62.1. The boundaries are
  // asserted for real in model-catalog.check.ts.
  assert.ok(CATALOG.includes("export function classifyMediaGgufFit("));
  assert.match(
    PICKERS,
    /if \(diffusionLoad\) \{\n\s*return classifyMediaGgufFit\(/,
  );
  // Diffusion tasks only. Audio is task-scoped too but runs its GGUFs under llama.cpp (TTS) and
  // the whisper sidecars (ASR), so scoring it at 70% hid runnable models and shrank the quant.
  assert.ok(PICKERS.includes("const DIFFUSION_TASKS: ReadonlySet<string>"));
  assert.match(PICKERS, /\.\.\.IMAGE_GEN_TASKS,\n\s*\.\.\.VIDEO_GEN_TASKS,/);
  assert.ok(
    !PICKERS.includes("mediaLoad: taskScoped,"),
    "no task-wide media rule",
  );
  assert.ok(RECOMMENDED.includes("mediaLoad: opts.diffusionLoad"));
  // The RULE and the DEVICE SOURCE both follow the runtime that places the row. A diffusion load
  // is torch whatever the file format is, and on a Vulkan chat build inferenceGpu is a different
  // device set: it can see a card torch cannot, so a media GGUF was badged and recommended
  // against capacity the diffusion loader never gets.
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
  // And every expander reads that one budget, rather than reaching for inferenceGpu itself.
  assert.ok(
    !PICKERS.includes("inferenceGpu.systemRamAvailableGb"),
    "no expander takes the GGUF backend's RAM directly",
  );
  // The custom-folder expander was the one that kept reaching for the raw aggregate: on a media
  // page it classified against memory the diffusion loader never sees, and it was not even
  // load-scoped. Every row-level budget in this file now comes from the one chosen source.
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
  // The Hub card answers llama.cpp's question, so it must not answer it about a diffusion repo:
  // an oom there would read "still works with offloading" for a load the planner refuses.
  assert.ok(
    HUB_CARD.includes(
      "const showFitInfo = !mediaRuntime && (Boolean(gpuGb) || Boolean(systemRamGb));",
    ),
  );
  assert.match(
    INSPECTOR,
    /showMemoryBar=\{!runsOnMediaRuntime\}\n\s*mediaRuntime=\{runsOnMediaRuntime\}/,
  );
  // Parent rows and the fit gate take the same rule as the quant rows under them, or a media row
  // reads as fitting while everything inside it reads as oom.
  assert.match(PICKERS, /diffusionLoad\n\s*\? classifyMediaGgufFit\(/);
  // Both gates read the flag the same way, or one hides a row the other keeps.
  assert.ok(PICKERS.includes("mediaLoad: diffusionLoad,"));
  // And every media call site sets the flag, or the guard silently does nothing.
  assert.equal(
    PICKERS.split("diffusionLoad={diffusionLoad}").length - 1,
    7,
    "every task-scoped expander",
  );
  // The RAM tier is dropped on a host pool, or the media rule adds the APU's window to the very
  // RAM it is a window INTO and returns partial for a load the planner refuses.
  assert.ok(PICKERS.includes("function mediaRamBudgetGb("));
  assert.ok(
    PICKERS.includes("return hostPooled ? 0 : systemRamGb;"),
    "no RAM budget beside a host pool",
  );
  assert.ok(
    RECOMMENDED.includes("hostPooledMemory ? 0 : (systemRamGb ?? 0)"),
    "the gate drops it too",
  );
  // The saved VRAM Budget now reaches chat. Moving the slider used to change the Hub's verdicts
  // and leave the picker's untouched.
  assert.ok(PICKERS.includes("useVramBudgetFraction()"));
  assert.ok(PICKERS.includes("budgetFraction,"));
  // Threaded through the row filters as well. Passing it to the quant rows alone left the parent
  // rows and the "Fits on device" gate scoring against the 0.97 default.
  assert.ok(HUB_PAGE.includes("useVramBudgetFraction()"));
  assert.ok(RECOMMENDED.includes("budgetFraction?: number;"));
  // Both surfaces count the cards, or the Hub badge and the picker badge diverge again the moment
  // the VRAM Budget goes above the default on a multi-GPU host. The two row gates take it from
  // the SCOPED source, which is 1 for a media row and the host count otherwise.
  assert.ok(HUB_PAGE.includes("gpuCount: source.deviceCount"));
  assert.ok(HUB_PAGE.includes("gpuCount: inferenceGpu.deviceCount"));
  // And both gates ask one helper, which picks the budget by the runtime that places the row: an
  // image or video repo goes to the diffusion planner on one torch device, under the media rule.
  // Judged by llama.cpp the Hub kept a 52 GiB media GGUF that clears 62.1 GiB on a 64 GiB Mac
  // and blows the planner's 44.8.
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
  // The count is narrowed WITH the capacity, so it can never describe a different inventory than
  // the gpuGb beside it. A task page puts the load on one device; charging the per-card reserve
  // once per host GPU against that one card scored an audio quant at 23.28 GiB where the loader
  // offers the selected card's 23.5.
  assert.ok(RECOMMENDED.includes("deviceCount: 1,"), "scoped to one device");
  assert.ok(PICKERS.includes("gpuCount: rowInferenceGpu.deviceCount"));
  assert.ok(
    RECOMMENDED.includes("gpuCount: source.deviceCount ?? opts.gpuCount"),
  );
  assert.ok(
    !PICKERS.includes("gpuCount={inferenceGpu.deviceCount}"),
    "never the unscoped host count",
  );
  // Counted against the expanders themselves, not a fixed 7: the fine-tuned / exported GGUF list
  // is the eighth and was missed, so its host read as one card and over-budgeted at a 1.0 setting.
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
  // The APU window comes out of the RAM tier where the figure is built, so every rule downstream
  // sees one pool counted once.
  assert.ok(
    GPU_INFO.includes("shared_memory: sharesHostMemory({"),
    "the RAM tier folds unified in",
  );
  assert.ok(HUB_CARD.includes("gpuCount?: number;"));
  assert.ok(RECOMMENDED.includes("budgetFraction: opts.budgetFraction,"));
});

test("each fit verdict is an info mark that explains itself", () => {
  // A pill shouted a three letter acronym; the mark says what it means on hover.
  assert.ok(PICKERS.includes("icon={InformationCircleIcon}"));
  // Keyed by the Hub's five classes, which is how the picker gained the tier it was missing:
  // over budget but still card-sized needs no system RAM, so it fires on unified memory too.
  assert.ok(PICKERS.includes("marginal: MIGHT_FIT"));
  assert.ok(PICKERS.includes("partial: OFFLOADS"));
  // Every over-budget GGUF says the same thing, however far over. llama-server never refuses one
  // on size: _select_gpus returns use_fit and --fit offloads the rest. Splitting the copy here hid
  // the honest answer on unified memory, where the RAM tier cannot fire and everything reads oom.
  assert.ok(PICKERS.includes("oom: OFFLOADS"));
  // A torch pipeline has no --fit, so that one keeps a refusal.
  assert.ok(PICKERS.includes("exceeds: WONT_FIT"));
  assert.ok(
    PICKERS.includes(
      'hint: "Needs more memory than this device has. This model will not load."',
    ),
  );
  assert.ok(!PICKERS.includes("Larger than your VRAM and system RAM together"));
  // The training estimator's three words map onto those rather than carrying their own copy.
  // `tight` reaches the badge only from checkVramFit's 75-100% band, a torch estimate that still
  // fits on the card. Aliasing it to the GGUF copy told those rows they spill into system RAM.
  assert.ok(PICKERS.includes("tight: DEVICE_TIGHT"));
  assert.ok(
    PICKERS.includes(
      'hint: "Uses nearly all your VRAM, with little headroom for anything else."',
    ),
  );
  // Only oom fails to load, so only its hint says so; partial offloads and runs slower.
  assert.ok(
    PICKERS.includes(
      'hint: "Model may not fit but still works with offloading. Expect slower inference."',
    ),
  );
  // The Hub says it in the same words, which is the point of sharing the classifier.
  assert.ok(
    HUB_CARD.includes(
      '"Model may not fit but still works with offloading. Expect slower inference."',
    ),
  );
  // Neither surface makes a marginal offload conditional on other apps. _vram_usable_mib gives
  // free - reserve, which on a completely idle card IS the budget this tier has already passed,
  // so _select_gpus takes --fit every time; other apps only shrink free further. Saying it
  // "fits with almost no room to spare" promised a resident load the loader never admits.
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
  // The Hub's oom row says it too, so the two surfaces read alike on a host where every
  // over-budget quant lands in that class.
  assert.ok(!HUB_CARD.includes('label: "Won\'t fit"'));
  assert.equal(
    HUB_CARD.split(
      '"Model may not fit but still works with offloading. Expect slower inference."',
    ).length - 1,
    2,
    "partial and oom both",
  );

  // Marks are reachable by screen readers without the tooltip.
  assert.ok(PICKERS.includes('label: "Over budget"'));
  assert.ok(PICKERS.includes('label: "Does not fit"'));
  // "will not load" is right for `exceeds`, which only ever comes from a torch row: inference.py
  // calls raise_if_offloaded after loading, and that raises ValueError on any CPU or disk offload
  // ("Inference does not support models loaded with CPU or disk offload"). A GGUF over budget is
  // handed to --fit instead, which is why it says the opposite.
  assert.ok(
    PICKERS.includes(
      'hint: "Needs more memory than this device has. This model will not load."',
    ),
  );
  assert.ok(PICKERS.includes("aria-label={verdict.label}"));
  // An over-budget figure is a TOTAL: `partial` splits across VRAM and RAM, and the number is
  // weights plus activations plus KV, so "Needs ~47GB VRAM" argued with the offload verdict.
  assert.ok(PICKERS.includes("`Needs ~${vramEst}GB memory (GPU: ${gpuGb}GB)`"));
  assert.ok(
    !PICKERS.includes("GB VRAM (GPU:"),
    "no VRAM wording on an overage",
  );
  // A load that stays on the card still says VRAM, which is what it means there.
  assert.ok(PICKERS.includes("GB VRAM (tight fit on"));
  // A marginal row is a full GPU load, so it is not dimmed with the over budget ones.
  assert.ok(
    PICKERS.includes(
      'return status === "partial" || status === "ram" || status === "oom" || status === "exceeds";',
    ) || /function isOverBudget[\s\S]{0,320}"exceeds"/.test(PICKERS),
    "dimming covers the orange verdicts only",
  );
});

test("a GGUF row takes the GGUF verdict, not the torch refusal", () => {
  // This branch used to set "exceeds", the one verdict that says a model will not load. A GGUF is
  // offloaded by llama-server rather than refused, so the row said the opposite of what happens,
  // and it is the verdict shown on the Recommended list where most rows are over budget.
  assert.ok(
    PICKERS.includes("status: ggufRowFit(sizeBytes, rowInferenceGpu),"),
  );
  // A boolean here collapsed marginal and partial into "no badge", so a repo whose smallest quant
  // already needed offload rendered as a clean fit beside variant rows saying otherwise.
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
    assert.match(line, /curatedFits \? null : "exceeds"/);
  }
});

test("a diffusion model too big for a shared pool is refused, not offloaded", () => {
  // diffusion_memory.py refuses up front on unified memory: offload "moves bytes within that pool
  // and frees nothing", and without the refusal the load "allocates past physical memory" with
  // "the failure is the OS killing the process with no Python exception". Saying it still works
  // with offloading is the worst thing this badge could say there.
  assert.ok(PICKERS.includes("function diffusionRefuses("));
  assert.ok(
    PICKERS.includes('return fit === "oom" && diffusionLoad && hostPooled;'),
  );
  // The LOAD DEVICE's pool, folding unified_memory in. hardware.py sets shared_memory only on
  // Windows, so a Linux ROCm APU arrives unified true / shared false while diffusion_memory.py
  // still calls it unified_memory and refuses. The aggregate shared flag missed that host.
  assert.ok(!PICKERS.includes("gpu.sharedMemory"), "not the aggregate flag");
  assert.ok(
    GPU_INFO.includes("loadDeviceSharesHostMemory: sharesHostMemory({"),
    "the flag is the load device's, folded",
  );
  // Both the quant rows and the parent row take it.
  assert.match(
    PICKERS,
    /diffusionRefuses\(fit, diffusionLoad, hostPooledMemory\)\n\s*\? "exceeds"/,
  );
  assert.ok(
    PICKERS.includes(
      "diffusionRefuses(fit, diffusionLoad, gpu.loadDeviceSharesHostMemory)",
    ),
  );
  // On a shared pool "VRAM" would name a number the user does not have.
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
  // classifyGgufFit scores weights PLUS activations and KV, so a 20 GiB quant needing 24 GiB read
  // "tight fit" beside a tooltip saying "~20GB VRAM". The media rule scores raw size, so it keeps
  // the raw number.
  assert.ok(PICKERS.includes("requiredGgufMemoryGb(sizeBytes)"));
  assert.match(
    PICKERS,
    /diffusionLoad\n\s*\? sizeBytes \/ 1024 \*\* 3\n\s*: requiredGgufMemoryGb\(sizeBytes\)/,
  );
});

test("aligned meta slots spend their slack on the name", () => {
  // Centring a lone glyph splits the slack either side of it, reading as a gap on both sides.
  assert.ok(
    PICKERS.includes(
      '"flex shrink-0 items-center justify-end gap-1 text-ui-10"',
    ),
  );
});

test("every select-model surface shares that one badge", () => {
  // Images, Video and Audio render the same ModelSelector, so there is no
  // second copy of the badge to keep in step.
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
  // A row action is `right-0 pr-1.5` inside a pill the list inset by
  // unrailedRowPadding: 12px in normally, 11px under the desktop titlebar.
  assert.match(
    CSS,
    /\.sidebar-row-action \{\n\t\t@apply absolute top-0 bottom-0 right-0[^;]*pr-1\.5/,
  );
  const label = CSS.slice(CSS.indexOf(".sidebar-sticky-label {"));
  assert.match(label.slice(0, 400), /pl-\[16px\] pr-3 /);

  assert.ok(
    CSS.includes(
      ".sidebar-sticky-label.sidebar-sticky-label-desktop {\n\t\tpadding-right: 11px;",
    ),
  );
  assert.ok(
    CSS.includes(
      ".sidebar-sticky-label.sidebar-sticky-label-desktop-recents {\n\t\tpadding-right: 13px;",
    ),
  );

  assert.match(
    SIDEBAR,
    /const unrailedRowPadding = usesDesktopTitlebar \? "px-\[5px\]" : "px-1\.5";/,
  );
  assert.ok(
    SIDEBAR.includes(
      'const headerRightPadding = usesDesktopTitlebar\n    ? "sidebar-sticky-label-desktop"\n    : null;',
    ),
  );
  // Recents is nudged 2px right there and carries its padding with it.
  assert.ok(
    SIDEBAR.includes(
      'const recentsHeaderRightPadding = usesDesktopTitlebar\n    ? "sidebar-sticky-label-desktop-recents"\n    : null;',
    ),
  );
});

test("all three list headers take the same alignment", () => {
  // Pinned and Projects share one class string; Recents has its own because of
  // the translate. Two of the first, one of the second.
  const shared =
    SIDEBAR.split(
      '"sidebar-sticky-label sidebar-sticky-label-following group/sidebar-header gap-1", headerRightPadding,',
    ).length - 1;
  assert.equal(shared, 2, "Pinned and Projects");
  assert.ok(
    SIDEBAR.includes("recentsHeaderRightPadding,"),
    "and Recents applies its own",
  );
});
