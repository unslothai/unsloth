// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shape assertions for the model-info panel's wiring (issue #11017). The pure logic is
// covered by picker-model-info-facts / -adapter; what is left is the wiring those cannot
// reach without a DOM, so it is asserted against the shipped source the way the other
// component-level checks in this suite are.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const DIALOG = readSrc(
  "features/model-picker/components/model-selector/model-info-dialog.tsx",
);
const ROW_MENU = readSrc(
  "features/model-picker/components/model-selector/model-row-menu.tsx",
);
const PICKERS = readSrc(
  "features/model-picker/components/model-selector/pickers.tsx",
);

test("the row menu offers Model info", () => {
  assert.match(ROW_MENU, /Model info/);
  assert.match(ROW_MENU, /ModelInfoDialog/);
});

// The menu renders nothing at all when it has no sections; forgetting `info` in that guard
// would hide the whole menu on a row whose only action is this one.
test("an info-only row still renders its menu", () => {
  // Conjuncts, not the whole line: #11196 added `!items?.length` to the same guard, and
  // pinning the exact spelling turns every later section into a false failure here.
  const guard = ROW_MENU.match(/if\s*\(([^)]*!info[^)]*)\)\s*\n?\s*return null;/)?.[1];
  assert.ok(guard, "row-menu empty guard not found");
  for (const section of ["!pin", "!update", "!del", "!cachePath", "!info"]) {
    assert.ok(guard.includes(section), `${section} missing from the empty guard`);
  }
});

// Mounting the dialog unconditionally would fetch metadata for every row with a menu, on
// every render of the list. It must stay behind the open flag.
test("the dialog mounts only once opened", () => {
  assert.match(ROW_MENU, /\{info\s*&&\s*infoOpen\s*&&\s*\(/);
});

// A local GGUF file has no Hub repo, so looking one up would 404 against whatever the path's
// basename happens to collide with.
test("local-path rows do not offer Hub info", () => {
  assert.match(
    PICKERS,
    /info=\{\s*isLocalPath\s*\? undefined\s*: \{\s*repoId,\s*variant: v\.quant,\s*hasLocalGguf: v\.downloaded,?\s*\}\s*\}/,
  );
});

// The local header probe reads one file, so a row naming a quant has to say which.
test("rows that name a quant pass it to the info panel", () => {
  const infos = PICKERS.match(/\n\s*info=\{[^}]*\}[^}]*\}/g) ?? [];
  const withVariant = infos.filter((i) => i.includes("variant:"));
  assert.ok(
    withVariant.length >= 3,
    `expected the quant-bearing rows to pass a variant, saw ${withVariant.length}`,
  );
});

// Every menu over a Hub-backed row offers info. The exception is real, not a gap: a
// Connected/provider row (#11196) has no Hub repo to look up, so it supplies its own
// "Model info" entry through `items` against connected-model-info-dialog instead. Counting
// those as missing would demand a repoId that does not exist.
test("every Hub-backed row menu in the picker passes a repo to look up", () => {
  const menus = PICKERS.split(/<ModelRowMenu\b/).slice(1);
  const infos = PICKERS.match(/\n\s*info=\{/g) ?? [];
  assert.ok(menus.length > 0, "no ModelRowMenu call sites found");
  const withoutInfo = menus.filter((m) => !/\n\s*info=\{/.test(m.split("/>")[0]));
  for (const m of withoutInfo) {
    assert.match(
      m.split("/>")[0],
      /label: "Model info"/,
      "a row menu offers neither a Hub info prop nor its own Model info entry",
    );
  }
  assert.equal(infos.length, menus.length - withoutInfo.length);
});

// The fetch is gated on `open` so a closed dialog costs nothing, and on `online` so an
// offline app does not queue a request that cannot succeed.
test("metadata is fetched only while open and online", () => {
  assert.match(DIALOG, /useSelectedModelMetadata\(\s*open \? repoId : null/);
  assert.match(DIALOG, /enabled: open/);
  assert.match(DIALOG, /online,/);
});

// Three states the panel must tell apart: no network, a repo that could not be read, and a
// fetch still in flight. Collapsing any pair reports the wrong cause. Asserted on the
// distinct copy rather than the branch spelling, which the formatter is free to rewrite.
test("offline, error and loading are distinguished", () => {
  assert.match(DIALOG, /Connect to the internet/);
  assert.match(DIALOG, /Could not reach Hugging Face/);
  assert.match(DIALOG, /Loading details/);
  assert.match(DIALOG, /Spinner/);
  // All three branch on the same two flags, so each must still be consulted.
  assert.match(DIALOG, /\bonline\b/);
  assert.match(DIALOG, /\berror\b/);
  assert.match(DIALOG, /\bresult\b/);
});

// The app routes outbound links through a confirmation gate; a bare target="_blank" here
// would be the one link that escapes it.
test("the Hugging Face link goes through the external-link gate", () => {
  assert.match(DIALOG, /confirmExternalLink/);
  assert.match(DIALOG, /rel="noopener noreferrer"/);
});

// The licence chip is the panel's headline answer, so each verdict needs its own tone --
// and `restricted` must not borrow the `open` one.
test("every licence verdict has a distinct tone", () => {
  for (const verdict of ["open", "restricted", "proprietary", "unknown"]) {
    assert.ok(new RegExp(`${verdict}:`).test(DIALOG), `no tone for ${verdict}`);
  }
  assert.match(DIALOG, /restricted:[\s\S]{0,80}amber/);
  assert.match(DIALOG, /open:[\s\S]{0,80}emerald/);
});

// Rendering decisions belong to modelInfoFacts so they stay testable; a formatter creeping
// into the dialog would be logic no test covers.
test("the dialog renders facts rather than deriving them", () => {
  assert.match(DIALOG, /modelInfoFacts/);
  assert.match(DIALOG, /metaFromHfResult/);
  assert.ok(
    !/toLocaleDateString|formatBytes\(/.test(DIALOG),
    "formatting logic leaked into the dialog",
  );
});

// The licence verdict is most useful *before* a download commits several gigabytes, so the
// menu can no longer sit entirely behind the downloaded/partial guard. An undownloaded
// variant still gets a menu carrying info alone (issue #11033 review).
test("an undownloaded variant still offers Model info", () => {
  assert.match(
    PICKERS,
    /v\.downloaded \|\| v\.partial === true\s*\n?\s*\?[\s\S]{0,200}?:\s*!isLocalPath\)\s*&&\s*\(/,
    "the variant row menu should fall back to an info-only menu when not downloaded",
  );
});

// The flip side: a variant that is not on disk has nothing to reveal in a file manager and
// nothing to delete, so those actions must not ride along with the info-only menu.
test("disk-only actions stay behind the downloaded guard", () => {
  assert.match(
    PICKERS,
    /isLocalPath \|\| !\(v\.downloaded \|\| v\.partial === true\)\s*\n?\s*\?\s*undefined/,
    "cachePath should be withheld for a variant that is not on disk",
  );
  assert.match(
    PICKERS,
    /onDeleteVariant && \(v\.downloaded \|\| v\.partial === true\)/,
    "delete should be withheld for a variant that is not on disk",
  );
});

// The template is already in hand from the header read, so the viewer must not refetch it.
test("the chat template row opens a read-only viewer", () => {
  assert.match(DIALOG, /ChatTemplateEditorDialog/);
  assert.match(
    DIALOG,
    /readOnly=\{true\}/,
    "the panel does not configure the model",
  );
  assert.match(
    DIALOG,
    /fact\.key === "chatTemplate" && template/,
    "only an embedded template is clickable",
  );
  assert.match(
    DIALOG,
    /defaultTemplate=\{template\}/,
    "viewer reads the probed template",
  );
});

// Popularity and the restated licence verdict were both dropped from the panel.
test("the panel does not restate the licence or show popularity", () => {
  assert.doesNotMatch(DIALOG, /"Downloads"/);
  assert.doesNotMatch(DIALOG, /"Likes"/);
  // The verdict line survives only for screen readers.
  assert.match(DIALOG, /DialogDescription className="sr-only"/);
});

// A non-GGUF Hub result never expands, so without a menu of its own there was no way to read
// its licence before selecting it and starting a download.
test("Hub result rows carry an info-only menu, whatever their format", () => {
  const rows = PICKERS.match(
    /<div className="group flex items-center">\s*<div className="min-w-0 flex-1">\s*<ModelRow/g,
  );
  assert.ok(
    rows && rows.length >= 3,
    `expected the Hub result rows wrapped, saw ${rows?.length ?? 0}`,
  );
  assert.match(
    PICKERS,
    /ariaLabel=\{`More options for \$\{id\}`\}\s*\n\s*info=\{\{\s*repoId: id,/,
    "the parent row's menu should carry info and nothing format-gated",
  );
});

test("local probes use the GGUF inventory, not other cached formats", () => {
  assert.match(ROW_MENU, /hasLocalGguf=\{info\.hasLocalGguf\}/);
  assert.match(PICKERS, /cachedGguf\.filter\(\(c\) => !c\.partial\)/);
  const parentProbes = PICKERS.match(
    /hasLocalGguf:\s*downloadedGgufSet\.has\(\s*id\.toLowerCase\(\),?\s*\)/g,
  );
  assert.equal(parentProbes?.length, 3);
  assert.match(
    PICKERS,
    /variant: variant\.quant,\s*hasLocalGguf: isDownloaded/,
  );
});
