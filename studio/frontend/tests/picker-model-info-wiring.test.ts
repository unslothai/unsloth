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
  assert.match(
    ROW_MENU,
    /if\s*\(!pin\s*&&\s*!update\s*&&\s*!del\s*&&\s*!cachePath\s*&&\s*!info\)\s*return null;/,
  );
});

// Mounting the dialog unconditionally would fetch metadata for every row with a menu, on
// every render of the list. It must stay behind the open flag.
test("the dialog mounts only once opened", () => {
  assert.match(ROW_MENU, /\{info\s*&&\s*infoOpen\s*&&\s*\(/);
});

// A local GGUF file has no Hub repo, so looking one up would 404 against whatever the path's
// basename happens to collide with.
test("local-path rows do not offer Hub info", () => {
  assert.match(PICKERS, /info=\{isLocalPath \? undefined : \{ repoId \}\}/);
});

test("every row menu in the picker passes a repo to look up", () => {
  const menus = PICKERS.match(/<ModelRowMenu\b/g) ?? [];
  const infos = PICKERS.match(/\n\s*info=\{/g) ?? [];
  assert.ok(menus.length > 0, "no ModelRowMenu call sites found");
  assert.equal(
    infos.length,
    menus.length,
    `${menus.length} row menus but ${infos.length} info props`,
  );
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
