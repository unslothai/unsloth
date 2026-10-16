// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The uploader is picked in the Hub's filter popover, Unsloth's uploads until a search is typed.

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const page = readSrc("features/hub/hub-page.tsx");
const popover = readSrc("features/hub/catalog/model-search-filters.tsx");

test("the results header's Unsloth / All dropdown is gone", () => {
  assert.doesNotMatch(page, /OwnerScopeToggle|actions=\{ownerToggle\}/);
  assert.equal(existsSync(new URL("../src/features/hub/catalog/owner-scope-toggle.tsx", import.meta.url)), false);
});

test("Unsloth's uploads by default, every uploader once a search is typed", () => {
  assert.match(page, /uploaderPick \?\? \(debouncedQuery\.trim\(\) \? HUB_UPLOADERS : UNSLOTH_UPLOADER\)/);
  assert.match(page, /uploaders\.length === 1 && uploaders\[0\] === "unsloth" \? "unsloth" : "all"/);
});

test("the popover's uploader toggle picks Unsloth or every uploader", () => {
  assert.match(readSrc("features/hub/lib/model-search-filters.ts"), /"unsloth",\s*"organization",\s*"personal",/);
  assert.match(popover, /\["all", "All uploaders"\]/);
  assert.doesNotMatch(popover, /Organizations only|Non-organization/);
});

test("the range notes are hover tooltips beside Parameters and Context length", () => {
  assert.doesNotMatch(popover, /Ranges include both limits/);
  assert.match(popover, /aria-label=\{`About \$\{label\.toLowerCase\(\)\}`\}/);
  assert.match(popover, /icon=\{InformationCircleIcon\}/);
});

test("the Hub's menus take the composer + menu's shape", () => {
  const css = readSrc("features/hub/hub.css");
  assert.match(css, /\.hub-composer-menu\[data-slot\] \{\s*border-radius: 21px !important;/);
  assert.match(readSrc("features/hub/catalog/hub-option-menu.tsx"), /hub-composer-menu/);
  assert.match(popover, /hub-composer-menu/);
  assert.match(popover, /<div aria-hidden=\{true\} className="mx-3 h-px shrink-0 bg-border" \/>/);
});

test("Reasoning draws the composer's thinking bulb", () => {
  assert.match(popover, /<BulbIcon className="size-4 shrink-0" \/>/);
  const shared = readSrc("features/hub/catalog/shared.tsx");
  assert.match(shared, /capability\.key === "reasoning" \? \(\s*<BulbIcon className="size-3 shrink-0" \/>/);
  assert.doesNotMatch(shared + popover, /BrainIcon/);
});

test("a picked Unsloth during a search counts as a filter, and the row range resets the list", () => {
  const page = readSrc("features/hub/hub-page.tsx");
  assert.match(page, /\(isModelDiscover &&\s*uploaderPick !== null &&\s*ownerScope === "unsloth" &&\s*debouncedQuery\.trim\(\) !== ""\)/);
  assert.match(page, /const filterResetSignature = useMemo\([\s\S]{0,400}minRows,\s*maxRows,/);
});
