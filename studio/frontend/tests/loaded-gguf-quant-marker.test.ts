// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const picker = readSrc("features/model-picker/components/model-selector/pickers.tsx");

test("every quant list and cached GGUF row marks the quant each loaded model runs", () => {
  const expanders = picker.match(/<GgufVariantExpander\b/g)?.length ?? 0;
  assert.equal(picker.match(/loadedQuants=\{/g)?.length ?? 0, expanders);
  assert.match(
    picker,
    /loadedQuants\?\.some\(\(q\) => ggufVariantsMatchForPicker\(q, v\.quant\)\) \? \(\s*<span[^>]*>\s*loaded\s*<\/span>/,
  );
  assert.match(picker, /loadedModels\s*\.filter\(\(m\) => m\.quant && modelIdsMatchForPicker\(m\.id, repoId\)\)/);
  assert.match(
    picker,
    /if \(activeGgufVariant && modelIdsMatchForPicker\(loadedModelId, repoId\)\) \{\s*quants\.push\(activeGgufVariant\);/,
  );
  assert.match(picker, /meta="GGUF"\s*quantChip=\{loadedQuants\.map\(ggufQuantChipLabel\)\.join\(", "\) \|\| undefined\}/);
});

test("a loaded row is selected, picked and ejected by its checkpoint, not its label", () => {
  const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(runtime, /const checkpoint = checkpoints\?\.\[i\] \|\| id;/);
  assert.match(runtime, /statusRes\.serving_checkpoints,\s*\);/);
  assert.match(runtime, /\.loadedModels\.map\(\(m\) => m\.checkpoint \?\? m\.id\)/);
  assert.match(picker, /const checkpoint = entry\.checkpoint \?\? entry\.id;\s*const optionKey = makeModelOptionKey\("loaded", checkpoint\);\s*const isSelected = modelIdsMatchForPicker\(value, checkpoint\);/);
  assert.match(runtime, /const known = new Map\(current\.map\(\(m\) => \[m\.checkpoint \?\? m\.id, m\]\)\);/);
  assert.match(picker, /onSelect\(checkpoint, \{/);
  assert.match(picker, /onEject\?\.\(checkpoint\);/);
});

test("an On Device GGUF row with several quants shows a right/down chevron its gutter toggles", () => {
  const start = picker.indexOf("const renderDownloadedGgufRow = (");
  const row = picker.slice(start, picker.indexOf("<GgufVariantExpander", start));
  assert.match(row, /if \(soleQuant\) return renderSoleQuantGgufRow\(c, soleQuant\);/);
  assert.match(
    row,
    /\{soleQuants\.pending\.has\(c\.repo_id\) \? null : \(\s*<button[\s\S]*?onClick=\{\(\) => toggleGgufExpanded\(c\.repo_id, expanderOpen\)\}/,
  );
  assert.match(row, /expanderOpen\s*\?\s*ChevronDownStandardIcon\s*:\s*ChevronRightStandardIcon/);
  assert.match(row, /<span className=\{cn\(ROW_ACTIONS_PINNED_CLASS, "h-6"\)\}>/);
});

test("a loaded hub GGUF row opens its quant list, so quants downloaded after the load stay reachable", () => {
  const start = picker.indexOf("const renderLoadedRow = (");
  const loadedRow = picker.slice(start, picker.indexOf("const renderConnectedModelRow", start));
  assert.match(picker, /if \(loadedRowIds\.has\(id\.toLowerCase\(\)\)\) return moreLabel;/);
  assert.match(loadedRow, /const expandable = Boolean\(quant\) && !isLocal && isKnownGgufRepo\(entry\.id\);/);
  assert.match(loadedRow, /expandable\s*\?\s*setExpandedGguf\(\(prev\) =>\s*prev === entry\.id \? null : entry\.id,?\s*\)/);
  assert.match(loadedRow, /\{expanded && \(\s*<GgufVariantExpander[\s\S]*?repoId=\{entry\.id\}/);
});
