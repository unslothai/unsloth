// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const picker = readSrc(
  "features/model-picker/components/model-selector/pickers.tsx",
);

function quantRowGear(): string {
  const start = picker.indexOf(
    "ariaLabel={`Inference settings for ${repoId} ${v.quant}`}",
  );
  assert.ok(start > 0, "the quant row has a settings gear");
  const open = picker.lastIndexOf(
    "{",
    picker.lastIndexOf("<ModelLoadSettingsAction", start),
  );
  return picker.slice(open, picker.indexOf("/>", start) + 2);
}

test("every GGUF quant row offers Run settings, downloaded or not", () => {
  const gear = quantRowGear();
  assert.match(gear, /^\{!unusableLocal && onConfigure && \(/);
  assert.doesNotMatch(gear, /downloaded/);
  // Every quant list renders through the expander, so no list keeps its own gate.
  assert.equal(
    picker.match(
      /ariaLabel=\{`Inference settings for \$\{repoId\} \$\{v\.quant\}`\}/g,
    )?.length,
    1,
  );
});

test("the gear targets the row's own quant and pick, without downloading", () => {
  const gear = quantRowGear();
  // Same meta as the row click, so Load from the page downloads or loads as the row would.
  assert.match(
    gear,
    /onConfigure=\{\(\) => onConfigure\(repoId, selectMeta\)\}/,
  );
  assert.match(picker, /onClick=\{\(\) => onSelect\(repoId, selectMeta\)\}/);
  const meta = picker.slice(
    picker.indexOf("const variantSelectMeta = useCallback("),
    picker.indexOf("const budgetFraction"),
  );
  assert.match(meta, /ggufVariant: quant,/);
  assert.match(meta, /isDownloaded: isLocalPath \? true : downloaded,/);
  assert.match(meta, /loadId: downloaded === true \? loadId : undefined,/);
  // Building the meta must not itself start anything.
  assert.doesNotMatch(meta, /onSelect\(|download\(/);
});

test("a quant without a menu keeps its slot so gears line up", () => {
  assert.match(
    picker,
    /const reserveMenuSlot = \(displayVariants \?\? \[\]\)\.some\(variantHasMenu\);/,
  );
  assert.match(
    picker,
    /\{!variantHasMenu\(v\) && reserveMenuSlot && \(\s*<span className="size-5 shrink-0" aria-hidden=\{true\} \/>\s*\)\}\s*\{variantHasMenu\(v\) && \(\s*<ModelRowMenu/,
  );
});
