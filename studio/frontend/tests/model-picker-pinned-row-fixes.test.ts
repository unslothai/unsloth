// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const pickers = readSrc(
  "features/model-picker/components/model-selector/pickers.tsx",
);

test("the Loaded tag beside a name is centred, not on the baseline", () => {
  assert.match(
    pickers,
    /label="Loaded"\s*className="ml-2 [^"]*\bself-center\b[^"]*"/,
  );
});

test("a repo whose only quant is pinned is not listed again below Pinned", () => {
  assert.match(pickers, /if \(sole && repo && sole\.variant\.quant === entry\.quant\) \{/);
  for (const list of ["unslothCachedGguf", "otherCachedGguf"]) {
    assert.match(
      pickers,
      new RegExp(
        `const ${list} = useMemo\\([\\s\\S]*?!pinnedSoleQuantRepoIds\\.has\\(c\\.repo_id\\)`,
      ),
    );
  }
});

test("a pinned quant row shows its size and vision mark", () => {
  assert.match(pickers, /sizes: new Map\(groups\.flatMap\(\(group\) => group\.sizes\)\)/);
  assert.match(
    pickers,
    /meta=\{\s*sizeBytes \? `GGUF · \$\{formatBytes\(sizeBytes\)\}` : "GGUF"\s*\}/,
  );
  assert.match(pickers, /quantChip=\{ggufQuantChipLabel\(entry\.quant\)\}\s*showVision=\{hasVision\}/);
});

test("that repo's pinned row is its own sole-quant row", () => {
  assert.match(
    pickers,
    /const renderPinnedQuantRow = [^\n]*\n(?:\s*\/\/[^\n]*\n)*\s*const soleRow = pinnedSoleQuantRows\.get\(pinKey\(entry\.repoId, entry\.quant\)\);\n\s*if \(soleRow\) return renderSoleQuantGgufRow\(soleRow\.repo, soleRow\.sole\);/,
  );
  // Keyed by repo, so the selected-option lookup (`::${value}`) still finds it.
  assert.match(
    pickers,
    /pinnedSoleQuantRows\.has\(row\.key\)\s*\?\s*makeModelOptionKey\("downloaded-gguf", row\.entry\.repoId\)/,
  );
});

test("the quant list's Vision badge and relay follow the picker's capability scope", () => {
  assert.match(
    pickers,
    /const visionInScope =\s*capabilityScope === null \|\| capabilityScope\.includes\("vision"\);\s*const showVision = hasVision === true && visionInScope;/,
  );
  assert.match(
    pickers,
    /if \(normalized\.hasVision !== undefined && visionInScope\) \{\s*onHasVision\?\.\(normalized\.hasVision\);/,
  );
  assert.doesNotMatch(
    pickers,
    /\{hasVision && \(\s*<span className="flex items-center gap-0\.5/,
  );
  assert.doesNotMatch(pickers, /!onDevice &&\s*hasVision &&/);
});

test("collapsed, pinned and sole-quant rows hide has_vision outside a vision scope too", () => {
  // Only the expander honoured the scope, so a one-quant audio row still drew Vision.
  const rowStart = pickers.indexOf("function ModelRow(");
  const row = pickers.slice(rowStart, pickers.indexOf("\nfunction ", rowStart + 10));
  assert.match(
    row,
    /const visionShown =\s*showVision === true &&\s*\(capabilityScope === null \|\| capabilityScope\.includes\("vision"\)\);/,
  );
  assert.doesNotMatch(row, /\{showVision && <VisionBadge \/>\}/);
  assert.equal(row.match(/\{visionShown && <VisionBadge \/>\}/g)?.length, 2);
});
