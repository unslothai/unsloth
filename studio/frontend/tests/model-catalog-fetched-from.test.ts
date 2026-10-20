// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A catalog row on a gated vendor id is still a one-click download when the backend fetches it from an
// open unsloth mirror, so it carries `fetchedFrom` and no `gated: true` (which would make the router skip
// it). The nightly network check then judges the mirror. That is only true while the backend really
// does redirect the id, so every `fetchedFrom` must be one of diffusion_families' gated mirror pairs.

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

const FAMILIES = fileURLToPath(
  new URL("../../backend/core/inference/diffusion_families.py", import.meta.url),
);

function gatedMirrorPairs(): Set<string> {
  const source = readFileSync(FAMILIES, "utf8");
  const pairs = new Set<string>();
  for (const table of ["_GATED_MIRROR_PAIRS", "_VIDEO_GATED_MIRROR_PAIRS"]) {
    const body = source.match(new RegExp(`^${table}: [^=]+= \\(\\n([\\s\\S]*?)^\\)`, "m"));
    assert.ok(body, `diffusion_families.py no longer defines ${table}`);
    for (const [, upstream, mirror] of body[1].matchAll(/\("([^"]+)",\s*"([^"]+)"\)/g)) {
      pairs.add(`${upstream} -> ${mirror}`);
    }
  }
  assert.ok(pairs.size >= 10, `read only ${pairs.size} gated mirror pairs; the table format moved`);
  return pairs;
}

const rows = [...IMAGE_CATALOG, ...VIDEO_CATALOG].flatMap((group) => group.artifacts);

test("every fetchedFrom is a mirror the backend actually fetches the vendor id from", () => {
  const pairs = gatedMirrorPairs();
  const declared = rows.filter((artifact) => artifact.fetchedFrom);
  assert.ok(declared.length > 0, "no catalog row declares fetchedFrom any more");
  for (const artifact of declared) {
    assert.ok(
      pairs.has(`${artifact.repoId} -> ${artifact.fetchedFrom}`),
      `${artifact.repoId} declares fetchedFrom ${artifact.fetchedFrom}, which is not in _GATED_MIRROR_PAIRS`,
    );
    assert.notEqual(artifact.gated, true, `${artifact.repoId}: a mirrored row is not gated for the user`);
  }
});

test("the FLUX.3 Klein bf16 rows on the gated BFL ids name their open mirror", () => {
  for (const id of ["black-forest-labs/flux-3-image-klein", "black-forest-labs/flux-3-image-klein-base"]) {
    const row = rows.find((artifact) => artifact.repoId === id && artifact.format === "bf16");
    assert.ok(row, id);
    assert.equal(row.fetchedFrom, `unsloth/${id.split("/")[1]}-FP8`);
  }
});
