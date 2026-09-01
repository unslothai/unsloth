// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function source(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
}

for (const [name, page] of [
  ["images", "../src/features/images/images-page.tsx"],
  ["video", "../src/features/video/video-page.tsx"],
] as const) {
  test(`${name} pins family override through plan and load`, () => {
    const text = source(page);
    assert.ok(text.includes('"family_override"'));
    assert.ok(text.includes("family_override: advanced.family_override"));
    assert.ok(text.includes("familyOverride={familyOverride}"));
    assert.ok(
      text.includes("familyOverrideOptions(status?.supported_families)"),
    );
  });
}

test("image defaults use explicit and resolved family keys for opaque paths", () => {
  const text = source("../src/features/images/images-page.tsx");
  assert.ok(text.includes("defaultsFor(defaultsKeyFor(repoId, familyOverride))"));
  assert.ok(
    text.includes("const seedKey = `${repoId}\\0${residentDefaults}`"),
  );
  assert.ok(text.includes("defaultsFor(residentDefaults)"));
});

test("video defaults use the explicit family for opaque paths", () => {
  const text = source("../src/features/video/video-page.tsx");
  assert.ok(text.includes("defaultsFor(defaultsKeyFor(repoId, familyOverride))"));
  assert.ok(text.includes("MODEL_DEFAULTS.some((entry) => id.includes(entry.match))"));
});

test("pinned pipeline paths retain their Hub selector identity across remounts", () => {
  for (const file of [
    "../src/features/images/images-page.tsx",
    "../src/features/video/video-page.tsx",
  ]) {
    const text = source(file);
    assert.ok(text.includes("display_repo_id: opts.displayRepoId"), file);
    assert.ok(text.includes("displayRepoId: l.displayRepoId"), file);
    assert.ok(text.includes("status.display_repo_id ?? status.repo_id"), file);
    assert.ok(!text.includes("pinnedPipelineDisplayIds"), file);
    assert.ok(text.includes("loadedModelIdOverride={selectorModelId}"), file);
  }
  const picker = source(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.ok(picker.includes("isPinnedDiffusionLoadId(c.repo_id, c.load_id)"));
});
