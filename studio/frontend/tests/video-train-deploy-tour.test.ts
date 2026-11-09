// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("video-family adapters get no Deploy to Create", () => {
  const src = readSrc("features/images/train/diffusion-train-panel.tsx");
  assert.match(src, /\?\.is_video === true/);
  assert.match(
    src,
    /viewRun\.catalog_path && !isVideoFamily\(viewRun\.family \|\| ""\) && \(\s*<Button[\s\S]{0,400}onDeploy\?\.\(/,
  );
  assert.match(
    src,
    /status\?\.catalog_path && !isVideoFamily\(status\.family \|\| family\?\.name \|\| ""\) && \(\s*<Button type="button" size="sm" onClick=\{onDeployClick\}>/,
  );
  assert.match(readSrc("features/images/api.ts"), /is_video\?: boolean;/);
});

test("the tour attaches to the first visible target, not a hidden page's", () => {
  const src = readSrc("features/tour/components/guided-tour.tsx");
  assert.match(src, /document\.querySelectorAll\(sel\)/);
  assert.match(src, /found\.find\(isUsableTarget\) \?\? found\[0\] \?\? null/);
  assert.doesNotMatch(src, /document\.querySelector\(sel\)/);
});

test("Video re-reads its model status when returning from Train to Create", () => {
  const src = readSrc("features/video/video-page.tsx");
  assert.match(
    src,
    /if \(pageMode === "train"\) \{\s*leftForTrain\.current = true;\s*return;\s*\}\s*if \(!active \|\| !leftForTrain\.current\) return;\s*leftForTrain\.current = false;\s*void refreshStatus\(\);\s*\}, \[active, pageMode, refreshStatus\]\);/,
  );
});

test("Images re-reads its model status when returning from Train to Create", () => {
  const src = readSrc("features/images/images-page.tsx");
  assert.match(
    src,
    /if \(pageMode === "train"\) \{\s*leftForTrain\.current = true;\s*return;\s*\}\s*if \(!active \|\| !leftForTrain\.current\) return;\s*leftForTrain\.current = false;\s*void refreshStatus\(\);\s*\}, \[active, pageMode, refreshStatus\]\);/,
  );
});

for (const page of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
  test(`${page} hides the Train panel instead of unmounting it`, () => {
    const src = readSrc(page);
    assert.match(src, /if \(pageMode === "train" && !trainMounted\) setTrainMounted\(true\);/);
    assert.match(
      src,
      /\{trainMounted && \(\s*<div className=\{pageMode === "train" \? "contents" : "hidden"\} inert=\{pageMode !== "train" \|\| undefined\}>\s*<DiffusionTrainPanel/,
    );
    assert.match(src, /\/>\s*<\/div>\s*\)\}\s*\{pageMode !== "train" && \(/);
    assert.doesNotMatch(src, /\{pageMode === "train" \? \(\s*<DiffusionTrainPanel/);
  });
}
