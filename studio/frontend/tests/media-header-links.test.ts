// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const LINK = readSrc("components/media-page-link.tsx");
const API = readSrc("components/media-api-button.tsx");
const IMAGES = readSrc("features/images/images-page.tsx");
const VIDEO = readSrc("features/video/video-page.tsx");
const STORE = readSrc("features/images/stores/image-workflow-store.ts");

test("media header links are the chat header's icon buttons, without a divider", () => {
  assert.match(
    LINK,
    /export const HEADER_ICON_BUTTON =\s*"flex size-\[calc\(30px\*var\(--ui-space-scale,1\)\)\] shrink-0 cursor-pointer items-center justify-center rounded-\[10px\] text-nav-fg/,
  );
  const iconOnly = LINK.slice(
    LINK.indexOf("if (iconOnly) {"),
    LINK.indexOf("  return (\n    <>"),
  );
  assert.match(iconOnly, /className=\{HEADER_ICON_BUTTON\}/);
  assert.match(iconOnly, /className="size-icon"/);
  assert.doesNotMatch(iconOnly, /w-px|ArrowRightIcon|labelClassName/);
  assert.match(API, /className=\{HEADER_ICON_BUTTON\}/);
  assert.doesNotMatch(API, /@\[84rem\]/);
});

test("Images links to Video and Video to Images, beside API and Library", () => {
  assert.match(
    IMAGES,
    /<MediaApiButton\s+kind="images"[\s\S]{0,120}\/>\s*<SiblingMediaPageLink to="\/video" \/>\s*<LibraryPageLink tab="images" iconOnly=\{true\} \/>/,
  );
  assert.match(
    VIDEO,
    /<MediaApiButton\s+kind="video"[\s\S]{0,120}\/>\s*<SiblingMediaPageLink to="\/images" \/>\s*<LibraryPageLink tab="videos" iconOnly=\{true\} \/>/,
  );
  assert.match(
    LINK,
    /icon=\{to === "\/images" \? Image03Icon : FlimSlateIcon\}/,
  );
});

test("Video has Create and Train, Train being Images' trainer", () => {
  assert.match(VIDEO, /dataTour="video-mode"/);
  assert.match(VIDEO, /\{ value: "train", label: "Train"/);
  assert.match(VIDEO, /\{pageMode === "train" \? \(\s*<DiffusionTrainPanel/);
  assert.match(VIDEO, /\{pageMode === "train" \? \(\s*<TrainBaseSelector/);
  // Video loads no adapters: Deploy hands the run to Images.
  assert.match(
    VIDEO,
    /useImageWorkflowStore\.getState\(\)\.requestDeploy\(args\);\s*void navigateToImages\(\{ to: "\/images" \}\);/,
  );
  assert.match(
    IMAGES,
    /const request = useImageWorkflowStore\.getState\(\)\.takeDeploy\(\);\s*if \(request\) handleDeployAdapter\(request\);/,
  );
  assert.match(
    IMAGES,
    /if \(trainingCompletions > 0\) setLoraRefreshKey\(\(k\) => k \+ 1\);/,
  );
  assert.match(
    STORE,
    /takeDeploy: \(\) => \{\s*const request = get\(\)\.pendingDeploy;\s*if \(request\) set\(\{ pendingDeploy: null \}\);/,
  );
});
