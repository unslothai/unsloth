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
  assert.match(iconOnly, /className=\{cn\(\s*HEADER_ICON_BUTTON,/);
  assert.match(iconOnly, /className="size-icon"/);
  assert.doesNotMatch(iconOnly, /w-px|labelClassName/);
  assert.match(API, /className=\{HEADER_ICON_BUTTON\}/);
  assert.doesNotMatch(API, /@\[84rem\]/);
});

test("API, Library, then the other media page with an arrow, at the chat header's inset", () => {
  assert.match(
    IMAGES,
    /<MediaApiButton\s+kind="images"[\s\S]{0,120}\/>\s*<LibraryPageLink tab="images" iconOnly=\{true\} \/>\s*<SiblingMediaPageLink to="\/video" \/>/,
  );
  assert.match(
    VIDEO,
    /<MediaApiButton\s+kind="video"[\s\S]{0,120}\/>\s*<LibraryPageLink tab="videos" iconOnly=\{true\} \/>\s*<SiblingMediaPageLink to="\/images" \/>/,
  );
  assert.match(
    LINK,
    /icon=\{to === "\/images" \? Image03Icon : FlimSlateIcon\}/,
  );
});

test("Video has Create and Train, Train being Images' trainer", () => {
  assert.match(VIDEO, /dataTour="video-mode"/);
  assert.match(VIDEO, /\{ value: "train", label: "Train"/);
  assert.match(VIDEO, /\{trainMounted && \(\s*<div className=\{pageMode === "train" \? "contents" : "hidden"\}[^>]*>\s*<DiffusionTrainPanel/);
  assert.match(VIDEO, /\{pageMode === "train" \? \(\s*<TrainBaseSelector/);
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

test("the sibling link keeps its icon where the square button puts it and trails an arrow", () => {
  assert.match(
    LINK,
    /iconArrow &&\s*"w-auto gap-1\.5 pr-2 pl-\[calc\(\(30px\*var\(--ui-space-scale,1\)-var\(--icon-size\)\)\/2\)\]"/,
  );
  assert.match(
    LINK,
    /\{iconArrow && \(\s*<>[\s\S]{0,200}<span className="hidden min-w-0 truncate text-ui-13 font-medium @\[50rem\]:inline">\s*\{label\}\s*<\/span>\s*<ArrowRightIcon className="size-3\.5 shrink-0 opacity-60" \/>/,
  );
  for (const page of [IMAGES, VIDEO]) {
    assert.match(
      page,
      /justify-end pr-\[calc\(var\(--thread-scrollbar-gutter,10px\)\+0\.5rem\*var\(--ui-space-scale,1\)\+var\(--studio-chat-header-right-inset,var\(--studio-window-control-inset,0px\)\)\)\]/,
    );
  }
});

test("Library's empty Images and Videos offer Upload and Generate", () => {
  const library = readSrc("features/library/library-page.tsx");
  assert.match(
    library,
    /images: \["\/images", "library\.empty\.generateImage"\]/,
  );
  assert.match(
    library,
    /if \(tab === "images" \|\| tab === "videos"\) \{[\s\S]{0,200}\{uploadButton\}\s*\{generate\}/,
  );
});

test("Create and Train get roomy tabs on Images and Video", () => {
  for (const page of [IMAGES, VIDEO]) {
    assert.match(page, /\[&>button\]:px-7 @\[68rem\]:\[&>button\]:px-14/);
  }
});

test("Reapply's icon sits at the button's own gap", () => {
  for (const page of [IMAGES, VIDEO]) {
    assert.match(page, /<HugeiconsIcon icon=\{Refresh01Icon\} className="size-4" \/>\s*Reapply/);
  }
});
