// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  ATTACHMENT_KIND_ICON_CLASS,
  ATTACHMENT_KIND_ICONS,
} from "../src/features/chat/lib/attachment-file-kind.ts";
import { IMAGES_ICON_CLASS, ImagesIcon } from "../src/lib/hugeicons-derived.ts";
import { readSrc } from "./helpers/kit.ts";

test("every image attachment shows the one Images icon in its own colour", () => {
  assert.equal(ATTACHMENT_KIND_ICONS.image, ImagesIcon);
  assert.equal(ATTACHMENT_KIND_ICON_CLASS.image, IMAGES_ICON_CLASS);
  const imageClass = IMAGES_ICON_CLASS;
  for (const [kind, cls] of Object.entries(ATTACHMENT_KIND_ICON_CLASS)) {
    if (kind !== "image") assert.notEqual(cls, imageClass, kind);
  }
  const library = readSrc("features/library/file-kind.ts");
  assert.match(library, /image: ImagesIcon,/);
  assert.match(library, /image: IMAGES_ICON_CLASS,/);
  assert.match(readSrc("components/media-viewer.tsx"), /icon=\{ImagesIcon\}/);
  assert.match(
    readSrc("components/assistant-ui/attachment.tsx"),
    /kind === "image" \? ATTACHMENT_KIND_ICONS\.image : FileEmpty02Icon/,
  );
});
