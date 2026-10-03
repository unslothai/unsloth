// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const page = readSrc("features/audio/pages/edit-page.tsx");

test("the recording takes uploads, recordings and history, not saved voices", () => {
  assert.match(page, /id="edit-recording"[\s\S]*?allowSavedVoice=\{false\}/);
});

test("history hides an edit's original", () => {
  const gallery = readSrc("features/audio/hooks/use-audio-gallery.tsx");
  assert.match(gallery, /clip\.role !== "source"/);
});

test("the compare draws its bars from the server's file, since the CSP blocks fetching object URLs", () => {
  const compare = readSrc("features/audio/components/ab-compare.tsx");
  assert.match(compare, /if \(src\.startsWith\("blob:"\)\) return null;/);
  assert.match(page, /fileUrl: galleryFileUrl\(clip\.id\)/);
});
