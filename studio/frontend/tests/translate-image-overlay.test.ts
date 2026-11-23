// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  fitText,
  isImageTextBlock,
  medianColor,
  overlayFileName,
  readableTextColor,
  wrapWords,
} from "../src/features/translate/translate-image-overlay.ts";
import { translateTexts } from "../src/features/translate/translate-subtitles.ts";

import { readSrc } from "./helpers/kit.ts";

// One unit per character, so widths are easy to reason about.
const chars = (text: string) => Array.from(text).length;

test("the background is the median around the box, ignoring the text pixels and transparency", () => {
  const rgba = [
    ...[200, 100, 50, 255],
    ...[210, 110, 60, 255],
    ...[0, 0, 0, 255],
    ...[205, 105, 55, 255],
    ...[9, 9, 9, 0],
  ];
  assert.deepEqual(
    medianColor(rgba, (pixel) => pixel === 2),
    [205, 105, 55],
  );
  assert.equal(medianColor([1, 2, 3, 0]), null);
});

test("text is black on light fills and white on dark ones", () => {
  assert.equal(readableTextColor([250, 240, 200]), "#000000");
  assert.equal(readableTextColor([30, 90, 160]), "#ffffff");
  assert.equal(readableTextColor([0, 0, 0]), "#ffffff");
});

test("words wrap to the width, and an unspaced script breaks between characters", () => {
  assert.deepEqual(wrapWords("Hello there my friend", 11, chars), [
    "Hello there",
    "my friend",
  ]);
  assert.deepEqual(wrapWords("你好世界再见", 4, chars), ["你好世界", "再见"]);
  assert.deepEqual(wrapWords("  ", 4, chars), []);
});

test("the font grows until the wrapped lines would no longer fit the box", () => {
  const measure = (text: string, size: number) => chars(text) * size * 0.5;
  const fitted = fitText("Good morning everyone", 120, 40, measure);
  assert.ok(fitted.size >= 4);
  assert.ok(fitted.lines.length * fitted.size * 1.2 <= 40);
  assert.ok(fitted.lines.every((line) => measure(line, fitted.size) <= 120));
  const bigger = fitted.size + 1;
  const lines = wrapWords("Good morning everyone", 120, (s) =>
    measure(s, bigger),
  );
  assert.ok(
    lines.length * bigger * 1.2 > 40 ||
      lines.some((line) => measure(line, bigger) > 120),
  );
});

test("only well-formed blocks from the server are drawn", () => {
  const block = { text: "Exit", lines: 1, box: [0.1, 0.2, 0.3, 0.4] };
  assert.equal(isImageTextBlock(block), true);
  assert.equal(isImageTextBlock({ ...block, text: " " }), false);
  assert.equal(
    isImageTextBlock({ ...block, box: [0.3, 0.2, 0.1, 0.4] }),
    false,
  );
  assert.equal(isImageTextBlock({ ...block, box: [0.1, 0.2, 0.3] }), false);
  assert.equal(isImageTextBlock(null), false);
});

test("the image saves as a PNG named for the target language", () => {
  assert.equal(overlayFileName("Sign.JPG", "fr"), "Sign.fr.png");
  assert.equal(overlayFileName(".png", "de"), "image.de.png");
});

test("block texts go out as one numbered batch and come back in order", async () => {
  const sent: string[] = [];
  const texts = await translateTexts(
    ["Hello world", "Exit here"],
    async (source) => {
      sent.push(source);
      return "1. Bonjour le monde\n2. Sortie ici";
    },
    { signal: new AbortController().signal, onPhase: () => {} },
  );
  assert.deepEqual(sent, ["1. Hello world\n2. Exit here"]);
  assert.deepEqual(texts, ["Bonjour le monde", "Sortie ici"]);
});

test("only an image Tesseract boxed, still in the text box, offers the overlay", () => {
  const page = readSrc("features/translate/translate-page.tsx");
  assert.match(
    page,
    /const OVERLAY_IMAGE = \/\\\.\(png\|jpe\?g\|webp\|gif\|bmp\)\$\/i;/,
  );
  assert.match(
    page,
    /const found = \(blocks \?\? \[\]\)\.filter\(isImageTextBlock\);\s*setImageSource\(\s*found\.length > 0/,
  );
  assert.match(
    page,
    /result &&\s*!stale &&\s*!busy &&\s*imageSource\?\.text === source && \(/,
  );
  const api = readSrc("features/rag/api/rag-api.ts");
  assert.match(api, /if \(boxes\) form\.append\("boxes", "true"\);/);
});
