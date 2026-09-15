// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getArtifactFilename } from "../src/features/chat/artifacts/types.ts";

test("the page's own title names the download", () => {
  // A fence-sourced card is titled "HTML preview" by the UI, so without reading
  // the document every generated page downloaded as `html-preview.html`.
  assert.equal(
    getArtifactFilename({
      title: "HTML preview",
      code: "<html><head><title>Snake Game</title></head><body></body></html>",
    }),
    "snake-game.html",
  );
});

test("the card title carries a document with no title", () => {
  assert.equal(
    getArtifactFilename({ title: "HTML canvas", code: "<div>hi</div>" }),
    "html-canvas.html",
  );
});

test("a still-streaming document degrades to the card title", () => {
  assert.equal(
    getArtifactFilename({ title: "HTML preview", code: "<html><head><tit" }),
    "html-preview.html",
  );
});

test("an empty title falls back to canvas", () => {
  assert.equal(getArtifactFilename({ title: "   ", code: "" }), "canvas.html");
});

test("the slug stays clipped at 48 characters", () => {
  const long = "a".repeat(80);
  const name = getArtifactFilename({
    title: "ignored",
    code: `<title>${long}</title>`,
  });
  assert.equal(name, `${"a".repeat(48)}.html`);
});
