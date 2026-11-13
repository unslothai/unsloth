// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { bakeScale, imageBox } from "../src/features/chat-background/artwork-geometry.ts";
import { readSrc } from "./helpers/kit.ts";

const ARTWORK = readSrc("features/chat-background/artwork.tsx");
const BACKGROUND = readSrc("features/chat-background/background.tsx");

test("the baked box is where CSS paints a no-repeat background for every sizing", () => {
	// A 400 x 200 image in a 1000 x 1000 box.
	assert.deepEqual(imageBox(1000, 1000, 400, 200, "cover", "50% 50%"), { x: -500, y: 0, w: 2000, h: 1000 });
	assert.deepEqual(imageBox(1000, 1000, 400, 200, "contain", "50% 50%"), { x: 0, y: 250, w: 1000, h: 500 });
	assert.deepEqual(imageBox(1000, 1000, 400, 200, "width", "0% 100%"), { x: 0, y: 500, w: 1000, h: 500 });
	assert.deepEqual(imageBox(1000, 1000, 400, 200, "height", "100% 0%"), { x: -1000, y: 0, w: 2000, h: 1000 });
	assert.deepEqual(imageBox(1000, 1000, 400, 200, "stretch", "30% 70%"), { x: 0, y: 0, w: 1000, h: 1000 });
	assert.deepEqual(imageBox(1000, 1000, 400, 200, "original", "50% 50%"), { x: 300, y: 400, w: 400, h: 200 });
	assert.deepEqual(imageBox(1000, 1000, 400, 200, "original", "bad"), { x: 300, y: 400, w: 400, h: 200 }, "an unparsable position centres, as the model does");
});

test("a wide blur bakes at a lower resolution, a narrow one at the device pixel ratio", () => {
	assert.equal(bakeScale(20, 2), 0.4);
	assert.equal(bakeScale(2, 2), 2);
	assert.equal(bakeScale(4, 1), 1);
	assert.equal(bakeScale(100, 3), 0.25, "never below a quarter of a CSS pixel");
});

test("the live blur filter shows only until the bitmap for the current inputs is baked", () => {
	assert.match(ARTWORK, /const ready = baked !== null && baked\.key === key;/);
	assert.match(ARTWORK, /backgroundSize: "100% 100%",[\s\S]{0,120}transform: `scale\(\$\{zoom \/ 100\}\)`/,
		"the baked bitmap covers the box and keeps the zoom transform, which CSS applies after the filter");
	assert.match(ARTWORK, /filter: blur \? `blur\(\$\{blur\}px\)` : undefined,/, "the pending path is the old live filter");
	assert.match(ARTWORK, /`\$\{url\}\|\$\{sizing\}\|\$\{position\}\|\$\{blur\}\|\$\{size\.w\}x\$\{size\.h\}\|/,
		"a new image, sizing, position, blur or box size bakes again");
	assert.match(ARTWORK, /octx\.filter = `blur\(\$\{blur \* k\}px\)`;\s*octx\.drawImage\(plain, 0, 0\);/,
		"the box is drawn clipped first, then blurred, as filter: blur() blurs the painted box");
	assert.match(ARTWORK, /"filter" in CanvasRenderingContext2D\.prototype/, "older WebKit without a canvas filter gets the downscale path");
	assert.match(ARTWORK, /URL\.revokeObjectURL\(bakedUrl\.current\)/, "baked URLs are released");
});

test("only the current frame bakes; a departing frame fades out with the live filter", () => {
	assert.match(BACKGROUND, /<WallpaperArtwork\s+url=\{frame\.url\}[\s\S]{0,200}bakeable=\{i === 1\}/);
	assert.equal(BACKGROUND.includes("filter: adjustment.blur"), false);
});
