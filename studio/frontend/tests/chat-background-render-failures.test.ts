// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Could not render the effect" must mean the image or effect really failed, not a transient step.

import assert from "node:assert/strict";
import test from "node:test";
import { loadImage } from "../src/features/chat-background/image.ts";
import { readSrc } from "./helpers/kit.ts";

type Listener = () => void;
function fakeImage(opts: { decodeFails: boolean; loads: boolean }) {
	const listeners = new Map<string, Listener>();
	const image = {
		complete: false,
		naturalWidth: 0,
		addEventListener: (type: string, fn: Listener) => listeners.set(type, fn),
		decode: () =>
			opts.decodeFails
				? Promise.reject(new Error("EncodingError"))
				: Promise.resolve(),
		set src(_url: string) {
			queueMicrotask(() => {
				if (opts.loads) {
					image.complete = true;
					image.naturalWidth = 640;
					listeners.get("load")?.();
				} else listeners.get("error")?.();
			});
		},
	};
	return image as unknown as HTMLImageElement;
}

test("a rejected decode() is not a failure when the image still loads", async () => {
	await loadImage(fakeImage({ decodeFails: true, loads: true }), "blob:x");
	await loadImage(fakeImage({ decodeFails: false, loads: true }), "blob:x");
});

test("an image that never loads still fails", async () => {
	await assert.rejects(
		loadImage(fakeImage({ decodeFails: true, loads: false }), "blob:x"),
	);
});

test("every wallpaper decode goes through loadImage, not a bare decode()", () => {
	for (const file of ["background.tsx", "artwork.tsx", "image.ts"]) {
		const src = readSrc(`features/chat-background/${file}`);
		const bare = src.replace(/await image\.decode\(\);\n\t} catch \{\n\t\tif \(!\(image\.complete/, "");
		assert.doesNotMatch(bare, /await image\.decode\(\)/, `${file} awaits a bare decode()`);
	}
});

test("a worker that fails to start falls back to the main thread", () => {
	const render = readSrc("features/chat-background/render.ts");
	assert.match(render, /worker\.onerror = onMainThread;/);
	assert.match(render, /renderEffect\(read\(\), canvas\.width, canvas\.height, effect, light\)/);
});

test("a later successful render clears a stale effect error", () => {
	const background = readSrc("features/chat-background/background.tsx");
	assert.match(background, /adoptFrame\(url\);\n\t+clearBackgroundError\("effect"\);/);
});
