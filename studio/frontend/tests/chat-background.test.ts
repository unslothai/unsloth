import assert from "node:assert/strict";
import test from "node:test";
import { renderEffect } from "../src/features/chat-background/effects.ts";
import {
	DEFAULT_ADJUSTMENTS,
	EFFECTS,
	sanitizeAdjustments,
	normalizedSize,
	validUpload,
} from "../src/features/chat-background/model.ts";
import {
	contrast,
	safeArtworkOpacity,
	type RGB,
} from "../src/features/chat-background/contrast.ts";
test("background preferences repair malformed persisted data", () => {
	assert.deepEqual(sanitizeAdjustments({}), DEFAULT_ADJUSTMENTS);
	assert.deepEqual(
		sanitizeAdjustments({
			effect: "invalid" as never,
			visibility: 200,
			blur: -12,
			position: "url(evil)",
		}),
		{ ...DEFAULT_ADJUSTMENTS, visibility: 100, blur: 0 },
	);
	assert.equal(sanitizeAdjustments({ visibility: NaN }).visibility, 70);
});
test("uploads bound processing dimensions", () => {
	assert.deepEqual(normalizedSize(6000, 4000), [2048, 1365]);
	assert.deepEqual(normalizedSize(3, 7), [3, 7]);
	assert.throws(() => normalizedSize(10000, 10000));
	assert.throws(() => normalizedSize(0, 1));
	assert.equal(validUpload({ type: "image/svg+xml", size: 40 }), false);
	assert.equal(
		validUpload({ type: "image/png", size: 21 * 1024 * 1024 }),
		false,
	);
	assert.equal(validUpload({ type: "image/png", size: 1024 }), true);
});
test("effects preserve alpha, bounds, input and deterministic output at cell edges", () => {
	const source = Uint8ClampedArray.from(
			{ length: 13 * 11 * 4 },
			(_, i) => (i * 37) % 256,
		),
		copy = new Uint8ClampedArray(source);
	for (const effect of EFFECTS)
		for (const light of [false, true]) {
			const result = renderEffect(source, 13, 11, effect, light);
			assert.equal(result.length, source.length);
			assert.deepEqual(result, renderEffect(source, 13, 11, effect, light));
			for (let i = 3; i < source.length; i += 4)
				assert.equal(result[i], source[i]);
			if (effect !== "None") assert.notDeepEqual(result, source);
			else assert.deepEqual(result, source);
			assert.deepEqual(source, copy);
		}
});
test("theme-dependent effects adapt to light mode", () => {
	const source = new Uint8ClampedArray(8 * 8 * 4).fill(100);
	for (const effect of EFFECTS) {
		const a = renderEffect(source, 8, 8, effect, false),
			b = renderEffect(source, 8, 8, effect, true);
		if (effect === "None" || effect === "Dither") assert.deepEqual(a, b);
		else assert.notDeepEqual(a, b);
	}
	assert.throws(() => renderEffect(source, 0, 8, "None", false));
});
test("contrast protection holds for arbitrary imagery and custom palettes", () => {
	const themes: [RGB, RGB[]][] = [
		[
			[255, 255, 255],
			[
				[20, 20, 20],
				[110, 110, 110],
			],
		],
		[
			[16, 14, 24],
			[
				[245, 245, 245],
				[160, 156, 170],
			],
		],
		[
			[60, 85, 70],
			[
				[220, 240, 225],
				[190, 200, 195],
			],
		],
		[[128, 128, 128], [[130, 130, 130]]],
	];
	for (const [bg, foregrounds] of themes) {
		const alpha = safeArtworkOpacity(bg, foregrounds);
		assert.ok(alpha >= 0 && alpha <= 1);
		for (let r = 0; r <= 255; r += 51)
			for (let g = 0; g <= 255; g += 51)
				for (let b = 0; b <= 255; b += 51) {
					const composite = bg.map(
						(v, i) => v * (1 - alpha) + [r, g, b][i] * alpha,
					) as unknown as RGB;
					for (const fg of foregrounds)
						assert.ok(
							contrast(composite, fg) >= Math.min(4.5, contrast(bg, fg)) - 1e-6,
						);
				}
	}
});

test("dither matches Zeron's opaque Bayer reference swatch", () => {
	// Upstream dither_color: RGB(128,64,32) lights thresholds 0–7;
	// the remaining dots use an 8% gain. Each dot occupies 2×2 pixels.
	const source = Uint8ClampedArray.from(
		{ length: 8 * 8 * 4 },
		(_, i) => [128, 64, 32, 255][i % 4],
	);
	const result = renderEffect(source, 8, 8, "Dither", false);
	for (let y = 0; y < 8; y++) {
		for (let x = 0; x < 8; x++) {
			const bright = (Math.floor(x / 2) + Math.floor(y / 2)) % 2 === 0;
			const offset = (y * 8 + x) * 4;
			assert.deepEqual(
				[...result.slice(offset, offset + 4)],
				bright ? [255, 128, 64, 255] : [10, 5, 3, 255],
			);
		}
	}
	assert.deepEqual(renderEffect(source, 8, 8, "Dither", true), result);
});
