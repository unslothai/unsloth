import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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
	assert.equal(sanitizeAdjustments({ fade: false }).fade, false);
    assert.equal(sanitizeAdjustments({ outline: false }).outline, false);
    assert.equal(sanitizeAdjustments({ outline: "false" as never }).outline, true);
    assert.equal(sanitizeAdjustments({ sizing: "contain" }).sizing, "contain");
    assert.equal(sanitizeAdjustments({ sizing: "invalid" as never }).sizing, "cover");
    assert.equal(sanitizeAdjustments({ zoom: 500 }).zoom, 300);
    assert.equal(sanitizeAdjustments({ zoom: 0 }).zoom, 100);
    assert.equal(sanitizeAdjustments({ zoom: NaN }).zoom, 100);
	assert.equal(sanitizeAdjustments({ fade: "false" as never }).fade, true);
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

test("dither preserves hue across the ordered threshold pattern", () => {
	// RGB(128,64,32) lights thresholds 0-7; other dots use an 8% gain; each dot is 2x2 px.
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

// Digests captured before the renderer rewrite; cover odd sizes, alpha, color and density.
const appearanceDigests: Record<string, string> = {
	"None:false":
		"7ee58829538bc7f04cd4e5ba7cc284309f7a8c9322b353ffcd728cdac36bd691",
	"None:true":
		"7ee58829538bc7f04cd4e5ba7cc284309f7a8c9322b353ffcd728cdac36bd691",
	"Dither:false":
		"d489c8d42fab551fb0eb18f2de57295c069b0b13b7b265a38a4a19a8c116952b",
	"Dither:true":
		"d489c8d42fab551fb0eb18f2de57295c069b0b13b7b265a38a4a19a8c116952b",
	"ASCII:false":
		"8f42a61785faaa10a0f21a07e5c9a9feb0667f1603d89c13aed50d4d2409dec5",
	"ASCII:true":
		"d6e415288dc7da96d7eb1fe696dce2f9df6f4dc0ccd26e8f03a6ef0a57f916ee",
	"Halftone:false":
		"7f4bda226cb36549fa8a21912645b258eb6db4bf8134a61ccb014fca4293681f",
	"Halftone:true":
		"c91bb71945d000777670219d46a3bbddbfaef7b7d22a646632cab2b32e14d39e",
	"Scanlines:false":
		"e308c1f64f34f3adf98838c554e9bf60a8d6a5f812255184ecc7272f178f7b77",
	"Scanlines:true":
		"b4b309f732feb40d141d535a71c64313db602b32cc494190c166853bbc06d341",
};

test("effect textures retain their established appearance", () => {
	for (const effect of EFFECTS) {
		for (const light of [false, true]) {
			const digest = createHash("sha256");
			for (const [width, height] of [
				[1, 1],
				[2, 3],
				[7, 9],
				[17, 19],
				[64, 48],
			]) {
				for (let seed = 0; seed < 32; seed++) {
					const image = Uint8ClampedArray.from(
						{ length: width * height * 4 },
						(_, i) =>
							seed < 16
								? (i * 37 + seed * 17) % 256
								: [seed * 8 - 128, seed * 8 - 128, seed * 8 - 128, 255][i % 4],
					);
					digest.update(renderEffect(image, width, height, effect, light));
				}
			}
			assert.equal(
				digest.digest("hex"),
				appearanceDigests[`${effect}:${light}`],
				`${effect}, light=${light}`,
			);
		}
	}
});
