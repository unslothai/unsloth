// SPDX-License-Identifier: MIT
// Adapted from Zeron's new_thread_background_effects.rs, Copyright (c) 2026 Wing.
// See THIRD_PARTY_LICENSE.txt in this directory for the permission notice.
import type { BackgroundEffect } from "./model.ts";
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
const GLYPHS = [
	[0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 4, 0],
	[0, 4, 0, 0, 4, 0, 0],
	[0, 0, 0, 14, 0, 0, 0],
	[0, 0, 14, 0, 14, 0, 0],
	[0, 4, 4, 31, 4, 4, 0],
	[0, 21, 14, 31, 14, 21, 0],
	[10, 10, 31, 10, 31, 10, 10],
	[17, 2, 4, 4, 8, 16, 17],
	[14, 17, 23, 21, 23, 16, 14],
];
/** Straight RGBA; independent of canvas, fonts, layout and device pixel ratio. */
export function renderEffect(
	source: Uint8ClampedArray,
	width: number,
	height: number,
	effect: BackgroundEffect,
	light: boolean,
): Uint8ClampedArray {
	if (
		!Number.isInteger(width) ||
		!Number.isInteger(height) ||
		width < 1 ||
		height < 1 ||
		source.length !== width * height * 4
	)
		throw new Error("Invalid image dimensions");
	const out = new Uint8ClampedArray(source);
	if (effect === "None") return out;
	const paper = light ? 255 : 0;
	const luminance = (i: number) =>
		(source[i] * 0.2126 + source[i + 1] * 0.7152 + source[i + 2] * 0.0722) /
		255;
	const at = (x: number, y: number) =>
		(Math.min(height - 1, y) * width + Math.min(width - 1, x)) * 4;
	for (let y = 0; y < height; y++)
		for (let x = 0; x < width; x++) {
			const i = at(x, y);
			if (effect === "Dither") {
				const sample = at(Math.floor(x / 2) * 2 + 1, Math.floor(y / 2) * 2 + 1);
				const peak = Math.max(
					source[sample],
					source[sample + 1],
					source[sample + 2],
				);
				const threshold =
					BAYER[(Math.floor(y / 2) % 4) * 4 + (Math.floor(x / 2) % 4)];
				const gain =
					peak / 255 > (threshold + 0.5) / 16 ? 255 / Math.max(peak, 1) : 0.08;
				for (let c = 0; c < 3; c++)
					out[i + c] = Math.round(source[sample + c] * gain);
			} else if (effect === "Scanlines") {
				const gain = y % 3 === 0 ? 0.52 : 1;
				for (let c = 0; c < 3; c++)
					out[i + c] = Math.floor(
						light
							? source[i + c] + (255 - source[i + c]) * (1 - gain)
							: source[i + c] * gain,
					);
			} else if (effect === "ASCII") {
				const sample = at(Math.floor(x / 6) * 6 + 3, Math.floor(y / 8) * 8 + 4);
				const density = light ? 1 - luminance(sample) : luminance(sample);
				const glyph = GLYPHS[Math.min(9, Math.floor(Math.sqrt(density) * 9))];
				const ink =
					x % 6 < 5 && y % 8 < 7 && (glyph[y % 8] & (1 << (4 - (x % 6)))) !== 0;
				for (let c = 0; c < 3; c++)
					out[i + c] = Math.floor(
						source[i + c] * 0.6 + (ink ? source[sample + c] : paper) * 0.4,
					);
			} else if (effect === "Halftone") {
				const l = luminance(at(Math.floor(x / 4) * 4, Math.floor(y / 4) * 4));
				const radius = 2 * (0.3 + 0.7 * Math.sqrt(light ? 1 - l : l));
				const coverage = Math.max(
					0,
					Math.min(1, radius + 0.5 - Math.hypot((x % 4) - 1.5, (y % 4) - 1.5)),
				);
				const sample = at(Math.floor(x / 4) * 4 + 2, Math.floor(y / 4) * 4 + 2);
				for (let c = 0; c < 3; c++)
					out[i + c] = Math.floor(
						source[i + c] * 0.6 +
							(source[sample + c] * coverage + paper * (1 - coverage)) * 0.4,
					);
			}
			// Preserve source transparency, including in partially transparent cells.
			out[i + 3] = source[i + 3];
		}
	return out;
}
