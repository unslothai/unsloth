// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BackgroundEffect } from "./model.ts";

// Recursive ordered dither: low coordinate bits give the most significant rank bits.
function orderedRank(column: number, row: number): number {
	let rank = 0;
	for (let bit = 0; bit < 2; bit++) {
		const horizontal = (column >> bit) & 1;
		const vertical = (row >> bit) & 1;
		rank = rank * 4 + 2 * (horizontal ^ vertical) + vertical;
	}
	return rank;
}

// 5x7 stroked glyphs with 1px right/bottom spacing: no platform font rasterization.
function punctuationInk(level: number, x: number, y: number): boolean {
	if (x === 5 || y === 7) return false;
	switch (level) {
		case 1:
			return x === 2 && y === 5; // .
		case 2:
			return x === 2 && (y === 1 || y === 4); // :
		case 3:
			return y === 3 && x >= 1 && x <= 3; // -
		case 4:
			return (y === 2 || y === 4) && x >= 1 && x <= 3; // =
		case 5:
			return y === 3 || (x === 2 && y >= 1 && y <= 5); // +
		case 6: // *
			return (
				y === 3 ||
				((y === 1 || y === 5) && x % 2 === 0) ||
				((y === 2 || y === 4) && x >= 1 && x <= 3)
			);
		case 7:
			return y === 2 || y === 4 || x === 1 || x === 3; // #
		case 8: // %
			return (
				((y === 0 || y === 6) && (x === 0 || x === 4)) ||
				(y >= 1 && y <= 5 && x === 4 - Math.round(y * 0.8))
			);
		case 9: // @
			if (y === 0 || y === 6) return x >= 1 && x <= 3;
			if (y === 1) return x === 0 || x === 4;
			if (y === 5) return x === 0;
			return x === 0 || x === 4 || x === 2 || (x === 3 && y !== 3);
		default:
			return false;
	}
}

export function renderEffect(
	source: Uint8ClampedArray,
	width: number,
	height: number,
	effect: BackgroundEffect,
	light: boolean,
): Uint8ClampedArray {
	if (
		![width, height].every((n) => Number.isSafeInteger(n) && n > 0) ||
		source.length / 4 !== width * height
	) {
		throw new Error("Invalid image dimensions");
	}
	const pixels = source.slice();
	if (effect === "None") return pixels;

	if (effect === "Scanlines") {
		for (let row = 0; row < height; row += 3) {
			const end = (row + 1) * width * 4;
			for (let offset = row * width * 4; offset < end; offset += 4) {
				for (let channel = 0; channel < 3; channel++) {
					const value = source[offset + channel];
					pixels[offset + channel] = Math.floor(
						light ? value + (255 - value) * 0.48 : value * 0.52,
					);
				}
			}
		}
		return pixels;
	}

	const tileWidth = effect === "ASCII" ? 6 : effect === "Dither" ? 2 : 4;
	const tileHeight = effect === "ASCII" ? 8 : tileWidth;
	const paper = light ? 255 : 0;
	const index = (x: number, y: number) =>
		4 * (Math.min(y, height - 1) * width + Math.min(x, width - 1));
	const brightness = (offset: number) =>
		(0.2126 * source[offset] +
			0.7152 * source[offset + 1] +
			0.0722 * source[offset + 2]) /
		255;

	// Clip the last cell so odd-sized uploads keep the pattern.
	for (let top = 0; top < height; top += tileHeight) {
		for (let left = 0; left < width; left += tileWidth) {
			const center = index(left + tileWidth / 2, top + tileHeight / 2);
			const color = source.subarray(center, center + 3);
			const level = brightness(
				effect === "Halftone" ? index(left, top) : center,
			);
			const density = Math.sqrt(light ? 1 - level : level);
			const glyph = Math.min(9, Math.floor(density * 9));
			const radius = 2 * (0.3 + 0.7 * density);
			const maximum = Math.max(...color);
			const lit = maximum / 255 > (orderedRank(left / 2, top / 2) + 0.5) / 16;
			const amplification = lit ? 255 / Math.max(1, maximum) : 0.08;
			const dot = Array.from(color, (value) =>
				Math.round(value * amplification),
			);

			for (let dy = 0; dy < Math.min(tileHeight, height - top); dy++) {
				for (let dx = 0; dx < Math.min(tileWidth, width - left); dx++) {
					const offset = ((top + dy) * width + left + dx) * 4;
					const coverage =
						effect === "ASCII"
							? Number(punctuationInk(glyph, dx, dy))
							: Math.max(
									0,
									Math.min(1, radius + 0.5 - Math.hypot(dx - 1.5, dy - 1.5)),
								);
					for (let channel = 0; channel < 3; channel++) {
						if (effect === "Dither") {
							pixels[offset + channel] = dot[channel];
						} else {
							const texture =
								color[channel] * coverage + paper * (1 - coverage);
							pixels[offset + channel] = Math.floor(
								source[offset + channel] * 0.6 + texture * 0.4,
							);
						}
					}
				}
			}
		}
	}
	return pixels;
}
