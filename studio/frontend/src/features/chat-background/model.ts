// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// On "all" the sidebar stays solid.
export const SCOPES = ["newChat", "chats", "all"] as const;
export type BackgroundScope = (typeof SCOPES)[number];

export const EFFECTS = [
	"None",
	"Dither",
	"ASCII",
	"Halftone",
	"Scanlines",
] as const;
export type BackgroundEffect = (typeof EFFECTS)[number];
export const IMAGE_SIZING = [
	"cover",
	"contain",
	"width",
	"height",
	"stretch",
	"original",
] as const;
export type ImageSizing = (typeof IMAGE_SIZING)[number];
export const POSITIONS = [
	"0% 0%",
	"50% 0%",
	"100% 0%",
	"0% 50%",
	"50% 50%",
	"100% 50%",
	"0% 100%",
	"50% 100%",
	"100% 100%",
] as const;
export const DEFAULT_ADJUSTMENTS = {
	scope: "chats" as BackgroundScope,
	effect: "None" as BackgroundEffect,
	visibility: 70,
	blur: 0,
	fade: true,
	outline: 30,
	sizing: "cover" as ImageSizing,
	zoom: 100,
	position: "50% 50%",
};
export type BackgroundAdjustments = typeof DEFAULT_ADJUSTMENTS;
export type ChatBackgroundValue = BackgroundAdjustments & { blob: Blob };
export type BackgroundError =
	| "storage"
	| "file"
	| "dimensions"
	| "decode"
	| "effect";
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
export const MAX_PIXELS = 40_000_000;
export const MAX_EDGE = 2048;
export function sanitizeAdjustments(
	input: Partial<BackgroundAdjustments>,
): BackgroundAdjustments {
	const bounded = (v: unknown, max: number, fallback: number) =>
		typeof v === "number" && Number.isFinite(v)
			? Math.max(0, Math.min(max, Math.round(v)))
			: fallback;
	return {
		scope: SCOPES.includes(input.scope as BackgroundScope)
			? input.scope!
			: "chats",
		effect: EFFECTS.includes(input.effect as BackgroundEffect)
			? input.effect!
			: "None",
		visibility: bounded(input.visibility, 100, 70),
		blur: bounded(input.blur, 20, 0),
		fade: typeof input.fade === "boolean" ? input.fade : true,
		// Booleans are the pre-slider saved shape.
		outline:
			typeof input.outline === "boolean"
				? input.outline
					? DEFAULT_ADJUSTMENTS.outline
					: 0
				: bounded(input.outline, 100, DEFAULT_ADJUSTMENTS.outline),
		sizing: IMAGE_SIZING.includes(input.sizing as ImageSizing)
			? input.sizing!
			: "cover",
		zoom: Math.max(100, bounded(input.zoom, 300, 100)),
		position: parsePosition(input.position) ? input.position! : "50% 50%",
	};
}
export function parsePosition(value: unknown): [number, number] | null {
	if (typeof value !== "string") return null;
	const match = /^(\d{1,3})% (\d{1,3})%$/.exec(value);
	if (!match) return null;
	const x = Number(match[1]),
		y = Number(match[2]);
	return x <= 100 && y <= 100 ? [x, y] : null;
}
export function validUpload(file: Pick<File, "type" | "size">): boolean {
	return (
		file.size > 0 &&
		file.size <= MAX_UPLOAD_BYTES &&
		["image/jpeg", "image/png", "image/webp"].includes(file.type)
	);
}
export function normalizedSize(
	width: number,
	height: number,
): [number, number] {
	if (
		width < 1 ||
		height < 1 ||
		!Number.isFinite(width * height) ||
		width * height > MAX_PIXELS
	)
		throw new Error("dimensions");
	const scale = Math.min(1, MAX_EDGE / Math.max(width, height));
	return [
		Math.max(1, Math.round(width * scale)),
		Math.max(1, Math.round(height * scale)),
	];
}
