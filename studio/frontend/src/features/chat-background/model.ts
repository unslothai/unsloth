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
export const MATERIALS = [
	"solid",
	"transparent",
	"frosted",
	"blurred",
	"liquid",
	"liquid-3d",
] as const;
export type WindowMaterial = (typeof MATERIALS)[number];
export function sanitizeMaterial(value: unknown): WindowMaterial {
	return MATERIALS.includes(value as WindowMaterial)
		? (value as WindowMaterial)
		: "solid";
}
export function sanitizeStrength(value: unknown, fallback = 50): number {
	return typeof value === "number" && Number.isFinite(value)
		? Math.max(0, Math.min(100, Math.round(value)))
		: fallback;
}
export type GlassStrengths = { light: number; dark: number };
export const DEFAULT_GLASS_STRENGTH: GlassStrengths = { light: 40, dark: 15 };
export function sanitizeStrengths(value: unknown): GlassStrengths {
	const stored =
		value && typeof value === "object" ? (value as Partial<GlassStrengths>) : {};
	return {
		light: sanitizeStrength(stored.light, DEFAULT_GLASS_STRENGTH.light),
		dark: sanitizeStrength(stored.dark, DEFAULT_GLASS_STRENGTH.dark),
	};
}
export type ImageTone = { lo: number; mid: number; hi: number };
export function toneOf(values: number[]): ImageTone {
	const sorted = [...values].sort((a, b) => a - b);
	const at = (q: number) =>
		sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? 0;
	return { lo: at(0.25), mid: at(0.5), hi: at(0.75) };
}

const clamp = (v: number, lo: number, hi: number) =>
	Math.max(lo, Math.min(hi, v));

/** Target pane luminance: under ~0.15 in dark mode, over ~0.72 in light mode (muted ink is mid-grey). */
export function glassReadability(
	tone: ImageTone,
	dark: boolean,
): { floor: number; shade: number } {
	if (dark)
		return {
			floor: clamp(1 - 0.15 / Math.max(tone.hi, 1e-3), 0, 0.9),
			shade: 0,
		};
	return {
		floor:
			tone.lo >= 0.72 ? 0 : clamp((0.72 - tone.lo) / (0.95 - tone.lo), 0, 0.88),
		shade: clamp((tone.mid - 0.45) / 0.4, 0, 1) * 0.12,
	};
}
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
	// Glass-only, so toggling glass never changes the non-glass look.
	glassLightVisibility: 70,
	glassDarkVisibility: 40,
	glassFade: false,
};
export type BackgroundAdjustments = typeof DEFAULT_ADJUSTMENTS;
export function visibilityKey(
	glass: boolean,
	dark: boolean,
): "visibility" | "glassLightVisibility" | "glassDarkVisibility" {
	if (!glass) return "visibility";
	return dark ? "glassDarkVisibility" : "glassLightVisibility";
}
export function fadeKey(glass: boolean): "fade" | "glassFade" {
	return glass ? "glassFade" : "fade";
}
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
		glassLightVisibility: bounded(input.glassLightVisibility, 100, 70),
		glassDarkVisibility: bounded(input.glassDarkVisibility, 100, 40),
		glassFade: typeof input.glassFade === "boolean" ? input.glassFade : false,
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
