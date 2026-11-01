// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// Enable to restore wallpaper on every main workspace page. The sidebar stays solid.
export const BACKGROUND_ON_ALL_PAGES = false;

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
/** Relative luminance of the image's darker quarter, its middle and its brighter quarter. */
export type ImageTone = { lo: number; mid: number; hi: number };
export function toneOf(values: number[]): ImageTone {
	const sorted = [...values].sort((a, b) => a - b);
	const at = (q: number) =>
		sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? 0;
	return { lo: at(0.25), mid: at(0.5), hi: at(0.75) };
}

const clamp = (v: number, lo: number, hi: number) =>
	Math.max(lo, Math.min(hi, v));

/** Dark mode: pane under ~0.15 luminance over the bright quarter; light mode: over ~0.72 over the dark quarter (pane ~0.95), since muted ink is mid-grey. */
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
	effect: "None" as BackgroundEffect,
	visibility: 70,
	darkVisibility: 40,
	blur: 0,
	// null: off under glass, on without it, until the user picks.
	fade: null as boolean | null,
	outline: true,
	sizing: "cover" as ImageSizing,
	zoom: 100,
	position: "50% 50%",
};
export type BackgroundAdjustments = typeof DEFAULT_ADJUSTMENTS;
export function effectiveFade(fade: boolean | null, glass: boolean): boolean {
	return fade ?? !glass;
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
		effect: EFFECTS.includes(input.effect as BackgroundEffect)
			? input.effect!
			: "None",
		visibility: bounded(input.visibility, 100, 70),
		darkVisibility: bounded(input.darkVisibility, 100, 40),
		blur: bounded(input.blur, 20, 0),
		fade: typeof input.fade === "boolean" ? input.fade : null,
		outline: typeof input.outline === "boolean" ? input.outline : true,
		sizing: IMAGE_SIZING.includes(input.sizing as ImageSizing)
			? input.sizing!
			: "cover",
		zoom: Math.max(100, bounded(input.zoom, 300, 100)),
		position: POSITIONS.includes(input.position as (typeof POSITIONS)[number])
			? input.position!
			: "50% 50%",
	};
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
