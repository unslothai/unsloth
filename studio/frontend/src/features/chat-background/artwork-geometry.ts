// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { type ImageSizing, parsePosition } from "./model.ts";

/** Where CSS paints a no-repeat background of an iw x ih image in a w x h box (CSS px). */
export function imageBox(
	w: number,
	h: number,
	iw: number,
	ih: number,
	sizing: ImageSizing,
	position: string,
): { x: number; y: number; w: number; h: number } {
	let dw = iw,
		dh = ih;
	if (sizing === "cover" || sizing === "contain") {
		const s = (sizing === "cover" ? Math.max : Math.min)(w / iw, h / ih);
		dw = iw * s;
		dh = ih * s;
	} else if (sizing === "width") {
		dw = w;
		dh = (ih * w) / iw;
	} else if (sizing === "height") {
		dh = h;
		dw = (iw * h) / ih;
	} else if (sizing === "stretch") {
		dw = w;
		dh = h;
	}
	const [px, py] = parsePosition(position) ?? [50, 50];
	return { x: ((w - dw) * px) / 100, y: ((h - dh) * py) / 100, w: dw, h: dh };
}

// After imageBox and scale(zoom) around the position, the image's leading edge on an axis is
// p * (box - drawn * zoom), with p the position as a 0-1 fraction.
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** The position fraction that moves the image by `delta` px from where `p` put it. */
export function dragPosition(
	p: number,
	delta: number,
	box: number,
	drawn: number,
	zoom: number,
): number {
	const span = box - drawn * zoom;
	return Math.abs(span) < 0.5 ? p : clamp01(p + delta / span);
}

/** The position fraction that keeps the image point under `at` (px in the box) still across a zoom change. */
export function zoomPosition(
	p: number,
	at: number,
	box: number,
	drawn: number,
	from: number,
	to: number,
): number {
	const span = box - drawn * to;
	if (Math.abs(span) < 0.5 || drawn <= 0) {
		return p;
	}
	const edge = p * (box - drawn * from);
	const along = (at - edge) / (drawn * from);
	return clamp01((at - along * drawn * to) / span);
}

/** Canvas px per CSS px: a wide blur keeps no fine detail, so it is baked at a lower resolution. */
export function bakeScale(blur: number, dpr: number): number {
	return Math.min(Math.max(dpr, 1), Math.max(0.25, 8 / blur));
}
