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

/** Canvas px per CSS px: a wide blur keeps no fine detail, so it is baked at a lower resolution. */
export function bakeScale(blur: number, dpr: number): number {
	return Math.min(Math.max(dpr, 1), Math.max(0.25, 8 / blur));
}
