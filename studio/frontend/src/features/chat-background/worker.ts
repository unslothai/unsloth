// SPDX-License-Identifier: AGPL-3.0-only
import { renderEffect } from "./effects";
import type { BackgroundEffect } from "./model";
// Main thread decodes; only the pixel transform runs here (no OffscreenCanvas needed).
self.onmessage = ({
	data,
}: MessageEvent<{
	pixels: Uint8ClampedArray;
	width: number;
	height: number;
	effect: BackgroundEffect;
	light: boolean;
}>) => {
	try {
		const pixels = renderEffect(
			data.pixels,
			data.width,
			data.height,
			data.effect,
			data.light,
		);
		self.postMessage({ pixels }, { transfer: [pixels.buffer] });
	} catch {
		self.postMessage({ error: true });
	}
};
