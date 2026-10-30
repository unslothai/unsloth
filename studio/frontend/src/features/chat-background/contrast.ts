// SPDX-License-Identifier: AGPL-3.0-only
export type RGB = readonly [number, number, number];
export function luminance(rgb: RGB): number {
	const linear = rgb.map((v) => {
		const c = v / 255;
		return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
	});
	return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
}
export function contrast(a: RGB, b: RGB): number {
	const x = luminance(a),
		y = luminance(b);
	return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}
/** Largest artwork alpha safe for any image; keeps a deliberately chosen theme contrast below 4.5. */
export function safeArtworkOpacity(
	background: RGB,
	foregrounds: RGB[],
): number {
	const targets = foregrounds.map((f) =>
		Math.min(4.5, contrast(background, f)),
	);
	const passes = (alpha: number) => {
		const low = luminance(
			background.map((c) => c * (1 - alpha)) as unknown as RGB,
		);
		const high = luminance(
			background.map((c) => c * (1 - alpha) + 255 * alpha) as unknown as RGB,
		);
		return foregrounds.every((f, i) => {
			const l = luminance(f);
			const nearest = Math.max(low, Math.min(high, l));
			return (
				(Math.max(l, nearest) + 0.05) / (Math.min(l, nearest) + 0.05) >=
				targets[i] - 1e-7
			);
		});
	};
	let lo = 0,
		hi = 1;
	for (let i = 0; i < 24; i++) {
		const mid = (lo + hi) / 2;
		if (passes(mid)) lo = mid;
		else hi = mid;
	}
	return lo;
}
