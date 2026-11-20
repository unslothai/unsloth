// SPDX-License-Identifier: AGPL-3.0-only
export function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
	return new Promise((resolve, reject) =>
		canvas.toBlob(
			(blob) => (blob ? resolve(blob) : reject(new Error("decode"))),
			"image/png",
		),
	);
}
/** decode() may reject while the image still loads (the spec allows it): only a failed load rejects. */
export async function loadImage(image: HTMLImageElement, url: string): Promise<void> {
	const loaded = new Promise<void>((resolve, reject) => {
		image.addEventListener("load", () => resolve(), { once: true });
		image.addEventListener("error", () => reject(new Error("decode")), { once: true });
	});
	image.src = url;
	try {
		await image.decode();
	} catch {
		if (!(image.complete && image.naturalWidth > 0)) await loaded;
	}
}
export async function decodeImage(blob: Blob): Promise<{
	source: CanvasImageSource;
	width: number;
	height: number;
	close: () => void;
}> {
	if (typeof createImageBitmap === "function") {
		const image = await createImageBitmap(blob);
		return {
			source: image,
			width: image.width,
			height: image.height,
			close: () => image.close(),
		};
	}
	const url = URL.createObjectURL(blob),
		image = new Image();
	try {
		await loadImage(image, url);
		return {
			source: image,
			width: image.naturalWidth,
			height: image.naturalHeight,
			close: () => URL.revokeObjectURL(url),
		};
	} catch (error) {
		URL.revokeObjectURL(url);
		throw error;
	}
}
