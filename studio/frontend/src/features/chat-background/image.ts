// SPDX-License-Identifier: AGPL-3.0-only
import { normalizedSize, validUpload } from "./model.ts";

export function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
	return new Promise((resolve, reject) =>
		canvas.toBlob(
			(blob) => (blob ? resolve(blob) : reject(new Error("decode"))),
			"image/png",
		),
	);
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
	image.src = url;
	try {
		await image.decode();
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

/** Rejects with Error("file" | "dimensions" | "decode"), the reasons the upload UI shows. */
export async function prepareBackgroundImage(file: Blob): Promise<Blob> {
	if (!validUpload(file)) throw new Error("file");
	const decoded = await decodeImage(file);
	try {
		const [width, height] = normalizedSize(decoded.width, decoded.height);
		const canvas = document.createElement("canvas");
		canvas.width = width;
		canvas.height = height;
		const ctx = canvas.getContext("2d");
		if (!ctx) throw new Error("decode");
		ctx.drawImage(decoded.source, 0, 0, width, height);
		return await canvasBlob(canvas);
	} finally {
		decoded.close();
	}
}
