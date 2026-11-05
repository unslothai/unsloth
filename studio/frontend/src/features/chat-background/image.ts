// SPDX-License-Identifier: AGPL-3.0-only
import { imageHeaderSize, normalizedSize, validUpload } from "./model.ts";

// Some engines quietly fall back to PNG for an unsupported type, so check what came back.
export function canvasBlob(
	canvas: HTMLCanvasElement,
	type = "image/png",
	quality?: number,
): Promise<Blob> {
	return new Promise((resolve, reject) =>
		canvas.toBlob(
			(blob) =>
				blob?.type === type ? resolve(blob) : reject(new Error("decode")),
			type,
			quality,
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

/** Decodes an image and draws it again at `size`, e.g. to cap it or change its format. */
export async function redrawImage(
	blob: Blob,
	options: {
		size?: (width: number, height: number) => [number, number];
		type?: string;
		quality?: number;
		/** Painted first, so transparent pixels don't turn black in a JPEG. */
		fill?: string;
	} = {},
): Promise<Blob> {
	const decoded = await decodeImage(blob);
	try {
		const [width, height] = options.size
			? options.size(decoded.width, decoded.height)
			: [decoded.width, decoded.height];
		const canvas = document.createElement("canvas");
		canvas.width = width;
		canvas.height = height;
		const ctx = canvas.getContext("2d");
		if (!ctx) throw new Error("decode");
		if (options.fill) {
			ctx.fillStyle = options.fill;
			ctx.fillRect(0, 0, width, height);
		}
		ctx.drawImage(decoded.source, 0, 0, width, height);
		return await canvasBlob(canvas, options.type, options.quality);
	} finally {
		decoded.close();
	}
}

/** Rejects with Error("file" | "dimensions" | "decode"), the reasons the upload UI shows. */
export async function prepareBackgroundImage(file: Blob): Promise<Blob> {
	if (!validUpload(file)) throw new Error("file");
	// Check the size from the header first: decoding a huge image could exhaust memory before
	// normalizedSize ever sees it.
	const size = imageHeaderSize(new Uint8Array(await file.arrayBuffer()));
	if (!size) throw new Error("decode");
	normalizedSize(...size);
	return redrawImage(file, { size: normalizedSize });
}
