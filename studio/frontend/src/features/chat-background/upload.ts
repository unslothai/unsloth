// SPDX-License-Identifier: AGPL-3.0-only
import { canvasBlob, prepareBackgroundImage } from "./image";
import { type BackgroundAdjustments, DEFAULT_ADJUSTMENTS } from "./model";
import {
	beginBackgroundUpload,
	failBackgroundUpload,
	finishBackgroundUpload,
} from "./storage";

/** Validates, re-encodes and stores an image as the background. Resolves false on failure. */
export async function uploadBackgroundImage(
	source: Blob,
	adjustments: BackgroundAdjustments,
): Promise<boolean> {
	const ticket = beginBackgroundUpload();
	try {
		const blob = await prepareBackgroundImage(source);
		return await finishBackgroundUpload(ticket, { ...adjustments, blob });
	} catch (error) {
		const reason = error instanceof Error ? error.message : "";
		failBackgroundUpload(
			ticket,
			reason === "file" || reason === "dimensions" ? reason : "decode",
		);
		return false;
	}
}

/** Stores a painted canvas (an onboarding wallpaper preset) as the background. */
export async function setBackgroundCanvas(
	canvas: HTMLCanvasElement,
): Promise<void> {
	const ticket = beginBackgroundUpload();
	try {
		await finishBackgroundUpload(ticket, {
			...DEFAULT_ADJUSTMENTS,
			blob: await canvasBlob(canvas),
		});
	} catch {
		failBackgroundUpload(ticket, "storage");
	}
}
