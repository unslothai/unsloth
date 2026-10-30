// SPDX-License-Identifier: AGPL-3.0-only
import { canvasBlob, decodeImage } from "./image";
import type { BackgroundEffect } from "./model";

type Job = { promise: Promise<Blob>; release: () => void };
const cache = new WeakMap<Blob, Map<string, Blob>>();
const jobs = new WeakMap<
	Blob,
	Map<string, { promise: Promise<Blob>; users: number; cancel: () => void }>
>();
/** Shared by chat and previews: the last release cancels pixel work; blobs are cached, never URLs. */
export function acquireRender(
	blob: Blob,
	effect: BackgroundEffect,
	light: boolean,
): Job {
	if (effect === "None")
		return { promise: Promise.resolve(blob), release: () => {} };
	const key = `${effect}:${effect === "Dither" ? false : light}`;
	const cached = cache.get(blob)?.get(key);
	if (cached) return { promise: Promise.resolve(cached), release: () => {} };
	const pending = jobs.get(blob) ?? new Map();
	jobs.set(blob, pending);
	let entry = pending.get(key);
	if (!entry) {
		let cancelled = false,
			worker: Worker | undefined,
			rejectJob: (reason: Error) => void = () => {};
		const promise = new Promise<Blob>((resolve, reject) => {
			rejectJob = reject;
			void (async () => {
				const decoded = await decodeImage(blob);
				const canvas = document.createElement("canvas");
				canvas.width = decoded.width;
				canvas.height = decoded.height;
				const ctx = canvas.getContext("2d", { willReadFrequently: true });
				try {
					if (cancelled) return;
					if (!ctx) throw new Error("effect");
					ctx.drawImage(decoded.source, 0, 0);
				} finally {
					decoded.close();
				}
				if (cancelled) return;
				const pixels = ctx!.getImageData(
					0,
					0,
					canvas.width,
					canvas.height,
				).data;
				worker = new Worker(new URL("./worker.ts", import.meta.url), {
					type: "module",
				});
				worker.onerror = () => reject(new Error("effect"));
				worker.onmessage = ({
					data,
				}: MessageEvent<{ pixels?: Uint8ClampedArray; error?: boolean }>) => {
					worker?.terminate();
					if (cancelled) return;
					if (!data.pixels) {
						reject(new Error("effect"));
						return;
					}
					ctx!.putImageData(
						new ImageData(
							new Uint8ClampedArray(data.pixels),
							canvas.width,
							canvas.height,
						),
						0,
						0,
					);
					void canvasBlob(canvas).then(resolve, reject);
				};
				worker.postMessage(
					{ pixels, width: canvas.width, height: canvas.height, effect, light },
					[pixels.buffer],
				);
			})().catch(reject);
		})
			.then((result) => {
				if (!cancelled) {
					const images = cache.get(blob) ?? new Map();
					images.set(key, result);
					if (images.size > 4) images.delete(images.keys().next().value!);
					cache.set(blob, images);
				}
				return result;
			})
			.finally(() => {
				worker?.terminate();
				if (pending.get(key)?.promise === promise) pending.delete(key);
			});
		entry = {
			promise,
			users: 0,
			cancel: () => {
				cancelled = true;
				worker?.terminate();
				rejectJob(new Error("cancelled"));
			},
		};
		pending.set(key, entry);
	}
	entry.users++;
	let released = false;
	return {
		promise: entry.promise,
		release: () => {
			if (released) return;
			released = true;
			if (--entry!.users === 0 && pending.get(key) === entry) {
				pending.delete(key);
				entry!.cancel();
			}
		},
	};
}
