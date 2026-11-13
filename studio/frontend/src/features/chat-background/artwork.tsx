// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { useEffect, useRef, useState, type CSSProperties } from "react";
import type { ImageSizing } from "./model";
import { bakeScale, imageBox } from "./artwork-geometry";

const BACKGROUND_SIZE: Record<ImageSizing, string> = {
	cover: "cover",
	contain: "contain",
	width: "100% auto",
	height: "auto 100%",
	stretch: "100% 100%",
	original: "auto",
};
// A resize or a slider drag keeps the live filter until the inputs hold this long.
const SETTLE_MS = 160;

const canvasFilter =
	typeof CanvasRenderingContext2D !== "undefined" &&
	"filter" in CanvasRenderingContext2D.prototype;

/** The box exactly as CSS paints it (clipped to the box), then blurred the way `filter: blur()` blurs it. */
async function bake(
	url: string,
	box: { w: number; h: number },
	sizing: ImageSizing,
	position: string,
	blur: number,
): Promise<Blob | null> {
	const image = new Image();
	image.src = url;
	await image.decode();
	const k = bakeScale(blur, window.devicePixelRatio || 1);
	const cw = Math.max(1, Math.round(box.w * k)),
		ch = Math.max(1, Math.round(box.h * k));
	const at = imageBox(box.w, box.h, image.naturalWidth, image.naturalHeight, sizing, position);
	const plain = document.createElement("canvas");
	plain.width = cw;
	plain.height = ch;
	const pctx = plain.getContext("2d");
	if (!pctx) return null;
	pctx.imageSmoothingQuality = "high";
	pctx.drawImage(image, at.x * k, at.y * k, at.w * k, at.h * k);
	const out = document.createElement("canvas");
	out.width = cw;
	out.height = ch;
	const octx = out.getContext("2d");
	if (!octx) return null;
	octx.imageSmoothingQuality = "high";
	if (canvasFilter) {
		octx.filter = `blur(${blur * k}px)`;
		octx.drawImage(plain, 0, 0);
	} else {
		// Engines without a canvas filter (older WebKit): a downscale and smooth upscale reads as the same blur.
		const f = Math.min(1, 1.5 / (blur * k));
		const small = document.createElement("canvas");
		small.width = Math.max(1, Math.round(cw * f));
		small.height = Math.max(1, Math.round(ch * f));
		const sctx = small.getContext("2d");
		if (!sctx) return null;
		sctx.imageSmoothingQuality = "high";
		sctx.drawImage(plain, 0, 0, small.width, small.height);
		octx.drawImage(small, 0, 0, cw, ch);
	}
	return new Promise((resolve) => out.toBlob(resolve, "image/png"));
}

type Props = {
	url: string;
	sizing: ImageSizing;
	position: string;
	zoom: number;
	blur: number;
	/** Departing frames only fade out: they keep the live filter rather than bake for 180 ms. */
	bakeable: boolean;
};

/**
 * The wallpaper artwork. A live `filter: blur()` behind a scrolling, see-through thread is re-run on
 * every scroll frame (about 4-6x the scroll cost of no wallpaper); a blurred bitmap baked once costs
 * nothing per frame. The live filter stays only while a bake for the current inputs is pending.
 */
export function WallpaperArtwork({ url, sizing, position, zoom, blur, bakeable }: Props) {
	const el = useRef<HTMLDivElement>(null);
	const [size, setSize] = useState<{ w: number; h: number } | null>(null);
	const [baked, setBaked] = useState<{ key: string; url: string } | null>(null);
	const bakedUrl = useRef<string | null>(null);
	const want = bakeable && blur > 0;
	useEffect(() => {
		const node = el.current;
		if (!node || !want || typeof ResizeObserver === "undefined") return;
		let timer: ReturnType<typeof setTimeout> | undefined;
		const read = () => {
			// The box before transform: scale(zoom), which CSS applies after the filter as well.
			const w = node.offsetWidth,
				h = node.offsetHeight;
			if (w > 0 && h > 0)
				setSize((s) => (s && s.w === w && s.h === h ? s : { w, h }));
		};
		const observer = new ResizeObserver(() => {
			clearTimeout(timer);
			timer = setTimeout(read, SETTLE_MS);
		});
		observer.observe(node);
		read();
		return () => {
			clearTimeout(timer);
			observer.disconnect();
		};
	}, [want]);
	const key =
		want && size
			? `${url}|${sizing}|${position}|${blur}|${size.w}x${size.h}|${window.devicePixelRatio || 1}`
			: null;
	useEffect(() => {
		if (!key || !size) return;
		let cancelled = false;
		const timer = setTimeout(() => {
			bake(url, size, sizing, position, blur)
				.then((blob) => {
					if (cancelled || !blob) return;
					const next = URL.createObjectURL(blob);
					if (bakedUrl.current) URL.revokeObjectURL(bakedUrl.current);
					bakedUrl.current = next;
					setBaked({ key, url: next });
				})
				.catch(() => undefined);
		}, SETTLE_MS);
		return () => {
			cancelled = true;
			clearTimeout(timer);
		};
	}, [key, size, url, sizing, position, blur]);
	useEffect(
		() => () => {
			if (bakedUrl.current) URL.revokeObjectURL(bakedUrl.current);
			bakedUrl.current = null;
		},
		[],
	);
	const ready = baked !== null && baked.key === key;
	const style: CSSProperties = ready
		? {
				backgroundImage: `url("${baked.url}")`,
				backgroundPosition: "0 0",
				backgroundSize: "100% 100%",
				transform: `scale(${zoom / 100})`,
				transformOrigin: position,
			}
		: {
				backgroundImage: `url("${url}")`,
				backgroundPosition: position,
				backgroundSize: BACKGROUND_SIZE[sizing],
				transform: `scale(${zoom / 100})`,
				transformOrigin: position,
				filter: blur ? `blur(${blur}px)` : undefined,
			};
	return (
		<div
			ref={el}
			className="chat-wallpaper-artwork"
			data-baked={ready || undefined}
			style={style}
		/>
	);
}
