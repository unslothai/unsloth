// SPDX-License-Identifier: AGPL-3.0-only
import {
	useEffect,
	useEffectEvent,
	useRef,
	useState,
	type CSSProperties,
} from "react";
import { useChatBackground, setBackgroundError } from "./storage";
import { acquireRender } from "./render";
import { safeArtworkOpacity, type RGB } from "./contrast";
import type { ChatBackgroundValue } from "./model";

type Frame = { url: string; value: ChatBackgroundValue };
// The window titlebar sits outside the wallpaper's subtree, so a ready wallpaper marks the root
// rather than having CSS ask html:has(), which re-matches on every DOM change.
const ROOT_READY = "data-chat-wallpaper-ready";
const ROOT_WORKSPACE_READY = "data-chat-wallpaper-workspace-ready";
const rootMarks = new Map<string, number>();
function markRoot(names: string[]): () => void {
	const root = document.documentElement;
	for (const name of names) {
		rootMarks.set(name, (rootMarks.get(name) ?? 0) + 1);
		root.setAttribute(name, "");
	}
	return () => {
		for (const name of names) {
			const left = (rootMarks.get(name) ?? 1) - 1;
			rootMarks.set(name, left);
			if (left <= 0) root.removeAttribute(name);
		}
	};
}
export function ChatBackground({ active = true }: { active?: boolean }) {
	const { value } = useChatBackground();
	const enabled = Boolean(value);
	const blob = value?.blob;
	const effect = value?.effect;
	const themeDependent = effect !== "None" && effect !== "Dither";
	const host = useRef<HTMLDivElement>(null);
	const urls = useRef(new Set<string>());
	const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
	const latest = useRef<Frame | null>(null);
	const [frames, setFrames] = useState<{
		current: Frame | null;
		previous: Frame | null;
	}>({ current: null, previous: null });
	const [skin, setSkin] = useState({ light: false, reduced: true, safe: 0 });
	useEffect(() => {
		if (!enabled || !active) return;
		const html = document.documentElement,
			media = window.matchMedia("(prefers-reduced-motion: reduce)");
		let frame = 0;
		const update = () => {
			const el = host.current;
			if (!el) return;
			const probe = document.createElement("span");
			probe.style.position = "absolute";
			probe.style.visibility = "hidden";
			el.append(probe);
			const ctx = document
				.createElement("canvas")
				.getContext("2d", { willReadFrequently: true })!;
			const rgb = (token: string): RGB => {
				probe.style.color = `var(${token})`;
				ctx.clearRect(0, 0, 1, 1);
				ctx.fillStyle = getComputedStyle(probe).color;
				ctx.fillRect(0, 0, 1, 1);
				const p = ctx.getImageData(0, 0, 1, 1).data;
				return [p[0], p[1], p[2]];
			};
			const safe = safeArtworkOpacity(rgb("--background"), [
				rgb("--foreground"),
				rgb("--muted-foreground"),
			]);
			probe.remove();
			setSkin({
				light: !html.classList.contains("dark"),
				reduced:
					html.classList.contains("force-reduced-motion") ||
					(!html.classList.contains("force-motion") && media.matches),
				safe,
			});
		};
		const schedule = () => {
			cancelAnimationFrame(frame);
			frame = requestAnimationFrame(update);
		};
		const observer = new MutationObserver(schedule);
		observer.observe(html, {
			attributes: true,
			attributeFilter: ["style", "class", "data-palette"],
		});
		media.addEventListener("change", schedule);
		update();
		return () => {
			observer.disconnect();
			media.removeEventListener("change", schedule);
			cancelAnimationFrame(frame);
		};
	}, [enabled, active]);
	useEffect(() => {
		const parent = host.current?.parentElement;
		if (!parent || !enabled || !active) return;
		parent.style.setProperty("--wallpaper-safe", String(skin.safe));
		return () => {
			parent.style.removeProperty("--wallpaper-safe");
		};
	}, [skin.safe, enabled, active]);
	const adoptFrame = useEffectEvent((url: string) => {
		urls.current.add(url);
		clearTimeout(timer.current);
		const previous = skin.reduced ? null : latest.current;
		const current = { url, value: value! };
		latest.current = current;
		setFrames({ current, previous });
		timer.current = setTimeout(
			() =>
				setFrames((s) =>
					s.current === current ? { current, previous: null } : s,
				),
			skin.reduced ? 0 : 180,
		);
	});
	const renderLight = themeDependent && skin.light;
	useEffect(() => {
		if (!active || !blob || !effect) {
			latest.current = null;
			let cancelled = false;
			queueMicrotask(() => {
				if (!cancelled) setFrames({ current: null, previous: null });
			});
			return () => {
				cancelled = true;
			};
		}
		let cancelled = false;
		const job = acquireRender(blob, effect, renderLight);
		void job.promise
			.then(async (blob) => {
				if (cancelled) return;
				const url = URL.createObjectURL(blob);
				const image = new Image();
				image.src = url;
				try {
					await image.decode();
				} catch {
					urls.current.delete(url);
					URL.revokeObjectURL(url);
					throw new Error("decode");
				}
				if (cancelled) {
					urls.current.delete(url);
					URL.revokeObjectURL(url);
					return;
				}
				adoptFrame(url);
			})
			.catch(() => {
				if (!cancelled) setBackgroundError("effect");
			});
		return () => {
			cancelled = true;
			job.release();
		};
		// Visual controls are CSS only, never inputs to image processing.
	}, [active, blob, effect, renderLight]);
	useEffect(() => {
		const alive = new Set([frames.current?.url, frames.previous?.url]);
		for (const url of urls.current)
			if (!alive.has(url)) {
				URL.revokeObjectURL(url);
				urls.current.delete(url);
			}
	}, [frames]);
	useEffect(
		() => () => {
			clearTimeout(timer.current);
			for (const url of urls.current) URL.revokeObjectURL(url);
			urls.current.clear();
		},
		[],
	);
	const shownReady = Boolean(value && active && frames.current);
	useEffect(() => {
		// Only the app's own wallpaper drives the titlebar; a settings preview never marks the root.
		const parent = host.current?.parentElement;
		if (!shownReady || !parent) return;
		const workspace = parent.hasAttribute("data-wallpaper-workspace");
		if (!workspace && !parent.hasAttribute("data-chat-content-root")) return;
		return markRoot(workspace ? [ROOT_READY, ROOT_WORKSPACE_READY] : [ROOT_READY]);
	}, [shownReady]);
	if (!value || !active) return null;
	return (
		<div
			ref={host}
			aria-hidden="true"
			className="chat-wallpaper"
			data-ready={Boolean(frames.current) || undefined}
			data-reduced={skin.reduced || undefined}
			data-outline={value.outline}
			style={{ "--wallpaper-safe": skin.safe } as CSSProperties}
		>
			{[frames.previous, frames.current].map((frame, i) => {
				if (!frame) return null;
				const adjustment =
					frame.value.blob === value.blob ? value : frame.value;
				return (
					<div
						key={frame.url}
						className="chat-wallpaper-frame"
						data-effect={frame.value.effect}
						data-departing={i === 0 || undefined}
						data-entering={(i === 1 && Boolean(frames.previous)) || undefined}
						style={
							{
								"--wallpaper-visibility": adjustment.visibility / 100,
							} as CSSProperties
						}
					>
						<div className="chat-wallpaper-image" data-fade={adjustment.fade}>
							<div
								className="chat-wallpaper-artwork"
								style={{
									backgroundImage: `url("${frame.url}")`,
									backgroundPosition: adjustment.position,
									backgroundSize:
										adjustment.sizing === "original"
											? "auto"
											: adjustment.sizing === "stretch"
												? "100% 100%"
												: adjustment.sizing,
									transform: `scale(${adjustment.zoom / 100})`,
									transformOrigin: adjustment.position,
									filter: adjustment.blur
										? `blur(${adjustment.blur}px)`
										: undefined,
								}}
							/>
						</div>
					</div>
				);
			})}
		</div>
	);
}
