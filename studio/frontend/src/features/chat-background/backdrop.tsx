// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type CSSProperties, useEffect, useRef, useState } from "react";
import { ROOT_READY, ROOT_WORKSPACE_READY, markRoot } from "./background";
import { ComposerGlassLenses } from "./lens";

/**
 * Glass without a background image: a soft built-in backdrop in the wallpaper's place, so every glass
 * surface (and the lens) has colour behind it. It is a ready .chat-wallpaper, so the glass rules apply
 * exactly as they do over an image.
 */
const BACKDROP_SAFE = "0.55";

export function GlassBackdrop() {
	const host = useRef<HTMLDivElement>(null);
	const [appWallpaper, setAppWallpaper] = useState<HTMLDivElement | null>(null);
	useEffect(() => {
		const parent = host.current?.parentElement;
		if (!parent?.hasAttribute("data-wallpaper-workspace")) return;
		setAppWallpaper(host.current);
		// Soft and low in contrast, so reading surfaces can let more of it through than of an image.
		parent.style.setProperty("--wallpaper-safe", BACKDROP_SAFE);
		const unmark = markRoot([ROOT_READY, ROOT_WORKSPACE_READY]);
		return () => {
			unmark();
			parent.style.removeProperty("--wallpaper-safe");
			setAppWallpaper(null);
		};
	}, []);
	return (
		<div
			ref={host}
			aria-hidden="true"
			className="chat-wallpaper"
			data-ready=""
			data-outline="true"
			data-glass-backdrop=""
			style={{ "--wallpaper-safe": BACKDROP_SAFE } as CSSProperties}
		>
			<div className="chat-wallpaper-image" data-fade="false">
				<div className="chat-wallpaper-artwork glass-backdrop" />
			</div>
			<ComposerGlassLenses wallpaper={appWallpaper} />
		</div>
	);
}
