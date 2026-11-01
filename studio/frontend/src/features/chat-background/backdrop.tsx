// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type CSSProperties, useEffect, useRef } from "react";
import { ROOT_READY, ROOT_WORKSPACE_READY, markRoot } from "./background";

const BACKDROP_SAFE = "0.55";
// Glass over the plain page, with no image: light mode tunes its menus for that.
const ROOT_PLAIN = "data-glass-plain";

export function GlassBackdrop() {
	const host = useRef<HTMLDivElement>(null);
	useEffect(() => {
		const parent = host.current?.parentElement;
		if (!parent?.hasAttribute("data-wallpaper-workspace")) return;
		parent.style.setProperty("--wallpaper-safe", BACKDROP_SAFE);
		const unmark = markRoot([ROOT_READY, ROOT_WORKSPACE_READY, ROOT_PLAIN]);
		return () => {
			unmark();
			parent.style.removeProperty("--wallpaper-safe");
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
		</div>
	);
}
