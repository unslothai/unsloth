// SPDX-License-Identifier: AGPL-3.0-only
import { lazy, Suspense } from "react";
import type { BackgroundScope } from "./model";
import { useMaterialStore } from "./material";
import { useChatBackground } from "./storage";
import { useChatWelcomeShown } from "./welcome";
function inScope(
	scope: BackgroundScope,
	route: "newChat" | "chat" | "page",
): boolean {
	if (scope === "all") return true;
	if (scope === "chats") return route !== "page";
	return route === "newChat";
}
const Renderer = lazy(() =>
	import("./background").then((module) => ({ default: module.ChatBackground })),
);
const Backdrop = lazy(() =>
	import("./backdrop").then((module) => ({ default: module.GlassBackdrop })),
);
export function ChatBackground({
	active = true,
	route,
	backdrop = false,
	image = true,
}: {
	active?: boolean;
	route?: "newChat" | "chat" | "page";
	backdrop?: boolean;
	image?: boolean;
}) {
	const { value, ready } = useChatBackground();
	const welcome = useChatWelcomeShown();
	const material = useMaterialStore((s) => s.material);
	if (!active) return null;
	const shown =
		value &&
		image &&
		(!route || inScope(value.scope, route === "chat" && welcome ? "newChat" : route));
	if (!shown)
		// Glass on the chat routes still needs a ready canvas when no image shows there.
		return backdrop && ready && material !== "solid" && material !== "macos" && route !== "page" ? (
			<Suspense fallback={null}>
				<Backdrop />
			</Suspense>
		) : null;
	return (
		<Suspense fallback={null}>
			<Renderer active={active} />
		</Suspense>
	);
}
