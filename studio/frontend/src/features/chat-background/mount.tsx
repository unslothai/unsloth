// SPDX-License-Identifier: AGPL-3.0-only
import { lazy, Suspense } from "react";
import type { BackgroundScope } from "./model";
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
/** `route` is where the caller sits; the user's scope decides whether the wallpaper shows there. */
export function ChatBackground({
	active = true,
	route,
}: { active?: boolean; route?: "newChat" | "chat" | "page" }) {
	const { value } = useChatBackground();
	const welcome = useChatWelcomeShown();
	if (!active || !value) return null;
	if (route && !inScope(value.scope, route === "chat" && welcome ? "newChat" : route))
		return null;
	return (
		<Suspense fallback={null}>
			<Renderer active={active} />
		</Suspense>
	);
}
