// SPDX-License-Identifier: AGPL-3.0-only
import { lazy, Suspense } from "react";
import { useAppliedMaterial } from "./material";
import { inScope } from "./model";
import { useChatBackground } from "./storage";
import { useChatWelcomeShown } from "./welcome";
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
	const material = useAppliedMaterial();
	if (!active) return null;
	const shown =
		value &&
		image &&
		(!route || inScope(value.scope, route === "chat" && welcome ? "newChat" : route));
	if (!shown)
		// Glass still needs a ready canvas on any route where no image shows.
		return backdrop && ready && material !== "solid" && material !== "macos" ? (
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
