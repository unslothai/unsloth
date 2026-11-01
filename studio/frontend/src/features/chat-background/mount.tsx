// SPDX-License-Identifier: AGPL-3.0-only
import { lazy, Suspense } from "react";
import { useMaterialStore } from "./material";
import { useChatBackground } from "./storage";
const Renderer = lazy(() =>
	import("./background").then((module) => ({ default: module.ChatBackground })),
);
const Backdrop = lazy(() =>
	import("./backdrop").then((module) => ({ default: module.GlassBackdrop })),
);
/** `backdrop`: the app's own instance, which shows the glass backdrop when glass is on without an image. */
export function ChatBackground({
	active = true,
	backdrop = false,
}: { active?: boolean; backdrop?: boolean }) {
	const { value, ready } = useChatBackground();
	const material = useMaterialStore((s) => s.material);
	if (!active) return null;
	if (!value)
		return backdrop && ready && material !== "solid" ? (
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
