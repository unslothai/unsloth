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
export function ChatBackground({
	active = true,
	backdrop = false,
	image = true,
}: { active?: boolean; backdrop?: boolean; image?: boolean }) {
	const { value, ready } = useChatBackground();
	const material = useMaterialStore((s) => s.material);
	if (!active) return null;
	if (!(value && image))
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
