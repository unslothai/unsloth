// SPDX-License-Identifier: AGPL-3.0-only
import { lazy, Suspense } from "react";
import { useChatBackground } from "./storage";
const Renderer = lazy(() =>
	import("./background").then((module) => ({ default: module.ChatBackground })),
);
export function ChatBackground({
	active = true,
}: { active?: boolean }) {
	const { value } = useChatBackground();
	if (!active || !value) return null;
	return (
		<Suspense fallback={null}>
			<Renderer active={active} />
		</Suspense>
	);
}
