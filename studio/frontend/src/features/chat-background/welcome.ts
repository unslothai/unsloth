// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { useEffect, useSyncExternalStore } from "react";

// Counted, not a flag: Compare mounts one welcome per pane.
let shown = 0;
const listeners = new Set<() => void>();
function emit() {
	for (const listener of listeners) listener();
}
function subscribe(listener: () => void) {
	listeners.add(listener);
	return () => {
		listeners.delete(listener);
	};
}

export function useMarkChatWelcome(): void {
	useEffect(() => {
		shown++;
		emit();
		return () => {
			shown--;
			emit();
		};
	}, []);
}

export function useChatWelcomeShown(): boolean {
	return useSyncExternalStore(
		subscribe,
		() => shown > 0,
		() => false,
	);
}
