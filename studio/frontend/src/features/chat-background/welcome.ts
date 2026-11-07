// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { useEffect, useLayoutEffect, useSyncExternalStore } from "react";

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

/*
 * Root flags for the wallpaper CSS. A descendant :has() on the workspace re-walks the whole
 * thread on every keystroke, so the welcome and compare states are mirrored onto <html>
 * in the same commit as the DOM they describe.
 */
export const ROOT_WELCOME = "data-chat-welcome";
export const ROOT_COMPARE = "data-chat-compare";
const rootCounts = new Map<string, number>();
function useRootFlag(name: string): void {
	useLayoutEffect(() => {
		const root = document.documentElement;
		rootCounts.set(name, (rootCounts.get(name) ?? 0) + 1);
		root.setAttribute(name, "");
		return () => {
			const left = Math.max(0, (rootCounts.get(name) ?? 1) - 1);
			rootCounts.set(name, left);
			if (left === 0) root.removeAttribute(name);
		};
	}, [name]);
}

export function useMarkChatWelcome(): void {
	useRootFlag(ROOT_WELCOME);
	useEffect(() => {
		shown++;
		emit();
		return () => {
			shown--;
			emit();
		};
	}, []);
}

export function useMarkChatCompare(): void {
	useRootFlag(ROOT_COMPARE);
}

export function useChatWelcomeShown(): boolean {
	return useSyncExternalStore(
		subscribe,
		() => shown > 0,
		() => false,
	);
}
