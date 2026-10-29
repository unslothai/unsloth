// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type ChatPageModule = typeof import("./chat-page");

// ChatPage is loaded on first use so other screens start without it. One promise for every caller,
// so the startup preload, the idle prefetch and the mount share a single fetch.
let chatPageModule: Promise<ChatPageModule> | null = null;
let loadedModule: ChatPageModule | null = null;

export function loadChatPage(): Promise<ChatPageModule> {
  if (!chatPageModule) {
    chatPageModule = import("./chat-page").then(
      (module) => {
        loadedModule = module;
        return module;
      },
      (error: unknown) => {
        // A failed fetch must not stick: the next caller tries again.
        chatPageModule = null;
        throw error;
      },
    );
  }
  return chatPageModule;
}

/** The page once its module has arrived, so a mount after that renders it without suspending. */
export function loadedChatPage(): ChatPageModule["ChatPage"] | null {
  return loadedModule?.ChatPage ?? null;
}

/** Starts the fetch early (after the first screen, on hover or focus of a Chat link); a failure is left for the mount. */
export function prefetchChatPage(): void {
  loadChatPage().catch(() => undefined);
}
