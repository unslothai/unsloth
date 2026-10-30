// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type ChatPageModule = typeof import("./chat-page");

// One shared promise so preload, idle prefetch and mount share a single fetch.
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
        chatPageModule = null;
        throw error;
      },
    );
  }
  return chatPageModule;
}

export function loadedChatPage(): ChatPageModule["ChatPage"] | null {
  return loadedModule?.ChatPage ?? null;
}

/** A failure is left for the mount to surface. */
export function prefetchChatPage(): void {
  loadChatPage().catch(() => undefined);
}
