// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { subscribeChatHistoryUpdated } from "@/features/chat/api/chat-api";
import { useEffect } from "react";
import { Portrait } from "./components/portrait";
import { useCharactersStore } from "./store";

let subscribed = false;
let loaded = false;
let timer: number | null = null;

function refreshSoon(): void {
  if (timer !== null) window.clearTimeout(timer);
  timer = window.setTimeout(() => {
    timer = null;
    const store = useCharactersStore.getState();
    void store.refreshIndex().catch(() => undefined);
    if (store.status === "idle") void store.refresh().catch(() => undefined);
  }, 400);
}

/** Loads the chat index once per sign-in and keeps it fresh as chats change, here or in another tab. */
function watchChatIndex(): void {
  if (typeof window === "undefined") return;
  if (!subscribed) {
    subscribed = true;
    subscribeChatHistoryUpdated((detail) => {
      if (!detail.coalesce) refreshSoon();
    });
    window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
      loaded = false;
    });
  }
  if (!loaded) {
    loaded = true;
    refreshSoon();
  }
}

/** A character chat's portrait at the start of its sidebar row. */
export function CharacterChatMark({ threadId }: { threadId: string }) {
  const ids = useCharactersStore((s) => s.chatIndex[threadId]);
  const first = useCharactersStore((s) => (ids?.length ? s.characters[ids[0]] : undefined));
  useEffect(watchChatIndex, []);
  if (!ids || !first) return null;
  return (
    <span className="relative -ml-1 flex shrink-0" aria-hidden={true} data-testid="character-chat-mark">
      <Portrait
        id={first.id}
        name={first.name}
        hasImage={first.hasAvatar}
        rev={first.avatarRev}
        className="size-4.5 rounded-full text-[0.5rem]"
      />
      {ids.length > 1 ? (
        <span className="absolute -bottom-1 -right-1.5 flex h-3 min-w-3 items-center justify-center rounded-full bg-foreground px-[3px] text-[length:calc(var(--text-ui-10)*0.8)] font-semibold leading-none text-background ring-[1.5px] ring-sidebar">
          {ids.length}
        </span>
      ) : null}
    </span>
  );
}
