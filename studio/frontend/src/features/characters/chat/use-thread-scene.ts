// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useState } from "react";
import { isCharacterChat } from "../chat-gate";
import { resolvePersona, useAccountName } from "../persona";
import { castOf } from "../prompt/round";
import { useCharactersStore } from "../store";
import type { Character, ChatScene, ResolvedPersona } from "../types";

export interface ThreadScene {
  threadId: string;
  scene: ChatScene;
  cast: Character[];
  persona: ResolvedPersona;
  isGroup: boolean;
}

const RETRIES = 3;

/** The character chat on screen, or null for an ordinary chat (or one whose characters were all deleted). */
export function useThreadScene(threadId: string | null | undefined): ThreadScene | null {
  const scene = useCharactersStore((s) => (threadId ? s.scenes[threadId] : undefined));
  const indexed = useCharactersStore((s) => Boolean(threadId && threadId in s.chatIndex));
  const indexLoaded = useCharactersStore((s) => s.indexLoaded);
  const maybeCharacterChat = useCharactersStore(isCharacterChat(threadId ?? null, false));
  const characters = useCharactersStore((s) => s.characters);
  const personas = useCharactersStore((s) => s.personas);
  const status = useCharactersStore((s) => s.status);
  const personasStatus = useCharactersStore((s) => s.personasStatus);
  // The persona falls back to the account's name, so follow it too.
  const accountName = useAccountName();
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!indexLoaded) void useCharactersStore.getState().refreshIndex().catch(() => undefined);
  }, [indexLoaded]);
  useEffect(() => {
    if (!threadId || !maybeCharacterChat) return;
    const store = useCharactersStore.getState();
    // A chat that turned out ordinary is asked again only if the index later lists it.
    const known = threadId in store.scenes;
    if (known && !(indexed && store.scenes[threadId] === null)) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    store.loadScene(threadId, known).catch(() => {
      if (live && attempt < RETRIES) timer = setTimeout(() => setAttempt((n) => n + 1), 2000 * (attempt + 1));
    });
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [threadId, indexed, maybeCharacterChat, attempt]);
  useEffect(() => setAttempt(0), [threadId]);
  useEffect(() => {
    if (!scene) return;
    const store = useCharactersStore.getState();
    if (status === "idle") void store.refresh().catch(() => undefined);
    if (personasStatus === "idle") void store.refreshPersonas().catch(() => undefined);
  }, [scene, status, personasStatus]);

  return useMemo(() => {
    if (!scene || !threadId) return null;
    const cast = castOf(scene, characters);
    if (cast.length === 0) return null;
    return { threadId, scene, cast, persona: resolvePersona(scene, personas, accountName), isGroup: cast.length > 1 };
  }, [threadId, scene, characters, personas, accountName]);
}

/** Whether this thread is a character chat, for UI outside CharacterStage. */
export function useIsCharacterChat(threadId: string | null | undefined): boolean {
  return useThreadScene(threadId) !== null;
}
