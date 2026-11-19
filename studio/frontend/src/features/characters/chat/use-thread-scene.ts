// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useState } from "react";
import { useUserProfileStore } from "@/features/profile";
import { castOf, resolvePersona } from "../chat-runtime";
import { isCharacterChat, useCharactersStore } from "../store";
import type { Character, ChatScene, ResolvedPersona } from "../types";

export interface ThreadScene {
  scene: ChatScene;
  cast: Character[];
  persona: ResolvedPersona;
  group: boolean;
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
  // The persona falls back to the profile name, so follow it too.
  const profileName = useUserProfileStore((s) => `${s.nickname}|${s.displayName}`);
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
    if (!scene) return null;
    const cast = castOf(scene, characters);
    if (cast.length === 0) return null;
    return { scene, cast, persona: resolvePersona(scene, personas), group: cast.length > 1 };
    // profileName is read through resolvePersona's store access.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, characters, personas, profileName]);
}
