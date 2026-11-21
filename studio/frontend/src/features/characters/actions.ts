// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { notifyChatHistoryUpdated } from "@/features/chat/api/chat-api";
import { writeBranchHead } from "@/features/chat/utils/branch-head";
import { useT } from "@/i18n";
import { errorMessage } from "@/lib/error-message";
import { downloadBlobStreaming, isDownloadCancelled } from "@/lib/native-files";
import { openFilePicker } from "@/lib/open-file-picker";
import { toast } from "@/lib/toast";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useRef, useState } from "react";
import * as charactersApi from "./api";
import { resolvePersona } from "./persona";
import { applyMacros } from "./prompt/macros";
import { useCharactersStore } from "./store";
import type { Character, Persona } from "./types";

type T = ReturnType<typeof useT>;

/** Opens a new chat with these characters and returns its thread id. */
export async function startCharacterChat(
  characterIds: string[],
  options: { projectId?: string | null } = {},
): Promise<string> {
  const store = useCharactersStore.getState();
  // Not swallowed: unloaded personas would address the stored greetings to the account instead.
  if (store.personasStatus !== "ready") await store.refreshPersonas();
  const persona = resolvePersona(null, useCharactersStore.getState().personas);
  // Resolved here, where the user's clock and locale are, and stored once so history never rerolls.
  const { characters } = useCharactersStore.getState();
  const cast = [...new Set(characterIds)].map((id) => characters[id]).filter((c): c is Character => Boolean(c));
  const group = cast.map((c) => c.name);
  const now = new Date();
  const greetings = Object.fromEntries(
    cast.map((c) => [
      c.id,
      [c.firstMes, ...c.alternateGreetings].map((text) =>
        applyMacros(text, { char: c.name, user: persona.name || "User", group, now }),
      ),
    ]),
  );
  const created = await charactersApi.createCharacterChat({
    characterIds,
    personaId: persona.id,
    userName: persona.name,
    projectId: options.projectId ?? null,
    greetings,
  });
  writeBranchHead(created.threadId, created.headId);
  for (const id of characterIds) {
    const character = useCharactersStore.getState().characters[id];
    if (character) store.upsert({ ...character, lastChatAt: now.getTime(), chatCount: character.chatCount + 1 });
  }
  store.setScene(created.scene);
  notifyChatHistoryUpdated();
  return created.threadId;
}

/** Starts a chat and opens it; resolves false (after a toast) when it could not start. */
export function useOpenCharacterChat(): { open: (characterIds: string[]) => Promise<boolean>; starting: boolean } {
  const t = useT();
  const navigate = useNavigate();
  const [starting, setStarting] = useState(false);
  const busy = useRef(false);
  const open = useCallback(
    async (characterIds: string[]) => {
      if (busy.current) return false;
      busy.current = true;
      setStarting(true);
      try {
        const threadId = await startCharacterChat(characterIds);
        void navigate({ to: "/chat", search: { thread: threadId } });
        return true;
      } catch (error) {
        toast.error(t("characters.chat.startFailed"), { description: errorMessage(error) });
        return false;
      } finally {
        busy.current = false;
        setStarting(false);
      }
    },
    [navigate, t],
  );
  return { open, starting };
}

export const CARD_EXTENSIONS = ".png,.json,.charx";
const CARD_ACCEPT = `${CARD_EXTENSIONS},image/png,application/json`;

/** Opens the file chooser for character cards; call it from a click (needs user activation). */
export function pickCardFiles(t: T): void {
  openFilePicker(CARD_ACCEPT, (files) => void importCards(files, t));
}

export async function importCards(files: File[], t: T): Promise<Character[]> {
  const imported: Character[] = [];
  for (const file of files) {
    const toastId = toast.loading(t("characters.importing", { name: file.name }));
    try {
      const result = await charactersApi.importCharacterFile(file, file.name);
      useCharactersStore.getState().upsert(result.character);
      imported.push(result.character);
      if (result.warnings.length) {
        toast.warning(t("characters.importedWithWarnings", { name: result.character.name }), {
          id: toastId,
          description: result.warnings.join(" "),
        });
      } else {
        toast.success(t("characters.imported", { name: result.character.name }), { id: toastId });
      }
    } catch (error) {
      toast.error(t("characters.importFailed", { name: file.name }), {
        id: toastId,
        description: errorMessage(error),
      });
    }
  }
  if (imported.length > 1) toast.success(t("characters.importedMany", { count: imported.length }));
  return imported;
}

export async function exportCharacter(character: Character, format: "png" | "json", t: T): Promise<void> {
  try {
    const { blob, filename } = await charactersApi.exportCharacter(character.id, format);
    await downloadBlobStreaming(blob, filename);
    toast.success(t("characters.exported", { name: filename }));
  } catch (error) {
    if (isDownloadCancelled(error)) return;
    toast.error(t("characters.exportFailed"), { description: errorMessage(error) });
  }
}

export async function duplicateCharacter(character: Character, t: T): Promise<Character | null> {
  try {
    const copy = await charactersApi.duplicateCharacter(character.id);
    useCharactersStore.getState().upsert(copy);
    toast.success(t("characters.duplicated", { name: copy.name }));
    return copy;
  } catch (error) {
    toast.error(t("characters.duplicateFailed"), { description: errorMessage(error) });
    return null;
  }
}

export async function deleteCharacter(character: Character, t: T): Promise<boolean> {
  try {
    await charactersApi.deleteCharacter(character.id);
    useCharactersStore.getState().remove(character.id);
    toast.success(t("characters.deleted", { name: character.name }));
    notifyChatHistoryUpdated();
    return true;
  } catch (error) {
    toast.error(t("characters.deleteFailed"), { description: errorMessage(error) });
    return false;
  }
}

export async function toggleFavorite(character: Character): Promise<void> {
  const store = useCharactersStore.getState();
  store.upsert({ ...character, favorite: !character.favorite });
  try {
    store.upsert(await charactersApi.updateCharacter(character.id, { favorite: !character.favorite }));
  } catch (error) {
    store.upsert(character);
    toast.error(errorMessage(error));
  }
}

export async function makeDefaultPersona(persona: Persona): Promise<void> {
  const store = useCharactersStore.getState();
  try {
    store.upsertPersona(await charactersApi.updatePersona(persona.id, { isDefault: true }));
    await store.refreshPersonas();
  } catch (error) {
    toast.error(errorMessage(error));
  }
}

export async function deletePersona(persona: Persona): Promise<void> {
  const store = useCharactersStore.getState();
  try {
    await charactersApi.deletePersona(persona.id);
    store.removePersona(persona.id);
    await store.refreshPersonas();
  } catch (error) {
    toast.error(errorMessage(error));
  }
}
