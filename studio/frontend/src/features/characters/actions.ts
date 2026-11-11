// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { notifyChatHistoryUpdated } from "@/features/chat/api/chat-api";
import { writeBranchHead } from "@/features/chat/utils/branch-head";
import { downloadBlobStreaming, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import type { TranslationKey } from "@/i18n";
import {
  createCharacterChat,
  deleteCharacter,
  duplicateCharacter,
  exportCharacter,
  importCharacterFile,
  listCharacterChats,
  updateCharacter,
} from "./api";
import { resolvePersona } from "./chat-runtime";
import { applyMacros } from "./prompt/macros";
import { useCharactersStore } from "./store";
import type { Character } from "./types";

type T = (key: TranslationKey, values?: Record<string, string | number>) => string;

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

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
  const created = await createCharacterChat({
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

export async function lastChatWith(characterId: string): Promise<string | null> {
  const chats = await listCharacterChats(characterId);
  return chats.find((chat) => !chat.archived)?.threadId ?? null;
}

export async function importCards(files: File[], t: T): Promise<Character[]> {
  const imported: Character[] = [];
  for (const file of files) {
    const toastId = toast.loading(t("characters.importing", { name: file.name }));
    try {
      const result = await importCharacterFile(file, file.name);
      useCharactersStore.getState().upsert(result.character);
      imported.push(result.character);
      toast.success(t("characters.imported", { name: result.character.name }), { id: toastId });
    } catch (error) {
      toast.error(t("characters.importFailed", { name: file.name }), {
        id: toastId,
        description: errorText(error),
      });
    }
  }
  return imported;
}

export async function exportCharacterFile(character: Character, format: "png" | "json", t: T): Promise<void> {
  try {
    const { blob, filename } = await exportCharacter(character.id, format);
    await downloadBlobStreaming(blob, filename);
    toast.success(t("characters.exported", { name: filename }));
  } catch (error) {
    if (isDownloadCancelled(error)) return;
    toast.error(t("characters.exportFailed"), { description: errorText(error) });
  }
}

export async function duplicateCharacterAction(character: Character, t: T): Promise<Character | null> {
  try {
    const copy = await duplicateCharacter(character.id);
    useCharactersStore.getState().upsert(copy);
    toast.success(t("characters.duplicated", { name: copy.name }));
    return copy;
  } catch (error) {
    toast.error(errorText(error));
    return null;
  }
}

export async function deleteCharacterAction(character: Character, t: T): Promise<boolean> {
  try {
    await deleteCharacter(character.id);
    useCharactersStore.getState().remove(character.id);
    toast.success(t("characters.deleted", { name: character.name }));
    notifyChatHistoryUpdated();
    return true;
  } catch (error) {
    toast.error(t("characters.deleteFailed"), { description: errorText(error) });
    return false;
  }
}

export async function toggleFavorite(character: Character): Promise<void> {
  const store = useCharactersStore.getState();
  store.upsert({ ...character, favorite: !character.favorite });
  try {
    store.upsert(await updateCharacter(character.id, { favorite: !character.favorite }));
  } catch (error) {
    store.upsert(character);
    toast.error(errorText(error));
  }
}

