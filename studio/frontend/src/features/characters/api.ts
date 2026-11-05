// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  Character,
  CharacterChatSummary,
  CharacterDraft,
  ChatScene,
  Persona,
  SceneSettings,
} from "./types";

export const CHARACTERS_API = "/api/characters";

async function json<T>(response: Response, fallback: string): Promise<T> {
  if (!response.ok) throw new Error(await readFastApiError(response, fallback));
  return (await response.json()) as T;
}

async function ok(response: Response, fallback: string): Promise<void> {
  if (!response.ok) throw new Error(await readFastApiError(response, fallback));
}

function body(value: unknown): RequestInit {
  return {
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  };
}

export async function listCharacters(): Promise<Character[]> {
  const result = await json<{ characters: Character[] }>(
    await authFetch(CHARACTERS_API),
    "Could not load characters",
  );
  return result.characters;
}

export async function getCharacter(id: string): Promise<Character> {
  return json(await authFetch(`${CHARACTERS_API}/${encodeURIComponent(id)}`), "Could not load the character");
}

export async function createCharacter(draft: Partial<CharacterDraft> & { name: string }): Promise<Character> {
  return json(
    await authFetch(CHARACTERS_API, { method: "POST", ...body(draft) }),
    "Could not create the character",
  );
}

export type CharacterPatch = Partial<CharacterDraft> & {
  clear?: ("depthPrompt" | "characterBook")[];
};

export async function updateCharacter(id: string, patch: CharacterPatch): Promise<Character> {
  return json(
    await authFetch(`${CHARACTERS_API}/${encodeURIComponent(id)}`, { method: "PATCH", ...body(patch) }),
    "Could not save the character",
  );
}

export async function deleteCharacter(id: string): Promise<void> {
  return ok(
    await authFetch(`${CHARACTERS_API}/${encodeURIComponent(id)}`, { method: "DELETE" }),
    "Could not delete the character",
  );
}

export async function duplicateCharacter(id: string): Promise<Character> {
  return json(
    await authFetch(`${CHARACTERS_API}/${encodeURIComponent(id)}/duplicate`, { method: "POST" }),
    "Could not duplicate the character",
  );
}

export interface ImportResult {
  character: Character;
  format: "png" | "json" | "charx";
  spec: string;
  warnings: string[];
}

export async function importCharacterFile(file: Blob, filename: string): Promise<ImportResult> {
  return json(
    await authFetch(`${CHARACTERS_API}/import?filename=${encodeURIComponent(filename.slice(0, 255))}`, {
      method: "POST",
      body: file,
      headers: { "Content-Type": "application/octet-stream" },
    }),
    "Could not import the card",
  );
}

export async function uploadCharacterAvatar(id: string, file: Blob): Promise<Character> {
  return json(
    await authFetch(`${CHARACTERS_API}/${encodeURIComponent(id)}/avatar`, {
      method: "PUT",
      body: file,
      headers: { "Content-Type": "application/octet-stream" },
    }),
    "Could not save the picture",
  );
}

export async function removeCharacterAvatar(id: string): Promise<Character> {
  return json(
    await authFetch(`${CHARACTERS_API}/${encodeURIComponent(id)}/avatar`, { method: "DELETE" }),
    "Could not remove the picture",
  );
}

// rev is in the URL because the server marks pictures immutable: a new picture needs a new URL.
export async function fetchImage(kind: "character" | "persona", id: string, rev: number): Promise<Blob> {
  const path =
    kind === "character"
      ? `${CHARACTERS_API}/${encodeURIComponent(id)}/avatar?rev=${rev}`
      : `${CHARACTERS_API}/personas/${encodeURIComponent(id)}/avatar?rev=${rev}`;
  const response = await authFetch(path);
  if (!response.ok) throw new Error(`Picture unavailable (${response.status})`);
  return response.blob();
}

export async function exportCharacter(id: string, format: "png" | "json"): Promise<{ blob: Blob; filename: string }> {
  const response = await authFetch(
    `${CHARACTERS_API}/${encodeURIComponent(id)}/export?format=${format}`,
  );
  if (!response.ok) throw new Error(await readFastApiError(response, "Could not export the character"));
  const disposition = response.headers.get("content-disposition") ?? "";
  const match = /filename\*=UTF-8''([^;]+)/i.exec(disposition);
  const filename = match ? decodeURIComponent(match[1]) : `character.${format}`;
  return { blob: await response.blob(), filename };
}

export async function listCharacterChats(id: string): Promise<CharacterChatSummary[]> {
  const result = await json<{ chats: CharacterChatSummary[] }>(
    await authFetch(`${CHARACTERS_API}/${encodeURIComponent(id)}/chats`),
    "Could not load chats",
  );
  return result.chats;
}

export async function listPersonas(): Promise<Persona[]> {
  const result = await json<{ personas: Persona[] }>(
    await authFetch(`${CHARACTERS_API}/personas`),
    "Could not load personas",
  );
  return result.personas;
}

export async function createPersona(input: { name: string; description?: string; isDefault?: boolean }): Promise<Persona> {
  return json(
    await authFetch(`${CHARACTERS_API}/personas`, { method: "POST", ...body(input) }),
    "Could not create the persona",
  );
}

export async function updatePersona(
  id: string,
  patch: { name?: string; description?: string; isDefault?: boolean },
): Promise<Persona> {
  return json(
    await authFetch(`${CHARACTERS_API}/personas/${encodeURIComponent(id)}`, { method: "PATCH", ...body(patch) }),
    "Could not save the persona",
  );
}

export async function deletePersona(id: string): Promise<void> {
  return ok(
    await authFetch(`${CHARACTERS_API}/personas/${encodeURIComponent(id)}`, { method: "DELETE" }),
    "Could not delete the persona",
  );
}

export async function uploadPersonaAvatar(id: string, file: Blob): Promise<Persona> {
  return json(
    await authFetch(`${CHARACTERS_API}/personas/${encodeURIComponent(id)}/avatar`, {
      method: "PUT",
      body: file,
      headers: { "Content-Type": "application/octet-stream" },
    }),
    "Could not save the picture",
  );
}

export async function removePersonaAvatar(id: string): Promise<Persona> {
  return json(
    await authFetch(`${CHARACTERS_API}/personas/${encodeURIComponent(id)}/avatar`, { method: "DELETE" }),
    "Could not remove the picture",
  );
}

export interface CreatedCharacterChat {
  threadId: string;
  greetingIds: string[];
  headId: string;
  scene: ChatScene;
}

export async function createCharacterChat(input: {
  characterIds: string[];
  personaId?: string | null;
  userName?: string;
  projectId?: string | null;
  settings?: SceneSettings;
  /** Per character, [firstMes, ...alternateGreetings] with macros resolved. */
  greetings?: Record<string, string[]>;
}): Promise<CreatedCharacterChat> {
  return json(
    await authFetch(`${CHARACTERS_API}/chats`, { method: "POST", ...body(input) }),
    "Could not start the chat",
  );
}

export async function getChatScene(threadId: string): Promise<ChatScene | null> {
  const response = await authFetch(`${CHARACTERS_API}/chats/${encodeURIComponent(threadId)}`);
  // 204: no characters (an ordinary chat, or a new one not saved yet).
  if (response.status === 204) return null;
  return json(response, "Could not load the chat's characters");
}

/** One edit applied on top of the stored scene, so edits from several tabs compose. */
export interface ScenePatch {
  add?: string[];
  remove?: string[];
  mute?: Record<string, boolean>;
  personaId?: string | null;
  /** Merged into the stored settings: null clears a key, and the author's note merges field by field. */
  settings?: {
    [K in keyof SceneSettings]?: SceneSettings[K] | null;
  };
}

export async function updateChatScene(threadId: string, patch: ScenePatch): Promise<ChatScene> {
  return json(
    await authFetch(`${CHARACTERS_API}/chats/${encodeURIComponent(threadId)}`, { method: "PUT", ...body(patch) }),
    "Could not update the chat",
  );
}

export async function getChatIndex(): Promise<Record<string, string[]>> {
  const result = await json<{ chats: Record<string, string[]> }>(
    await authFetch(`${CHARACTERS_API}/chat-index`),
    "Could not load character chats",
  );
  return result.chats;
}
