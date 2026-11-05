// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import { getAuthToken } from "@/features/auth";
import { loginDisplayName } from "@/features/profile/hooks/use-effective-profile";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { decodeJwtSubject } from "@/features/profile/utils/jwt-subject";
import { roleplayMetaOf } from "./meta";
import { buildCharacterPrompt, type BuiltPrompt, type PromptCharacter } from "./prompt/build";
import { castOf, resolveSpeaker, textOf } from "./prompt/round";
import { useCharactersStore } from "./store";
import type { Character, ChatScene, Persona, ResolvedPersona } from "./types";

export const ROLEPLAY_SPEAKER_RUN_KEY = "roleplaySpeakerId";

export class CharacterChatUnavailableError extends Error {
  constructor(cause: unknown) {
    super("This chat's characters could not be loaded, so the message was not sent. Try again in a moment.");
    this.name = "CharacterChatUnavailableError";
    this.cause = cause;
  }
}

export interface CharacterRun {
  system: string;
  messages: ThreadMessage[];
  /** One message as this run sends it. */
  shape: <M extends { id: string }>(message: M) => M;
  stop: string[];
  /** Stamped on the reply as `metadata.custom.roleplay`. */
  replyMeta: { kind: "reply"; speakerId: string; speakerName: string; plan?: string[] };
  built: BuiltPrompt;
}

function withText(message: ThreadMessage, text: string): ThreadMessage {
  const parts = message.content as readonly { type: string }[];
  const firstText = parts.findIndex((part) => part.type === "text");
  const others = parts.filter((part) => part.type !== "text");
  const replacement = { type: "text" as const, text };
  const next =
    firstText < 0
      ? [replacement, ...others]
      : [...others.slice(0, firstText), replacement, ...others.slice(firstText)];
  return { ...message, content: next } as unknown as ThreadMessage;
}

export function toPromptCharacter(character: Character): PromptCharacter {
  return {
    id: character.id,
    name: character.name,
    description: character.description,
    personality: character.personality,
    scenario: character.scenario,
    mesExample: character.mesExample,
    systemPrompt: character.systemPrompt,
    postHistoryInstructions: character.postHistoryInstructions,
    depthPrompt: character.depthPrompt,
    characterBook: character.characterBook,
  };
}

/** The persona a chat speaks as: its own pick, the default persona, else the account's name. */
export function resolvePersona(scene: ChatScene | null, personas: Persona[]): ResolvedPersona {
  const chosen =
    (scene?.personaId && personas.find((p) => p.id === scene.personaId)) ||
    personas.find((p) => p.isDefault) ||
    null;
  if (chosen) {
    return {
      id: chosen.id,
      name: chosen.name,
      description: chosen.description,
      hasAvatar: chosen.hasAvatar,
      avatarRev: chosen.avatarRev,
    };
  }
  const profile = useUserProfileStore.getState();
  // Same order the app uses to address the user: nickname, first name, then the login.
  const name =
    profile.nickname.trim() ||
    profile.displayName.trim().split(/\s+/)[0] ||
    loginDisplayName(decodeJwtSubject(getAuthToken())) ||
    "User";
  return { id: null, name, description: "", hasAvatar: false, avatarRev: 0 };
}

async function ensureLoaded(): Promise<void> {
  const store = useCharactersStore.getState();
  await Promise.all([
    store.status === "ready" ? null : store.refresh().catch(() => undefined),
    store.personasStatus === "ready" ? null : store.refreshPersonas().catch(() => undefined),
  ]);
}

/** Whether this thread is known to be a character chat, from what the store has already loaded. */
export function isKnownCharacterChat(threadId: string | null | undefined): boolean {
  if (!threadId) return false;
  const store = useCharactersStore.getState();
  return Boolean(store.scenes[threadId]) || threadId in store.chatIndex;
}

/** A character chat's run: the transformed history, its system prompt and stop strings; null for ordinary chats. */
export async function prepareCharacterRun(args: {
  threadId: string | undefined;
  messages: readonly ThreadMessage[];
  runConfig?: { custom?: Record<string, unknown> } | null;
}): Promise<CharacterRun | null> {
  const { threadId, messages } = args;
  if (!threadId) return null;
  const store = useCharactersStore.getState();
  // Sending a character chat as an ordinary one would put its hidden turns in front of the model
  // and save an unattributed reply, so a chat known to have characters refuses instead.
  const knownCharacterChat = isKnownCharacterChat(threadId);
  let scene: ChatScene | null;
  try {
    scene = await store.loadScene(threadId);
  } catch (error) {
    scene = store.scenes[threadId] ?? null;
    if (!scene && knownCharacterChat) throw new CharacterChatUnavailableError(error);
  }
  if (!scene) return null;
  await ensureLoaded();
  const { characters, personas } = useCharactersStore.getState();
  const cast = castOf(scene, characters);
  if (cast.length === 0) {
    // Every character in the chat was deleted: it carries on as an ordinary chat.
    if (scene.cast.length === 0) return null;
    throw new CharacterChatUnavailableError(null);
  }
  const requested = args.runConfig?.custom?.[ROLEPLAY_SPEAKER_RUN_KEY];
  const speaker = resolveSpeaker({
    scene,
    cast,
    messages,
    requested: typeof requested === "string" ? requested : null,
  });
  if (!speaker) return null;
  const persona = resolvePersona(scene, personas);
  const chat = messages.filter((m) => m.role === "user" || m.role === "assistant");
  const built = buildCharacterPrompt({
    cast: cast.map(toPromptCharacter),
    speakerId: speaker.speakerId,
    persona: { name: persona.name, description: persona.description },
    scene: scene.settings,
    history: chat.map((m) => ({
      role: m.role as "user" | "assistant",
      text: textOf(m),
      meta: roleplayMetaOf(m),
    })),
  });
  const byId = new Map(chat.map((m, index) => [m.id, built.texts[index]]));
  // By id, so a later copy of the message (one with attachment paths added) is shaped the same.
  const shape = <M extends { id: string }>(message: M): M =>
    byId.has(message.id) ? (withText(message as unknown as ThreadMessage, byId.get(message.id)!) as unknown as M) : message;
  return {
    system: built.system,
    messages: messages.map(shape),
    shape,
    stop: built.stop,
    replyMeta: {
      kind: "reply",
      speakerId: built.speaker.id,
      speakerName: built.speaker.name,
      ...(speaker.plan && cast.length > 1 ? { plan: speaker.plan } : {}),
    },
    built,
  };
}

export { castOf, nextInRound, resolveSpeaker, textOf } from "./prompt/round";
