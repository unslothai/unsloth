// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import { getAuthToken } from "@/features/auth";
import { loginDisplayName } from "@/features/profile/hooks/use-effective-profile";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { decodeJwtSubject } from "@/features/profile/utils/jwt-subject";
import { roleplayMetaOf } from "./meta";
import {
  buildCharacterPrompt,
  replyNameOf,
  shapeReplayMessage,
  type BuiltPrompt,
  type PromptCharacter,
} from "./prompt/build";
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
  sends: (messageId: string) => boolean;
  stop: string[];
  /** Stamped on the reply as `metadata.custom.roleplay`. */
  replyMeta: { kind: "reply"; speakerId: string; speakerName: string; plan?: string[] };
  built: BuiltPrompt;
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

/**
 * The chat's characters and persona, loaded for a model call. Refuses rather than fall back: a stale
 * library would drop a member from the prompt, and unloaded personas would speak as the account.
 */
export async function loadParticipants(scene: ChatScene): Promise<{ cast: Character[]; persona: ResolvedPersona }> {
  const store = useCharactersStore.getState();
  await ensureLoaded();
  let cast = castOf(scene, useCharactersStore.getState().characters);
  if (cast.length < scene.cast.length) {
    const refreshed = await store.refresh().then(
      () => true,
      () => false,
    );
    cast = castOf(scene, useCharactersStore.getState().characters);
    if (!refreshed && cast.length < scene.cast.length) throw new CharacterChatUnavailableError(null);
  }
  const { personas, personasStatus } = useCharactersStore.getState();
  if (personasStatus !== "ready" || (scene.personaId && !personas.some((p) => p.id === scene.personaId))) {
    const refreshed = await store.refreshPersonas().then(
      () => true,
      () => false,
    );
    if (!refreshed) throw new CharacterChatUnavailableError(null);
  }
  return { cast, persona: resolvePersona(scene, useCharactersStore.getState().personas) };
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
  /** Re-read the scene first, since another tab may have changed the cast or persona. */
  revalidate?: boolean;
}): Promise<CharacterRun | null> {
  const { threadId, messages } = args;
  if (!threadId) return null;
  const store = useCharactersStore.getState();
  // Sending a character chat as an ordinary one would put its hidden turns in front of the model
  // and save an unattributed reply, so a chat known to have characters refuses instead.
  const knownCharacterChat = isKnownCharacterChat(threadId) || messages.some((m) => roleplayMetaOf(m) !== null);
  let scene: ChatScene | null;
  try {
    // Only a known character chat is re-read; an ordinary chat cannot gain characters.
    scene = await store.loadScene(threadId, args.revalidate === true && Boolean(store.scenes[threadId]));
  } catch (error) {
    scene = store.scenes[threadId] ?? null;
    if (!scene && knownCharacterChat) throw new CharacterChatUnavailableError(error);
  }
  if (!scene) return null;
  const { cast, persona } = await loadParticipants(scene);
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
  const unsent = new Set(chat.filter((_m, index) => !built.sent[index]).map((m) => m.id));
  // By id, so a later copy of the message (one with attachment paths added) is shaped the same.
  const shape = <M extends { id: string }>(message: M): M => {
    if (!byId.has(message.id)) return message;
    const thread = message as unknown as ThreadMessage;
    const meta = roleplayMetaOf(thread);
    const name = replyNameOf(meta, cast, built.speaker.name);
    const recorded = meta?.kind === "reply" ? meta.speakerName || name : name;
    return shapeReplayMessage(thread, byId.get(message.id)!, name, built.named, recorded) as unknown as M;
  };
  return {
    system: built.system,
    messages: messages.filter((m) => !unsent.has(m.id)).map(shape),
    shape,
    sends: (messageId) => !unsent.has(messageId),
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
