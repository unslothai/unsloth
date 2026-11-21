// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import { getLocale, translate } from "@/i18n";
import { getCharacter } from "./api";
import { isCharacterChat } from "./chat-gate";
import { isHiddenRoleplayTurn, roleplayMetaOf, type MessageLike } from "./meta";
import { resolvePersona } from "./persona";
import {
  buildCharacterPrompt,
  replyNameOf,
  resumedPrefill,
  shapeReplayMessage,
  type BuiltPrompt,
  type PromptCharacter,
} from "./prompt/build";
import { castOf, resolveSpeaker, textOf } from "./prompt/round";
import { useCharactersStore } from "./store";
import type { Character, CharacterReplyMeta, ChatScene, ResolvedPersona } from "./types";

export const CHARACTER_SPEAKER_RUN_KEY = "roleplaySpeakerId";

export class CharacterChatUnavailableError extends Error {
  constructor(cause: unknown) {
    super(translate("characters.chat.castUnavailable", undefined, getLocale()));
    this.name = "CharacterChatUnavailableError";
    this.cause = cause;
  }
}

export interface CharacterRun {
  system: string;
  messages: ThreadMessage[];
  shape: <M extends { id: string }>(message: M) => M;
  sends: (messageId: string) => boolean;
  /** Messages that carry a note placed at a depth, so View prompt can show them. */
  notedIds: ReadonlySet<string>;
  stop: string[];
  /** A resumed reply's partial as the model should see it: named like the rest of a group's replies. */
  prefill: (partial: string) => string;
  /** Stamped on the reply as `metadata.custom.roleplay`. */
  replyMeta: CharacterReplyMeta;
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

async function ensureLoaded(): Promise<void> {
  const store = useCharactersStore.getState();
  await Promise.all([
    store.status === "ready" ? null : store.refresh().catch(() => undefined),
    store.personasStatus === "ready" ? null : store.refreshPersonas().catch(() => undefined),
  ]);
}

// Edits from another device are picked up within this window; a group round's later speakers reuse the check.
const REVALIDATE_MS = 30_000;
const revalidatedAt = new Map<string, number>();

async function revalidateParticipants(scene: ChatScene): Promise<void> {
  const last = revalidatedAt.get(scene.threadId);
  if (last !== undefined && Date.now() - last < REVALIDATE_MS) return;
  revalidatedAt.set(scene.threadId, Date.now());
  await Promise.all([
    ...scene.cast.map((member) =>
      getCharacter(member.characterId).then(
        (fresh) => {
          const kept = useCharactersStore.getState().characters[fresh.id];
          if (kept?.updatedAt !== fresh.updatedAt || kept.avatarRev !== fresh.avatarRev) {
            useCharactersStore.getState().upsert(fresh);
          }
        },
        () => undefined,
      ),
    ),
    useCharactersStore.getState().refreshPersonas().catch(() => undefined),
  ]);
}

/** Refuses rather than fall back: a stale library drops members, unloaded personas speak as the account. */
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

function isKnownCharacterChat(threadId: string | null | undefined): boolean {
  if (!threadId) return false;
  const store = useCharactersStore.getState();
  return Boolean(store.scenes[threadId]) || threadId in store.chatIndex;
}

function stampOf(message: ThreadMessage | undefined): Date | undefined {
  return message?.createdAt instanceof Date && !Number.isNaN(message.createdAt.getTime()) ? message.createdAt : undefined;
}

/** A thread's chat turns as prompt history, seeded and dated by the chat so a re-send builds the same text. */
export function promptHistoryOf(threadId: string, messages: readonly ThreadMessage[]) {
  const chat = messages.filter((m) => m.role === "user" || m.role === "assistant");
  return {
    chat,
    input: {
      seed: threadId,
      startedAt: stampOf(chat[0]),
      now: stampOf(chat[chat.length - 1]),
      history: chat.map((m) => ({ role: m.role as "user" | "assistant", text: textOf(m), meta: roleplayMetaOf(m) })),
    },
  };
}

/** A character chat's run: the transformed history, its system prompt and stop strings; null for ordinary chats. */
export async function prepareCharacterRun(args: {
  threadId: string | undefined;
  messages: readonly ThreadMessage[];
  runConfig?: { custom?: Record<string, unknown> } | null;
  revalidate?: boolean;
}): Promise<CharacterRun | null> {
  const { threadId, messages } = args;
  if (!threadId) return null;
  const store = useCharactersStore.getState();
  // Sending as an ordinary chat would expose hidden turns and save an unattributed reply.
  const hasRoleplayMeta = messages.some((m) => roleplayMetaOf(m) !== null);
  const knownCharacterChat = isKnownCharacterChat(threadId) || hasRoleplayMeta;
  // An ordinary chat the loaded index doesn't list costs no scene request before its first token.
  if (!isCharacterChat(threadId, hasRoleplayMeta)(store)) return null;
  let scene: ChatScene | null;
  try {
    scene = await store.loadScene(threadId, args.revalidate === true && Boolean(store.scenes[threadId]));
  } catch (error) {
    scene = store.scenes[threadId] ?? null;
    if (!scene && knownCharacterChat) throw new CharacterChatUnavailableError(error);
  }
  if (!scene) return null;
  if (args.revalidate) await revalidateParticipants(scene);
  const { cast, persona } = await loadParticipants(scene);
  if (cast.length === 0) {
    // Every character in the chat was deleted: it carries on as an ordinary chat.
    if (scene.cast.length === 0) return null;
    throw new CharacterChatUnavailableError(null);
  }
  const requested = args.runConfig?.custom?.[CHARACTER_SPEAKER_RUN_KEY];
  const speaker = resolveSpeaker({
    scene,
    cast,
    messages,
    requested: typeof requested === "string" ? requested : null,
  });
  if (!speaker) return null;
  const { chat, input } = promptHistoryOf(threadId, messages);
  const built = buildCharacterPrompt({
    cast: cast.map(toPromptCharacter),
    speakerId: speaker.speakerId,
    persona: { name: persona.name, description: persona.description },
    scene: scene.settings,
    ...input,
  });
  const byId = new Map(chat.map((m, index) => [m.id, built.texts[index]]));
  const unsent = new Set(chat.filter((_m, index) => !built.sent[index]).map((m) => m.id));
  // By id, so a later copy of the message (one with attachment paths added) is shaped the same.
  const shape = <M extends { id: string }>(message: M): M => {
    if (!byId.has(message.id)) return message;
    const thread = message as unknown as ThreadMessage;
    const meta = roleplayMetaOf(thread);
    const name = replyNameOf(meta, cast, cast[0].name);
    const recorded = meta?.kind === "reply" ? meta.speakerName || name : name;
    return shapeReplayMessage(thread, byId.get(message.id)!, name, built.namesTurns, recorded) as unknown as M;
  };
  return {
    system: built.system,
    messages: messages.filter((m) => !unsent.has(m.id)).map(shape),
    shape,
    sends: (messageId) => !unsent.has(messageId),
    notedIds: new Set(chat.filter((_m, index) => built.noted[index] && built.sent[index]).map((m) => m.id)),
    stop: built.stop,
    prefill: (partial) => resumedPrefill(partial, built.speaker.name, built.namesTurns),
    replyMeta: {
      kind: "reply",
      speakerId: built.speaker.id,
      speakerName: built.speaker.name,
      ...(speaker.plan && cast.length > 1 ? { plan: speaker.plan } : {}),
    },
    built,
  };
}

/** What a run sends: a character chat's shaped turns, else the history minus hidden turn markers (a chat whose cast was all deleted). */
export function characterOutbound<M extends MessageLike & { id: string }>(run: CharacterRun | null, messages: readonly M[]): M[] {
  return run ? messages.filter((m) => run.sends(m.id)).map(run.shape) : messages.filter((m) => !isHiddenRoleplayTurn(m));
}

// The context meter recounts the same branch often; reuse its run (and lorebook scan) until the branch changes.
const countRuns = new WeakMap<object, { threadId: string | undefined; run: Promise<CharacterRun | null> }>();

/** prepareCharacterRun for a token count, cached per raw branch; only a built run is reused. */
export function prepareCharacterCountRun(
  rawMessages: object,
  threadId: string | undefined,
  messages: readonly ThreadMessage[],
): Promise<CharacterRun | null> {
  const cached = countRuns.get(rawMessages);
  if (cached && cached.threadId === threadId) return cached.run;
  const run = prepareCharacterRun({ threadId, messages }).catch(() => null);
  countRuns.set(rawMessages, { threadId, run });
  // An ordinary or not-yet-loaded chat is asked again next count.
  void run.then((built) => {
    if (!built && countRuns.get(rawMessages)?.run === run) countRuns.delete(rawMessages);
  });
  return run;
}
