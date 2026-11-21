// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Character, ChatScene, Persona, RoleplayMessageMeta } from "./types";

/** A new chat takes the default persona; a saved chat keeps its pick (null = the account), so a later default doesn't rename it. */
export function pickPersona(scene: ChatScene | null, personas: Persona[]): Persona | null {
  const fallback = personas.find((p) => p.isDefault) ?? null;
  if (!scene) return fallback;
  if (!scene.personaId) return null;
  // A pick that no longer exists falls back as a persona delete does on the server.
  return personas.find((p) => p.id === scene.personaId) ?? fallback;
}

export type MessageLike = { role?: string; metadata?: unknown };

/** The raw roleplay record on a thread message (`metadata.custom.roleplay`) or a stored one (`metadata.roleplay`). */
function roleplayRecordOf(message: MessageLike | null | undefined): Record<string, unknown> | undefined {
  const metadata = message?.metadata as Record<string, unknown> | null | undefined;
  if (!metadata) return undefined;
  const custom = (metadata.custom as Record<string, unknown> | undefined) ?? metadata;
  return custom?.roleplay as Record<string, unknown> | undefined;
}

export function roleplayMetaOf(message: MessageLike | null | undefined): RoleplayMessageMeta | null {
  const value = roleplayRecordOf(message);
  if (!value || typeof value !== "object" || typeof value.kind !== "string") return null;
  switch (value.kind) {
    case "opening":
      return { kind: "opening" };
    case "turn":
      return typeof value.speakerId === "string" ? { kind: "turn", speakerId: value.speakerId } : null;
    case "reply":
      return typeof value.speakerId === "string"
        ? {
            kind: "reply",
            speakerId: value.speakerId,
            speakerName: typeof value.speakerName === "string" ? value.speakerName : "",
            ...(typeof value.greeting === "number" ? { greeting: value.greeting } : {}),
          }
        : null;
    default:
      return null;
  }
}

/** Which bookkeeping turn a user message is (the opening, or a turn marker); they reach the model but never the screen. */
export function hiddenRoleplayTurnKind(message: MessageLike | null | undefined): "opening" | "turn" | null {
  if (message?.role !== "user") return null;
  const kind = roleplayMetaOf(message)?.kind;
  return kind === "opening" || kind === "turn" ? kind : null;
}

export function isHiddenRoleplayTurn(message: MessageLike | null | undefined): boolean {
  return hiddenRoleplayTurnKind(message) !== null;
}

/** Drops hidden turns except where one is the only user turn before a reply, so strict templates still
 *  see roles alternate (a chat whose characters were all deleted, training exports). */
export function alternatingTurns<M extends MessageLike>(messages: readonly M[]): M[] {
  return messages.filter((message, index) => !isHiddenRoleplayTurn(message) || messages[index + 1]?.role === "assistant");
}

/** The planned speakers for a group round, stamped on its first reply. */
export function roundPlanOf(message: MessageLike | null | undefined): string[] | null {
  const plan = roleplayRecordOf(message)?.plan;
  return Array.isArray(plan) && plan.every((id) => typeof id === "string") ? (plan as string[]) : null;
}

/** The newest speaker among replies (and, if asked, turn markers). */
export function lastSpeakerId(messages: readonly MessageLike[], includeTurns = false): string | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const meta = roleplayMetaOf(messages[i]);
    if (meta?.kind === "reply" || (includeTurns && meta?.kind === "turn")) return meta.speakerId;
  }
  return null;
}

/** Keeps unchanged records (and the map itself) so a refetch doesn't re-render readers. */
export function mergeCharacters(
  previous: Record<string, Character>,
  list: readonly Character[],
): Record<string, Character> {
  let changed = list.length !== Object.keys(previous).length;
  const next: Record<string, Character> = {};
  for (const character of list) {
    const kept = previous[character.id];
    const same =
      kept !== undefined &&
      kept.updatedAt === character.updatedAt &&
      kept.avatarRev === character.avatarRev &&
      kept.lastChatAt === character.lastChatAt &&
      kept.chatCount === character.chatCount;
    if (!same) changed = true;
    next[character.id] = same ? kept : character;
  }
  return changed ? next : previous;
}
