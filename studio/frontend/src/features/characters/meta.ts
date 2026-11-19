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

type MessageLike = { role?: string; metadata?: unknown };

/** The roleplay record on a thread message (`metadata.custom.roleplay`) or a stored one (`metadata.roleplay`). */
export function roleplayMetaOf(message: MessageLike | null | undefined): RoleplayMessageMeta | null {
  const metadata = message?.metadata as Record<string, unknown> | null | undefined;
  if (!metadata) return null;
  const custom = (metadata.custom as Record<string, unknown> | undefined) ?? metadata;
  const value = custom?.roleplay as Record<string, unknown> | undefined;
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

/** The planned speakers for a group round, stamped on its first reply. */
export function roundPlanOf(message: MessageLike | null | undefined): string[] | null {
  const metadata = message?.metadata as Record<string, unknown> | null | undefined;
  const custom = (metadata?.custom as Record<string, unknown> | undefined) ?? metadata;
  const plan = (custom?.roleplay as Record<string, unknown> | undefined)?.plan;
  return Array.isArray(plan) && plan.every((id) => typeof id === "string") ? (plan as string[]) : null;
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
