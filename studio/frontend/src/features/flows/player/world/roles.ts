// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which autonomy a world opted into, and who plays the rival or the hostile villager.

import { seedFrom } from "../../expr";
import type { WorldNpc, WorldQuest, WorldSpec } from "../../types";

export const AUTONOMY = [
  "trade",
  "homes",
  "quests",
  "rivals",
  "enemies",
] as const;
export type AutonomyFlag = (typeof AUTONOMY)[number];
export type Flags = Record<AutonomyFlag, boolean>;
export type NpcRole = "rival" | "hostile";
export const ROLES: readonly NpcRole[] = ["rival", "hostile"];
/** Rivals or hostiles the quest director may add, each. */
export const MAX_ADDED_ROLE = 2;

export function autonomyOf(spec: WorldSpec): Flags {
  const a = spec.autonomy ?? {};
  return {
    trade: a.trade === true,
    homes: a.homes === true,
    quests: a.quests === true,
    rivals: a.rivals === true,
    enemies: a.enemies === true,
  };
}

export const autonomyOn = (spec: WorldSpec) =>
  AUTONOMY.some((f) => autonomyOf(spec)[f]);

const walks = (n: WorldNpc) => n.wander !== false;

function pick<T>(list: T[], ...seed: unknown[]): T | null {
  return list.length ? list[seedFrom(...seed) % list.length] : null;
}

/**
 * The hostile villagers: those the world marks hostile, or (when enemies are on and none is marked) one villager
 * who gives no quest, picked by seed.
 */
export function hostilesOf(spec: WorldSpec): string[] {
  if (!autonomyOf(spec).enemies) return [];
  const marked = spec.npcs.filter((n) => n.role === "hostile" && walks(n));
  if (marked.length) return marked.map((n) => n.id);
  const givers = new Set(spec.quests.map((q) => q.giver));
  const one = pick(
    spec.npcs.filter((n) => !givers.has(n.id) && walks(n) && !n.role),
    spec.seed,
    "hostile",
  );
  return one ? [one.id] : [];
}

/** Who races the player for a quest's item: a marked rival other than the giver, else one picked by seed. */
export function rivalFor(spec: WorldSpec, q: WorldQuest): string | null {
  if (!autonomyOf(spec).quests) return null;
  const hostile = new Set(hostilesOf(spec));
  const ok = spec.npcs.filter(
    (n) => n.id !== q.giver && walks(n) && !hostile.has(n.id),
  );
  const marked = ok.filter((n) => n.role === "rival");
  if (marked.length) return marked[0].id;
  return pick(ok, spec.seed, "rival", q.id)?.id ?? null;
}

export function roleOf(spec: WorldSpec, npcId: string): NpcRole | null {
  if (hostilesOf(spec).includes(npcId)) return "hostile";
  const npc = spec.npcs.find((n) => n.id === npcId);
  if (npc?.role === "rival" && autonomyOf(spec).quests) return "rival";
  return spec.quests.some((q) => rivalFor(spec, q) === npcId) &&
    autonomyOf(spec).quests
    ? "rival"
    : null;
}
