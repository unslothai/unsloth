// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The world's game state outside the renderer: inventory, quests, variables, conversations and save data.
// Phaser owns positions frame to frame; this owns everything a save slot or an NPC needs to know.

import { applySets } from "../../expr";
import type { WorldSpec } from "../../types";
import type { Point } from "./procgen";
import {
  type MemoryRecord,
  type NpcMind,
  type PlanBlock,
  emptyMind,
  initialMinds,
  simOf,
} from "./world-sim";

export const WORLD_SAVE_VERSION = 2;
const MAX_TURNS = 40;

export interface ChatTurn {
  role: "player" | "npc";
  text: string;
}

export interface WorldState {
  v: number;
  player: Point | null;
  inventory: string[];
  picked: string[];
  activeQuests: string[];
  completedQuests: string[];
  vars: Record<string, unknown>;
  talks: Record<string, ChatTurn[]>;
  /** Game minutes since day 1 00:00. */
  clock: number;
  npc: Record<string, NpcMind>;
}

export interface NpcReply {
  text: string;
  give_item?: string;
  take_item?: string;
  complete_quest?: string;
  set?: Record<string, unknown>;
  remember?: { text: string; importance: number }[];
  recalled?: string[];
}

export function initialWorldState(spec: WorldSpec): WorldState {
  const clock = simOf(spec)?.start_minute ?? 0;
  return {
    v: WORLD_SAVE_VERSION,
    player: null,
    inventory: [],
    picked: [],
    activeQuests: [],
    completedQuests: [],
    vars: { ...(spec.vars ?? {}) },
    talks: {},
    clock,
    npc: simOf(spec) ? initialMinds(spec, clock) : {},
  };
}

export function pickUp(s: WorldState, itemId: string): WorldState {
  if (s.picked.includes(itemId)) return s;
  return {
    ...s,
    picked: [...s.picked, itemId],
    inventory: [...s.inventory, itemId],
  };
}

/** Talking to a quest giver starts their open quests. */
export function startQuestsOf(
  spec: WorldSpec,
  s: WorldState,
  npcId: string,
): WorldState {
  const fresh = spec.quests
    .filter(
      (q) =>
        q.giver === npcId &&
        !s.activeQuests.includes(q.id) &&
        !s.completedQuests.includes(q.id),
    )
    .map((q) => q.id);
  return fresh.length
    ? { ...s, activeQuests: [...s.activeQuests, ...fresh] }
    : s;
}

export function withMind(
  s: WorldState,
  npcId: string,
  fn: (mind: NpcMind) => NpcMind,
): WorldState {
  return { ...s, npc: { ...s.npc, [npcId]: fn(s.npc[npcId] ?? emptyMind()) } };
}

export function addTurn(
  s: WorldState,
  npcId: string,
  turn: ChatTurn,
): WorldState {
  const prev = s.talks[npcId] ?? [];
  const next = [...prev, turn].slice(-MAX_TURNS);
  return { ...s, talks: { ...s.talks, [npcId]: next } };
}

/** Apply a validated reply; re-checks every effect against the world so a bad reply cannot corrupt a save. */
export function applyReply(
  spec: WorldSpec,
  s: WorldState,
  npcId: string,
  reply: NpcReply,
): { state: WorldState; notes: string[] } {
  const notes: string[] = [];
  const itemName = (id: string) =>
    spec.items.find((i) => i.id === id)?.name ?? id;
  let st = addTurn(s, npcId, { role: "npc", text: reply.text });
  if (reply.take_item && st.inventory.includes(reply.take_item)) {
    st = {
      ...st,
      inventory: st.inventory.filter((i) => i !== reply.take_item),
    };
    notes.push(`Gave away ${itemName(reply.take_item)}`);
  }
  if (
    reply.give_item &&
    spec.items.some((i) => i.id === reply.give_item) &&
    !st.inventory.includes(reply.give_item)
  ) {
    st = {
      ...st,
      inventory: [...st.inventory, reply.give_item],
      picked: [...new Set([...st.picked, reply.give_item])],
    };
    notes.push(`Received ${itemName(reply.give_item)}`);
  }
  const quest = spec.quests.find((q) => q.id === reply.complete_quest);
  if (
    quest &&
    quest.giver === npcId &&
    !st.completedQuests.includes(quest.id)
  ) {
    let vars = st.vars;
    try {
      vars = applySets(quest.set, st.vars);
    } catch {
      // A quest effect that does not evaluate leaves the variables as they were.
    }
    st = {
      ...st,
      vars,
      completedQuests: [...st.completedQuests, quest.id],
      activeQuests: st.activeQuests.filter((q) => q !== quest.id),
    };
    notes.push(
      `Quest complete: ${quest.title}${quest.reward_text ? ` · ${quest.reward_text}` : ""}`,
    );
  }
  if (reply.set) {
    const vars = { ...st.vars };
    for (const [k, v] of Object.entries(reply.set)) {
      if (Object.hasOwn(spec.vars ?? {}, k) || Object.hasOwn(st.vars, k))
        vars[k] = v;
    }
    st = { ...st, vars };
  }
  return { state: st, notes };
}

export function serialize(s: WorldState): WorldState {
  return JSON.parse(JSON.stringify(s)) as WorldState;
}

const num = (v: unknown, d: number) =>
  typeof v === "number" && Number.isFinite(v) ? v : d;

function mindOf(raw: unknown): NpcMind {
  const r = (raw ?? {}) as Partial<NpcMind>;
  const memory = (Array.isArray(r.memory) ? r.memory : []).filter(
    (m): m is MemoryRecord =>
      !!m && typeof m.id === "string" && typeof m.text === "string",
  );
  const plan = (Array.isArray(r.plan) ? r.plan : []).filter(
    (b): b is PlanBlock =>
      !!b && typeof b.start === "number" && typeof b.place === "string",
  );
  return {
    ...emptyMind(),
    memory,
    nextId: num(r.nextId, memory.length),
    plan,
    planDay: num(r.planDay, -1),
    lastChat:
      r.lastChat && typeof r.lastChat === "object" ? { ...r.lastChat } : {},
  };
}

/** A save from this or an older version (v1 has no clock or NPC minds), or null when it is not one. */
export function deserialize(spec: WorldSpec, raw: unknown): WorldState | null {
  const v = (raw as WorldState | null)?.v;
  if (!raw || typeof raw !== "object" || (v !== 1 && v !== WORLD_SAVE_VERSION))
    return null;
  const r = raw as Partial<WorldState>;
  const itemIds = new Set(spec.items.map((i) => i.id));
  const questIds = new Set(spec.quests.map((q) => q.id));
  const fresh = initialWorldState(spec);
  const npc: Record<string, NpcMind> = {};
  for (const id of Object.keys(fresh.npc))
    npc[id] = r.npc?.[id] ? mindOf(r.npc[id]) : fresh.npc[id];
  return {
    v: WORLD_SAVE_VERSION,
    player:
      r.player && typeof r.player.x === "number"
        ? { x: r.player.x, y: r.player.y }
        : null,
    inventory: (r.inventory ?? []).filter((i) => itemIds.has(i)),
    picked: (r.picked ?? []).filter((i) => itemIds.has(i)),
    activeQuests: (r.activeQuests ?? []).filter((q) => questIds.has(q)),
    completedQuests: (r.completedQuests ?? []).filter((q) => questIds.has(q)),
    vars: { ...(r.vars ?? {}) },
    talks: r.talks ?? {},
    clock: num(r.clock, fresh.clock),
    npc,
  };
}
