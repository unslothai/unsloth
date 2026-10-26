// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { applySets } from "../../expr";
import type { WorldSpec } from "../../types";
import { type AutoState, autoOf, initialAuto } from "./autonomy";
import { type CityState, cityOf, parseCity } from "./city-sim";
import {
  REPLY_ACTIONS,
  type ReplyAction,
  clampNeeds,
  parseDoing,
} from "./needs";
import type { Point } from "./procgen";
import { autonomyOn } from "./roles";
import {
  type WorldAdditions,
  additionsOf,
  noAdditions,
  withAdditions,
} from "./world-director";
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
/** Keep in sync with MAX_HISTORY in backend world.py. */
export const MAX_HISTORY = 24;
export const SUMMARY_AFTER = 16;
const SUMMARY_FOLD = 10;
const MAX_KEPT = 120;

export interface ChatTurn {
  role: "player" | "npc";
  text: string;
  seq?: number;
}

export interface NpcSummary {
  text: string;
  upto: number;
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
  summaries: Record<string, NpcSummary>;
  clock: number;
  npc: Record<string, NpcMind>;
  following?: string[];
  errands?: Record<string, Errand>;
  additions?: WorldAdditions;
  auto?: AutoState;
  npcAt?: Record<string, Point>;
  action?: { combat?: unknown; pose?: { z: number } };
  city?: CityState;
}

export interface Errand {
  place: string;
  until: number | null;
}

export interface NpcReply {
  text: string;
  give_item?: string;
  take_item?: string;
  complete_quest?: string;
  set?: Record<string, unknown>;
  remember?: { text: string; importance: number }[];
  recalled?: string[];
  action?: ReplyAction;
  place?: string;
  offer_trade?: string;
}

export function placeIds(spec: WorldSpec): Set<string> {
  return new Set([
    ...(spec.simulation?.places ?? []).map((p) => p.id),
    ...(spec.regions ?? []).map((r) => r.id),
  ]);
}

export function applyAction(
  spec: WorldSpec,
  s: WorldState,
  npcId: string,
  action: unknown,
  place: unknown,
): WorldState {
  if (!REPLY_ACTIONS.includes(action as ReplyAction)) return s;
  if (spec.npcs.find((n) => n.id === npcId)?.wander === false) return s;
  const following = (s.following ?? []).filter((id) => id !== npcId);
  const { [npcId]: _, ...errands } = s.errands ?? {};
  if (action === "follow_player")
    return { ...s, following: [...following, npcId], errands };
  if (action === "stop_following") return { ...s, following, errands };
  if (typeof place !== "string" || !placeIds(spec).has(place)) return s;
  const until = simOf(spec) ? s.clock + 120 : null;
  return {
    ...s,
    following,
    errands: { ...errands, [npcId]: { place, until } },
  };
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
    summaries: {},
    clock,
    npc: simOf(spec) ? initialMinds(spec, clock) : {},
    following: [],
    errands: {},
    additions: noAdditions(),
    ...(autonomyOn(spec) ? { auto: initialAuto(spec) } : {}),
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

function numbered(turns: ChatTurn[]): (ChatTurn & { seq: number })[] {
  return turns.every((t) => typeof t.seq === "number")
    ? (turns as (ChatTurn & { seq: number })[])
    : turns.map((t, i) => ({ ...t, seq: i }));
}

const uptoOf = (s: WorldState, npcId: string) =>
  s.summaries?.[npcId]?.upto ?? -1;

function trimTalk(
  turns: (ChatTurn & { seq: number })[],
  upto: number,
  summarising: boolean,
): ChatTurn[] {
  if (!summarising) return turns.slice(-MAX_TURNS);
  let drop = 0;
  while (
    turns.length - drop > MAX_TURNS &&
    (turns[drop].seq <= upto || turns.length - drop > MAX_KEPT)
  )
    drop++;
  return turns.slice(drop);
}

export function addTurn(
  s: WorldState,
  npcId: string,
  turn: ChatTurn,
  summarising = false,
): WorldState {
  const prev = numbered(s.talks[npcId] ?? []);
  const seq = Math.max(prev.at(-1)?.seq ?? -1, uptoOf(s, npcId)) + 1;
  const next = trimTalk(
    [...prev, { ...turn, seq }],
    uptoOf(s, npcId),
    summarising,
  );
  return { ...s, talks: { ...s.talks, [npcId]: next } };
}

export function talkHistory(s: WorldState, npcId: string): ChatTurn[] {
  const upto = uptoOf(s, npcId);
  return numbered(s.talks[npcId] ?? [])
    .filter((t) => t.seq > upto)
    .slice(-MAX_HISTORY);
}

export function summaryDue(
  s: WorldState,
  npcId: string,
): { turns: ChatTurn[]; upto: number } | null {
  const upto = uptoOf(s, npcId);
  const open = numbered(s.talks[npcId] ?? []).filter((t) => t.seq > upto);
  if (open.length <= SUMMARY_AFTER) return null;
  const turns = open.slice(0, SUMMARY_FOLD);
  return {
    turns: turns.map(({ role, text }) => ({ role, text })),
    upto: turns[turns.length - 1].seq,
  };
}

export function foldSummary(
  s: WorldState,
  npcId: string,
  text: string,
  upto: number,
): WorldState {
  if (upto <= uptoOf(s, npcId)) return s;
  const summaries = { ...s.summaries, [npcId]: { text, upto } };
  const turns = trimTalk(numbered(s.talks[npcId] ?? []), upto, true);
  return { ...s, summaries, talks: { ...s.talks, [npcId]: turns } };
}

export function applyReply(
  spec: WorldSpec,
  s: WorldState,
  npcId: string,
  reply: NpcReply,
  summarising = false,
): { state: WorldState; notes: string[] } {
  const notes: string[] = [];
  const itemName = (id: string) =>
    spec.items.find((i) => i.id === id)?.name ?? id;
  let st = addTurn(s, npcId, { role: "npc", text: reply.text }, summarising);
  if (reply.take_item && st.inventory.includes(reply.take_item)) {
    st = {
      ...st,
      inventory: st.inventory.filter((i) => i !== reply.take_item),
    };
    notes.push(`Gave away ${itemName(reply.take_item)}`);
  }
  const holder = reply.give_item ? st.auto?.held[reply.give_item] : undefined;
  if (holder && st.auto) {
    const { [reply.give_item as string]: _, ...held } = st.auto.held;
    if (holder === npcId) st = { ...st, auto: { ...st.auto, held } };
  }
  if (
    reply.give_item &&
    (!holder || holder === npcId) &&
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
  if (reply.action) {
    const before = st;
    st = applyAction(spec, st, npcId, reply.action, reply.place);
    const name = spec.npcs.find((n) => n.id === npcId)?.name ?? npcId;
    if (st !== before && reply.action === "follow_player")
      notes.push(`${name} is following you`);
    else if (st !== before && reply.action === "stop_following")
      notes.push(`${name} stopped following you`);
  }
  return { state: st, notes };
}

export function serialize(s: WorldState): WorldState {
  return JSON.parse(JSON.stringify(s)) as WorldState;
}

const num = (v: unknown, d: number) =>
  typeof v === "number" && Number.isFinite(v) ? v : d;

function mindOf(raw: unknown, fresh: NpcMind): NpcMind {
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
    ...(fresh.needs ? { needs: clampNeeds(r.needs, fresh.needs) } : {}),
    ...(fresh.needs && parseDoing(r.doing)
      ? { doing: parseDoing(r.doing) }
      : {}),
  };
}

function summariesOf(raw: unknown): Record<string, NpcSummary> {
  const out: Record<string, NpcSummary> = {};
  for (const [id, v] of Object.entries(
    (raw ?? {}) as Record<string, unknown>,
  )) {
    const x = v as Partial<NpcSummary> | null;
    if (x && typeof x.text === "string" && typeof x.upto === "number")
      out[id] = { text: x.text, upto: x.upto };
  }
  return out;
}

export function deserialize(spec: WorldSpec, raw: unknown): WorldState | null {
  const v = (raw as WorldState | null)?.v;
  if (!raw || typeof raw !== "object" || (v !== 1 && v !== WORLD_SAVE_VERSION))
    return null;
  const r = raw as Partial<WorldState>;
  const additions = additionsOf(r.additions, spec);
  const played = withAdditions(spec, additions);
  const itemIds = new Set(played.items.map((i) => i.id));
  const questIds = new Set(played.quests.map((q) => q.id));
  const fresh = initialWorldState(played);
  const city = cityOf(played) ? parseCity(r.city, played) : null;
  const npc: Record<string, NpcMind> = {};
  for (const id of Object.keys(fresh.npc))
    npc[id] = r.npc?.[id] ? mindOf(r.npc[id], fresh.npc[id]) : fresh.npc[id];
  const npcIds = new Set(played.npcs.map((n) => n.id));
  const places = placeIds(spec);
  const errands: Record<string, Errand> = {};
  for (const [id, e] of Object.entries(
    r.errands && typeof r.errands === "object" ? r.errands : {},
  ))
    if (npcIds.has(id) && e && places.has(e.place))
      errands[id] = {
        place: e.place,
        until: typeof e.until === "number" ? e.until : null,
      };
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
    summaries: summariesOf(r.summaries),
    clock: num(r.clock, fresh.clock),
    npc,
    following: (Array.isArray(r.following) ? r.following : []).filter((id) =>
      npcIds.has(id),
    ),
    errands,
    additions,
    ...(autonomyOn(played)
      ? { auto: autoOf(r.auto, played), npcAt: npcAtOf(r.npcAt, npcIds) }
      : {}),
    ...(played.action?.movement && r.action && typeof r.action === "object"
      ? { action: r.action }
      : {}),
    ...(city ? { city } : {}),
  };
}

function npcAtOf(raw: unknown, ids: Set<string>): Record<string, Point> {
  const out: Record<string, Point> = {};
  for (const [id, p] of Object.entries((raw ?? {}) as Record<string, Point>))
    if (ids.has(id) && p && typeof p.x === "number" && typeof p.y === "number")
      out[id] = { x: p.x, y: p.y };
  return out;
}
