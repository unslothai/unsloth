// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import type { WorldPlace, WorldSimulation, WorldSpec } from "../../types";
import {
  type ActionKind,
  type Doing,
  type Need,
  type Needs,
  initialNeeds,
} from "./needs";

export const MAX_MEMORIES = 120;
export const MEET_TILES = 2;

export interface MemoryRecord {
  id: string;
  text: string;
  kind: "observation" | "reflection" | "plan";
  t_created: number;
  t_access: number;
  importance: number;
  /** Who passed this on, nearest first ("player" for the player). */
  source?: string[];
  /** Importance is a code guess still waiting for a batched rating. */
  guess?: boolean;
}

export interface PlanBlock {
  start: number;
  minutes: number;
  activity: string;
  place: string;
}

export interface NpcMind {
  memory: MemoryRecord[];
  nextId: number;
  plan: PlanBlock[];
  planDay: number;
  lastChat: Record<string, number>;
  needs?: Needs;
  doing?: Doing | null;
  options?: PlanOption[];
  optionAt?: number;
  optionSince?: number;
  planAt?: number;
  replan?: boolean;
  /** Bumped on every replan request, so a plan answer clears only the requests it saw. */
  replanSeq?: number;
  since?: number;
  rel?: Record<string, PairMemory>;
}

export type Until =
  | { type: "time"; at: number }
  | { type: "need"; need: Need; above: number }
  | { type: "item"; item: string }
  | { type: "place"; place: string };

export interface PlanOption {
  activity: string;
  place: string;
  kind: ActionKind;
  until: Until;
}

export interface PairMemory {
  affinity: number;
  facts: string[];
  talked?: number;
}

export interface Note {
  text: string;
  importance: number;
  source?: string[];
  guess?: boolean;
}

export const NOTABLE = 8;

export function simOf(spec: WorldSpec): WorldSimulation | null {
  const sim = spec.simulation;
  return sim?.enabled && sim.places.length ? sim : null;
}

export const needsOn = (spec: WorldSpec) =>
  !!simOf(spec) && spec.simulation?.needs !== false;

export function emptyMind(): NpcMind {
  return { memory: [], nextId: 0, plan: [], planDay: -1, lastChat: {} };
}

export function initialMinds(
  spec: WorldSpec,
  clock: number,
): Record<string, NpcMind> {
  const minds: Record<string, NpcMind> = {};
  for (const n of spec.npcs) {
    minds[n.id] = remember(emptyMind(), n.seed_memories ?? [], clock);
    if (needsOn(spec))
      minds[n.id] = { ...minds[n.id], needs: initialNeeds(spec.seed, n.id) };
  }
  return minds;
}

export function remember(
  mind: NpcMind,
  notes: Note[],
  clock: number,
  kind: MemoryRecord["kind"] = "observation",
): NpcMind {
  const fresh = notes.filter((n) => n.text?.trim());
  if (!fresh.length) return mind;
  let nextId = mind.nextId;
  const added = fresh.map((n) => ({
    id: `m${nextId++}`,
    text: n.text.trim().slice(0, 300),
    kind,
    t_created: clock / 60,
    t_access: clock / 60,
    importance: Math.max(1, Math.min(10, Math.round(n.importance || 4))),
    ...(n.source?.length ? { source: n.source.slice(0, 4) } : {}),
    ...(n.guess ? { guess: true } : {}),
  }));
  const life: Partial<NpcMind> = {};
  if (mind.since !== undefined)
    life.since =
      mind.since + added.reduce((a, r) => a + (r.guess ? 0 : r.importance), 0);
  if (
    mind.options &&
    kind === "observation" &&
    added.some((r) => r.importance >= NOTABLE)
  ) {
    life.replan = true;
    life.replanSeq = (mind.replanSeq ?? 0) + 1;
  }
  let memory = [...mind.memory, ...added];
  if (memory.length > MAX_MEMORIES) {
    const drop = new Set(
      [...memory]
        .sort((a, b) => a.importance - b.importance || a.t_access - b.t_access)
        .slice(0, memory.length - MAX_MEMORIES)
        .map((r) => r.id),
    );
    memory = memory.filter((r) => !drop.has(r.id));
  }
  return { ...mind, ...life, memory, nextId };
}

export function touch(mind: NpcMind, ids: string[], clock: number): NpcMind {
  if (!ids?.length) return mind;
  const hit = new Set(ids);
  return {
    ...mind,
    memory: mind.memory.map((r) =>
      hit.has(r.id) ? { ...r, t_access: clock / 60 } : r,
    ),
  };
}

export const dayOf = (clock: number) => Math.floor(clock / 1440);

export function clockLabel(clock: number): string {
  const m = Math.max(0, Math.floor(clock));
  const hh = String(Math.floor((m % 1440) / 60)).padStart(2, "0");
  const mm = String(m % 60).padStart(2, "0");
  return `Day ${dayOf(m) + 1} · ${hh}:${mm}`;
}

export function currentBlock(
  plan: PlanBlock[],
  clock: number,
): PlanBlock | null {
  return (
    plan.find((b) => clock >= b.start && clock < b.start + b.minutes) ?? null
  );
}

const EMOJI: [RegExp, string][] = [
  [/sleep|bed|nap|rest/i, "💤"],
  [/bak|cook|kitchen|bread/i, "🍞"],
  [/coffee|cafe|tea|breakfast|lunch|dinner|eat|meal/i, "☕"],
  [/party|celebrat|dance/i, "🎉"],
  [/fish|pier|boat/i, "🎣"],
  [/garden|farm|plant|water/i, "🌱"],
  [/read|study|book|write|paint|art/i, "📖"],
  [/shop|market|sell|buy|stall/i, "🛒"],
  [/walk|stroll|explore|jog/i, "🚶"],
  [/work|repair|build|forge|craft/i, "🔨"],
  [/chat|talk|visit|meet|gossip/i, "💬"],
];

export function activityEmoji(activity: string): string {
  return EMOJI.find(([re]) => re.test(activity))?.[1] ?? "";
}

export function placeAt(
  places: WorldPlace[],
  tile: { x: number; y: number },
): WorldPlace | null {
  return (
    places.find(
      (p) =>
        tile.x >= p.x &&
        tile.x < p.x + p.w &&
        tile.y >= p.y &&
        tile.y < p.y + p.h,
    ) ?? null
  );
}

export function chatReady(
  minds: Record<string, NpcMind>,
  a: string,
  b: string,
  clock: number,
  cooldown: number,
): boolean {
  const last = minds[a]?.lastChat[b] ?? minds[b]?.lastChat[a];
  return last === undefined || clock - last >= cooldown;
}

export function markChat(
  minds: Record<string, NpcMind>,
  a: string,
  b: string,
  clock: number,
): Record<string, NpcMind> {
  const ma = minds[a] ?? emptyMind();
  const mb = minds[b] ?? emptyMind();
  return {
    ...minds,
    [a]: { ...ma, lastChat: { ...ma.lastChat, [b]: clock } },
    [b]: { ...mb, lastChat: { ...mb.lastChat, [a]: clock } },
  };
}

export function findEncounter(
  tiles: Record<string, { x: number; y: number }>,
  busy: Set<string>,
  minds: Record<string, NpcMind>,
  clock: number,
  cooldown: number,
): [string, string] | null {
  const ids = Object.keys(tiles).filter((id) => !busy.has(id));
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = tiles[ids[i]];
      const b = tiles[ids[j]];
      if (
        Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y)) <= MEET_TILES &&
        chatReady(minds, ids[i], ids[j], clock, cooldown)
      )
        return [ids[i], ids[j]];
    }
  }
  return null;
}

export interface SimJob {
  key: string;
  kind: "plan" | "chat" | "rate" | "reflect";
  /** Resolves false when it answered without a model call (a plan cache hit): its slot is refunded. */
  run: (signal: AbortSignal) => Promise<void | boolean>;
}

export class BackgroundQueue {
  private jobs: SimJob[] = [];
  private starts: number[] = [];
  private current: { job: SimJob; ctrl: AbortController } | null = null;
  private paused = false;
  readonly calls: Record<string, number> = {};

  readonly perMinute: number;
  private now: () => number;

  constructor(perMinute: number, now: () => number = () => Date.now()) {
    this.perMinute = perMinute;
    this.now = now;
  }

  get busy(): SimJob | null {
    return this.current?.job ?? null;
  }

  get pending(): number {
    return this.jobs.length;
  }

  has(key: string): boolean {
    return (
      this.current?.job.key === key || this.jobs.some((j) => j.key === key)
    );
  }

  push(job: SimJob): void {
    if (this.has(job.key)) return;
    const at =
      job.kind === "plan" ? this.jobs.findIndex((j) => j.kind !== "plan") : -1;
    if (at < 0) this.jobs.push(job);
    else this.jobs.splice(at, 0, job);
  }

  setPaused(paused: boolean): void {
    this.paused = paused;
    if (paused) this.current?.ctrl.abort();
  }

  clear(): void {
    this.jobs = [];
    this.current?.ctrl.abort();
  }

  allows(): boolean {
    const t = this.now();
    this.starts = this.starts.filter((s) => t - s < 60_000);
    return this.starts.length < this.perMinute;
  }

  callsLabel(): string {
    const kinds = Object.entries(this.calls);
    const total = kinds.reduce((a, [, n]) => a + n, 0);
    return `${total} background calls${kinds.length ? ` (${kinds.map(([k, n]) => `${k} ${n}`).join(", ")})` : ""}`;
  }

  pump(): void {
    if (this.paused || this.current || !this.jobs.length || !this.allows())
      return;
    const job = this.jobs.shift() as SimJob;
    const ctrl = new AbortController();
    this.current = { job, ctrl };
    const started = this.now();
    this.starts.push(started);
    this.calls[job.kind] = (this.calls[job.kind] ?? 0) + 1;
    void job
      .run(ctrl.signal)
      .then((called) => {
        if (called !== false) return;
        this.calls[job.kind]--;
        const i = this.starts.indexOf(started);
        if (i >= 0) this.starts.splice(i, 1);
      })
      .catch(() => {
      })
      .finally(() => {
        if (this.current?.job === job) this.current = null;
        this.pump();
      });
  }
}
