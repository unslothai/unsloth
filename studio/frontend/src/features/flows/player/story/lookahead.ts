// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import type { StoryCharacter, StoryManifest, StoryScene } from "../../types";
import {
  CHOICE_SPEAKER,
  type StoryState,
  reduce,
  sceneOf,
  visibleChoices,
} from "./reducer";

export type NextSceneRequest = {
  state: {
    story: {
      title: string;
      style: StoryManifest["style"];
      characters: Array<Pick<StoryCharacter, "id" | "name" | "look">>;
      scenes: Array<{ id: string }>;
    };
    scene_id: string;
    vars: Record<string, unknown>;
    history: string[];
    choice: string;
  };
};

export interface PlannedWrite {
  key: string;
  req: NextSceneRequest;
}

export const MAX_WRITE_AHEAD_DEPTH = 2;
export const MAX_WRITE_AHEAD = 6;

export function storyContext(m: StoryManifest, s: StoryState) {
  return {
    title: m.title,
    style: m.style,
    characters: m.characters.map(({ id, name, look }) => ({ id, name, look })),
    scenes: [
      ...m.scenes.map((x) => ({ id: x.id })),
      ...Object.keys(s.extraScenes).map((id) => ({ id })),
    ],
  };
}

export function historyText(m: StoryManifest, s: StoryState): string[] {
  const names = new Map(m.characters.map((c) => [c.id, c.name]));
  return s.backlog.slice(-14).map((b) => {
    if (b.speaker === CHOICE_SPEAKER) return `(Player chose: ${b.text})`;
    const who = b.speaker ? (names.get(b.speaker) ?? b.speaker) : "Narrator";
    return `${who}: ${b.text}`;
  });
}

export function nextSceneRequest(
  m: StoryManifest,
  s: StoryState,
): NextSceneRequest | null {
  if (s.status !== "generating" || !s.pending) return null;
  return {
    state: {
      story: storyContext(m, s),
      scene_id: s.sceneId,
      vars: s.vars,
      history: historyText(m, s),
      choice: s.pending.choice,
    },
  };
}

function stable(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(stable).join(",")}]`;
  if (v && typeof v === "object") {
    const o = v as Record<string, unknown>;
    return `{${Object.keys(o)
      .filter((k) => o[k] !== undefined)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stable(o[k])}`)
      .join(",")}}`;
  }
  return JSON.stringify(v) ?? "null";
}

export function writeKey(s: StoryState, req: NextSceneRequest): string {
  return `${s.rev}|${s.pending?.sceneId ?? ""}|${stable(req)}`;
}

export function readToStop(m: StoryManifest, from: StoryState): StoryState {
  let s = from;
  for (let i = 0; i < 1000 && s.status === "reading"; i++) {
    const next = reduce(m, s, { type: "advance" });
    if (next === s) break;
    s = next;
  }
  return s;
}

function branches(m: StoryManifest, s: StoryState): StoryState[] {
  const at = readToStop(m, s);
  if (at.status === "generating") return [at];
  if (at.status !== "choosing") return [];
  const scene = sceneOf(m, at, at.sceneId);
  if (!scene) return [];
  return visibleChoices(scene, at.vars)
    .map((c) => reduce(m, at, { type: "choose", choiceId: c.id }))
    .filter((x) => x.status === "generating");
}

export function planWriteAhead(
  m: StoryManifest,
  s: StoryState,
  depth: number,
  written: (key: string) => StoryScene | undefined,
): PlannedWrite[] {
  if (!m.entry_points?.next_scene || depth < 1) return [];
  if (s.status !== "reading" && s.status !== "choosing") return [];
  const out: PlannedWrite[] = [];
  const seen = new Set<string>();
  let level = branches(m, s);
  for (let d = 1; d <= Math.min(depth, MAX_WRITE_AHEAD_DEPTH); d++) {
    const below: StoryState[] = [];
    for (const b of level) {
      const req = nextSceneRequest(m, b);
      if (!req || !b.pending) continue;
      const key = writeKey(b, req);
      if (seen.has(key)) continue;
      if (out.length >= MAX_WRITE_AHEAD) return out;
      seen.add(key);
      out.push({ key, req });
      const scene = written(key);
      if (scene && d < depth)
        below.push(
          ...branches(
            m,
            reduce(m, b, {
              type: "addScene",
              scene,
              asId: b.pending.sceneId,
              rev: b.rev,
            }),
          ),
        );
    }
    level = below;
  }
  return out;
}

type Write = (
  req: NextSceneRequest,
  signal: AbortSignal,
) => Promise<StoryScene>;

interface Job extends PlannedWrite {
  scene?: StoryScene;
  failed?: boolean;
  running?: Promise<StoryScene>;
  ctl?: AbortController;
  adopted?: boolean;
}

export class WriteAhead {
  private jobs = new Map<string, Job>();
  private queue: string[] = [];
  private current: Job | null = null;
  private live = new Set<AbortController>();
  private busy = 0;
  private closed = false;
  readonly stats = { started: 0, hits: 0, misses: 0, dropped: 0 };
  private write: Write;
  private onWritten?: (key: string) => void;

  constructor(write: Write, onWritten?: (key: string) => void) {
    this.write = write;
    this.onWritten = onWritten;
  }

  written(key: string): StoryScene | undefined {
    return this.jobs.get(key)?.scene;
  }

  plan(writes: PlannedWrite[]): void {
    if (this.closed) return;
    const wanted = new Set(writes.map((w) => w.key));
    for (const [key, job] of this.jobs) {
      if (wanted.has(key) || job.adopted) continue;
      job.ctl?.abort();
      this.jobs.delete(key);
      this.stats.dropped++;
    }
    for (const w of writes)
      if (!this.jobs.has(w.key)) this.jobs.set(w.key, { ...w });
    this.queue = writes
      .map((w) => w.key)
      .filter((k) => {
        const j = this.jobs.get(k);
        return j && !j.scene && !j.failed && !j.running;
      });
    this.pump();
  }

  async take(
    req: NextSceneRequest,
    key: string,
    signal?: AbortSignal,
  ): Promise<StoryScene> {
    this.busy++;
    let ctl: AbortController | undefined;
    try {
      const job = this.jobs.get(key);
      if (job?.scene) {
        this.stats.hits++;
        this.jobs.delete(key);
        return job.scene;
      }
      if (job?.running) {
        this.stats.hits++;
        job.adopted = true;
        try {
          return await job.running;
        } finally {
          this.jobs.delete(key);
        }
      }
      this.stats.misses++;
      this.jobs.delete(key);
      this.preempt();
      ctl = new AbortController();
      const live = ctl;
      if (signal?.aborted) live.abort();
      else
        signal?.addEventListener("abort", () => live.abort(), { once: true });
      this.live.add(live);
      return await this.write(req, live.signal);
    } finally {
      // reset() already released the hold of a live miss it cancelled.
      if (!ctl || this.live.delete(ctl)) this.busy--;
    }
  }

  async hold<T>(run: () => Promise<T>, preempt = true): Promise<T> {
    this.busy++;
    if (preempt) this.preempt();
    try {
      return await run();
    } finally {
      this.busy--;
      this.pump();
    }
  }

  reset(): void {
    for (const job of this.jobs.values()) job.ctl?.abort();
    for (const ctl of this.live) ctl.abort();
    this.busy -= this.live.size;
    this.live.clear();
    this.jobs.clear();
    this.queue = [];
    this.current = null;
  }

  open(): void {
    this.closed = false;
  }

  close(): void {
    this.closed = true;
    this.reset();
  }

  private preempt(): void {
    const job = this.current;
    if (!job || job.adopted) return;
    job.ctl?.abort();
  }

  private pump(): void {
    if (this.closed || this.busy > 0 || this.current) return;
    let job: Job | undefined;
    while (this.queue.length && !job) {
      const j = this.jobs.get(this.queue.shift() as string);
      if (j && !j.scene && !j.failed && !j.running) job = j;
    }
    if (!job) return;
    const ctl = new AbortController();
    const run = job;
    run.ctl = ctl;
    this.current = run;
    this.stats.started++;
    run.running = this.write(run.req, ctl.signal);
    run.running
      .then((scene) => {
        if (this.jobs.get(run.key) !== run) return;
        run.scene = scene;
        this.onWritten?.(run.key);
      })
      .catch(() => {
        if (this.jobs.get(run.key) !== run) return;
        if (ctl.signal.aborted && !run.adopted) this.queue.unshift(run.key);
        else run.failed = true;
      })
      .finally(() => {
        run.running = undefined;
        run.ctl = undefined;
        if (this.current === run) this.current = null;
        this.pump();
      });
  }
}

export class Lanes {
  private tails: Record<"text" | "art", Promise<unknown>> = {
    text: Promise.resolve(),
    art: Promise.resolve(),
  };

  run<T>(lane: "text" | "art", job: () => Promise<T>): Promise<T> {
    const next = this.tails[lane].then(job, job);
    this.tails[lane] = next.catch(() => undefined);
    return next;
  }
}
