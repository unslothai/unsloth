// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The story player's state machine: pure, deterministic and serializable (save slots store `StoryState`).
// Scene conventions match studio/backend/core/flows/story.py:
// - a scene with no visible lines is a router: on_enter, then `random`, the first passing empty-text choice, or `next`;
// - a scene with no way out and no `ending` ends the story ("The End");
// - a choice whose `goto` is not a known scene asks the Next Scene entry point to write it (infinite mode).

import {
  ExprError,
  Prng,
  applySets,
  evalExpr,
  fmt,
  seedFrom,
  truthy,
} from "../../expr";
import type {
  MediaValue,
  StoryChoice,
  StoryLine,
  StoryManifest,
  StoryPosition,
  StoryScene,
  StoryTransition,
} from "../../types";

export const STORY_SAVE_VERSION = 1;
const MAX_HOPS = 200;
const MAX_BACKLOG = 300;
/** Backlog entries for the player's own picks use this speaker. */
export const CHOICE_SPEAKER = "__choice__";

export type StoryStatus =
  | "reading"
  | "choosing"
  | "generating"
  | "ended"
  | "error";

export interface BacklogEntry {
  sceneId: string;
  speaker: string | null;
  text: string;
}

export interface StoryState {
  v: number;
  sceneId: string;
  line: number;
  vars: Record<string, unknown>;
  rng: number;
  status: StoryStatus;
  ending: { title: string; kind: string } | null;
  backlog: BacklogEntry[];
  visited: string[];
  /** Scenes written live (infinite mode) and backgrounds painted live, keyed by scene id. */
  extraScenes: Record<string, StoryScene>;
  backgrounds: Record<string, MediaValue>;
  /** Set while waiting for the Next Scene entry point: the scene id to fill and the choice that led there. */
  pending: { sceneId: string; choice: string } | null;
  error: string | null;
  /** Bumped on restart/load so late live results for an older playthrough are dropped. */
  rev: number;
  seed: number;
}

export type StoryAction =
  | { type: "start"; seed?: number }
  | { type: "advance" }
  | { type: "choose"; choiceId: string }
  | { type: "jump"; sceneId: string }
  | { type: "set"; vars: Record<string, unknown> }
  | { type: "load"; state: StoryState }
  | { type: "addScene"; scene: StoryScene; asId: string; rev: number }
  | { type: "generationFailed"; message: string; rev: number }
  | { type: "setBackground"; sceneId: string; media: MediaValue; rev: number }
  | { type: "setLines"; sceneId: string; lines: StoryLine[]; rev: number }
  | { type: "linesFailed"; sceneId: string; rev: number };

export function sceneOf(
  m: StoryManifest,
  s: StoryState,
  id: string,
): StoryScene | undefined {
  return s.extraScenes[id] ?? m.scenes.find((x) => x.id === id);
}

function cond(e: unknown, vars: Record<string, unknown>): boolean {
  if (e === undefined || e === null) return true;
  try {
    return truthy(evalExpr(e, vars));
  } catch {
    return false;
  }
}

export function visibleLines(
  scene: StoryScene,
  vars: Record<string, unknown>,
): StoryLine[] {
  return (scene.lines ?? []).filter((l) => cond(l.if, vars));
}

/** A text-less line of stage directions: applied on the way past, never shown. */
export function isDirection(line: StoryLine | undefined): boolean {
  return Boolean(line && line.text === "" && line.stage);
}

/** Seconds a `[pause n]` line holds the reader, or null. */
export function pauseOf(line: StoryLine | null | undefined): number | null {
  const p = line && isDirection(line) ? line.stage?.pause : null;
  return typeof p === "number" ? p : null;
}

/** The first line at or after `from` the reader stops on: text or a pause. `lines.length` if none. */
export function restAt(lines: StoryLine[], from: number): number {
  let i = Math.max(0, from);
  while (
    i < lines.length &&
    isDirection(lines[i]) &&
    pauseOf(lines[i]) === null
  )
    i++;
  return i;
}

/** The reader has not moved past the scene's first stop (directions before it are not a stop). */
export function atSceneStart(m: StoryManifest, s: StoryState): boolean {
  const scene = sceneOf(m, s, s.sceneId);
  return !scene || s.line <= restAt(visibleLines(scene, s.vars), 0);
}

export function visibleChoices(
  scene: StoryScene,
  vars: Record<string, unknown>,
): StoryChoice[] {
  return (scene.choices ?? []).filter((c) => c.text !== "" && cond(c.if, vars));
}

/** `{{name}}` -> the variable's value. */
export function interpolate(
  text: string,
  vars: Record<string, unknown>,
): string {
  return text.replace(/\{\{\s*([A-Za-z_][\w.]*)\s*\}\}/g, (_, name: string) => {
    const v = evalExpr({ var: name }, vars);
    return v === null ? "" : fmt(v);
  });
}

export function initialState(m: StoryManifest, seed = 0): StoryState {
  const base: StoryState = {
    v: STORY_SAVE_VERSION,
    sceneId: m.start,
    line: 0,
    vars: { ...(m.vars ?? {}) },
    rng: seedFrom("story", m.title, seed),
    status: "reading",
    ending: null,
    backlog: [],
    visited: [],
    extraScenes: {},
    backgrounds: {},
    pending: null,
    error: null,
    rev: 0,
    seed,
  };
  return enter(m, base, m.start, 0);
}

function withBacklog(
  s: StoryState,
  scene: StoryScene,
  line: StoryLine | undefined,
): BacklogEntry[] {
  if (!line?.text) return s.backlog;
  const next = [
    ...s.backlog,
    {
      sceneId: scene.id,
      speaker: line.speaker ?? null,
      text: interpolate(line.text, s.vars),
    },
  ];
  return next.length > MAX_BACKLOG
    ? next.slice(next.length - MAX_BACKLOG)
    : next;
}

function applyOrFail(
  s: StoryState,
  sets: Record<string, unknown> | undefined,
  rng: Prng,
): StoryState | string {
  if (!sets || Object.keys(sets).length === 0) return s;
  try {
    return {
      ...s,
      vars: applySets(sets as never, s.vars, rng),
      rng: rng.state,
    };
  } catch (e) {
    return e instanceof ExprError ? e.message : String(e);
  }
}

function fail(s: StoryState, message: string): StoryState {
  return { ...s, status: "error", error: message };
}

function end(
  s: StoryState,
  ending: { title: string; kind: string } | null,
): StoryState {
  return {
    ...s,
    status: "ended",
    ending: ending ?? { title: "The End", kind: "neutral" },
    pending: null,
  };
}

/** Enter `sceneId`: apply on_enter, then either start reading or route onwards (routers chain up to MAX_HOPS). */
function enter(
  m: StoryManifest,
  s0: StoryState,
  sceneId: string,
  hops: number,
): StoryState {
  if (hops > MAX_HOPS)
    return fail(s0, "The story loops forever without showing anything.");
  const scene = sceneOf(m, s0, sceneId);
  if (!scene) {
    if (m.mode === "infinite" && m.entry_points?.next_scene) {
      return {
        ...s0,
        sceneId,
        line: 0,
        status: "generating",
        pending: { sceneId, choice: s0.pending?.choice ?? "" },
      };
    }
    return fail(
      s0,
      `The story points to a scene that does not exist (${sceneId}).`,
    );
  }
  const rng = new Prng(s0.rng);
  let s: StoryState = {
    ...s0,
    sceneId,
    line: 0,
    pending: null,
    error: null,
    visited: s0.visited.includes(sceneId)
      ? s0.visited
      : [...s0.visited, sceneId],
  };
  const applied = applyOrFail(
    s,
    scene.on_enter as Record<string, unknown> | undefined,
    rng,
  );
  if (typeof applied === "string") return fail(s, applied);
  s = applied;
  const lines = visibleLines(scene, s.vars);
  const first = restAt(lines, 0);
  if (first < lines.length) {
    return {
      ...s,
      line: first,
      status: "reading",
      backlog: withBacklog(s, scene, lines[first]),
    };
  }
  // Its dialogue is being written: wait on the scene (shown as "...") instead of skipping past it.
  if (scene.generate_lines && m.entry_points?.next_scene)
    return { ...s, status: "reading" };
  return leave(m, s, scene, hops + 1);
}

/** Leave the current scene after its last line (or at once for a router). */
function leave(
  m: StoryManifest,
  s: StoryState,
  scene: StoryScene,
  hops: number,
): StoryState {
  const visible = visibleChoices(scene, s.vars);
  if (visible.length > 0) return { ...s, status: "choosing" };
  const routes = (scene.choices ?? []).filter((c) => c.text === "");
  for (const c of routes) {
    if (cond(c.if, s.vars)) return take(m, s, c, hops);
  }
  const random = (scene.random ?? []).filter((r) => r.weight > 0);
  if (random.length > 0) {
    const rng = new Prng(s.rng);
    const total = random.reduce((t, r) => t + r.weight, 0);
    let roll = rng.random() * total;
    let pick = random[random.length - 1];
    for (const r of random) {
      roll -= r.weight;
      if (roll < 0) {
        pick = r;
        break;
      }
    }
    return enter(m, { ...s, rng: rng.state }, pick.goto, hops);
  }
  if (scene.next) return enter(m, s, scene.next, hops);
  return end(s, scene.ending ?? null);
}

function take(
  m: StoryManifest,
  s: StoryState,
  choice: StoryChoice,
  hops: number,
): StoryState {
  const rng = new Prng(s.rng);
  const applied = applyOrFail(
    s,
    choice.set as Record<string, unknown> | undefined,
    rng,
  );
  if (typeof applied === "string") return fail(s, applied);
  const next = {
    ...applied,
    pending: choice.text ? { sceneId: choice.goto, choice: choice.text } : null,
  };
  if (
    !sceneOf(m, next, choice.goto) &&
    m.mode === "infinite" &&
    m.entry_points?.next_scene
  ) {
    return {
      ...next,
      status: "generating",
      pending: { sceneId: choice.goto, choice: choice.text },
    };
  }
  return enter(m, next, choice.goto, hops);
}

export function reduce(
  m: StoryManifest,
  s: StoryState,
  a: StoryAction,
): StoryState {
  switch (a.type) {
    case "start": {
      const fresh = initialState(m, a.seed ?? s.seed);
      return { ...fresh, rev: s.rev + 1 };
    }
    case "advance": {
      if (s.status !== "reading") return s;
      const scene = sceneOf(m, s, s.sceneId);
      if (!scene) return fail(s, "Missing scene");
      const lines = visibleLines(scene, s.vars);
      // Its dialogue is still being written: the "..." placeholder cannot be read past.
      if (lines.length === 0 && scene.generate_lines) return s;
      const next = restAt(lines, s.line + 1);
      if (next < lines.length) {
        return {
          ...s,
          line: next,
          backlog: withBacklog(s, scene, lines[next]),
        };
      }
      return leave(m, s, scene, 0);
    }
    case "choose": {
      if (s.status !== "choosing") return s;
      const scene = sceneOf(m, s, s.sceneId);
      const choice = scene
        ? visibleChoices(scene, s.vars).find((c) => c.id === a.choiceId)
        : undefined;
      if (!choice) return s;
      const picked = {
        sceneId: s.sceneId,
        speaker: CHOICE_SPEAKER,
        text: choice.text,
      };
      return take(m, { ...s, backlog: [...s.backlog, picked] }, choice, 0);
    }
    case "jump":
      return enter(m, s, a.sceneId, 0);
    case "set":
      return { ...s, vars: { ...s.vars, ...a.vars } };
    case "load": {
      if (!saveFits(m, a.state)) return s;
      const loaded = { ...a.state, rev: s.rev + 1 };
      // A save from before the story gained directions may point at one: move to the next readable line.
      const scene = sceneOf(m, loaded, loaded.sceneId);
      if (!scene || loaded.status !== "reading") return loaded;
      const lines = visibleLines(scene, loaded.vars);
      const at = restAt(lines, loaded.line);
      return at < lines.length ? { ...loaded, line: at } : loaded;
    }
    case "addScene": {
      if (
        a.rev !== s.rev ||
        s.status !== "generating" ||
        s.pending?.sceneId !== a.asId
      )
        return s;
      const scene: StoryScene = { ...a.scene, id: a.asId };
      const withScene = {
        ...s,
        extraScenes: { ...s.extraScenes, [a.asId]: scene },
      };
      return enter(m, withScene, a.asId, 0);
    }
    case "generationFailed":
      if (a.rev !== s.rev || s.status !== "generating") return s;
      return fail(s, a.message);
    case "setBackground":
      if (a.rev !== s.rev) return s;
      return { ...s, backgrounds: { ...s.backgrounds, [a.sceneId]: a.media } };
    case "linesFailed": {
      if (a.rev !== s.rev) return s;
      const scene = sceneOf(m, s, a.sceneId);
      if (!scene?.generate_lines) return s;
      return {
        ...s,
        extraScenes: {
          ...s.extraScenes,
          [a.sceneId]: { ...scene, generate_lines: false },
        },
      };
    }
    case "setLines": {
      if (a.rev !== s.rev) return s;
      const scene = sceneOf(m, s, a.sceneId);
      if (!scene || a.lines.length === 0) return s;
      // Mid-scene the reader's line index belongs to the current list; the next visit writes fresh lines.
      const here = s.sceneId === a.sceneId && s.status === "reading";
      if (here && !atSceneStart(m, s)) return s;
      const replaced: StoryScene = {
        ...scene,
        lines: a.lines,
        generate_lines: false,
      };
      const next = {
        ...s,
        extraScenes: { ...s.extraScenes, [a.sceneId]: replaced },
      };
      // Only restart the scene if the reader is still on its first line.
      if (here) {
        const old = visibleLines(scene, s.vars);
        const shown = Boolean(old[restAt(old, 0)]?.text);
        const backlog = shown ? s.backlog.slice(0, -1) : s.backlog;
        const lines = visibleLines(replaced, s.vars);
        const first = Math.min(restAt(lines, 0), Math.max(0, lines.length - 1));
        return {
          ...next,
          line: first,
          backlog: withBacklog({ ...next, backlog }, replaced, lines[first]),
        };
      }
      return next;
    }
  }
}

/** The scene ids reachable in one step from `sceneId` (for prefetching their art). */
export function nextSceneIds(
  m: StoryManifest,
  s: StoryState,
  sceneId: string,
): string[] {
  const scene = sceneOf(m, s, sceneId);
  if (!scene) return [];
  const ids = new Set<string>();
  for (const c of scene.choices ?? []) ids.add(c.goto);
  for (const r of scene.random ?? []) ids.add(r.goto);
  if (scene.next) ids.add(scene.next);
  return [...ids].filter((id) => sceneOf(m, s, id));
}

/** A save made for this story: right version, and its scene still exists (a re-roll can rename scenes). */
export function saveFits(
  m: StoryManifest,
  state: StoryState | null | undefined,
): boolean {
  if (state?.v !== STORY_SAVE_VERSION) return false;
  if (state.status === "ended" || state.status === "generating") return true;
  return Boolean(sceneOf(m, state, state.sceneId));
}

export function currentLine(m: StoryManifest, s: StoryState): StoryLine | null {
  const scene = sceneOf(m, s, s.sceneId);
  if (!scene) return null;
  const line = visibleLines(scene, s.vars)[s.line];
  if (!line && scene.generate_lines && s.status === "reading")
    return { text: "..." };
  return line ?? null;
}

/** The nearest scene at or before the current one that has a background (routers have none). */
export function backgroundFor(
  m: StoryManifest,
  s: StoryState,
  sceneId: string,
): MediaValue | null {
  const scene = sceneOf(m, s, sceneId);
  return s.backgrounds[sceneId] ?? scene?.background ?? null;
}

export interface StageSlot {
  id: string;
  at: StoryPosition | null;
  expression: string | null;
}

export interface StageView {
  cast: (StageSlot & { enter: StoryTransition | null })[];
  /** Taken off since the previous stop, kept for the exit transition. */
  leaving: (StageSlot & { exit: StoryTransition | null })[];
  /** `[sound]` cues reached since the previous stop. */
  sounds: string[];
}

function castAfter(
  scene: StoryScene,
  lines: StoryLine[],
  last: number,
): Map<string, StageSlot> {
  const cast = new Map<string, StageSlot>(
    (scene.on_stage ?? []).map((id) => [
      id,
      { id, at: null, expression: null },
    ]),
  );
  for (let i = 0; i <= last && i < lines.length; i++) {
    const st = lines[i].stage;
    if (!st) continue;
    if (st.hide) cast.delete(st.hide);
    if (st.show) {
      const prev = cast.get(st.show);
      cast.set(st.show, {
        id: st.show,
        at: st.at ?? prev?.at ?? null,
        expression: st.expression ?? prev?.expression ?? null,
      });
    }
  }
  return cast;
}

/** Who is on stage at the reader's stop: the scene's `on_stage` plus the directions read so far.
 * Derived, so saves carry no stage state and saves from before directions existed still render. */
export function stageAt(m: StoryManifest, s: StoryState): StageView {
  const scene = sceneOf(m, s, s.sceneId);
  if (!scene) return { cast: [], leaving: [], sounds: [] };
  const lines = visibleLines(scene, s.vars);
  const upto = s.status === "choosing" ? lines.length - 1 : s.line;
  let prev = -1;
  const stop = (i: number) =>
    i < lines.length && (!isDirection(lines[i]) || pauseOf(lines[i]) !== null);
  if (s.status !== "reading" && stop(s.line)) {
    // Choices or the ending open after the last line was shown: its cues already played there.
    prev = s.line;
  } else {
    for (let i = 0; i < upto; i++) if (stop(i)) prev = i;
  }
  const before = castAfter(scene, lines, prev);
  const now = castAfter(scene, lines, upto);
  const sounds: string[] = [];
  const shown = new Map<string, StoryTransition | null>();
  const hidden = new Map<string, StoryTransition | null>();
  for (let i = prev + 1; i <= upto && i < lines.length; i++) {
    const st = lines[i].stage;
    if (st?.sound) sounds.push(st.sound);
    if (st?.show) shown.set(st.show, st.with ?? null);
    if (st?.hide) hidden.set(st.hide, st.with ?? null);
  }
  return {
    cast: [...now.values()].map((c) => ({
      ...c,
      enter: before.has(c.id) ? null : (shown.get(c.id) ?? null),
    })),
    leaving: [...before.values()]
      .filter((c) => !now.has(c.id))
      .map((c) => ({ ...c, exit: hidden.get(c.id) ?? null })),
    sounds,
  };
}
