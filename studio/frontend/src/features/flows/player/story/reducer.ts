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
  StoryScene,
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
  | { type: "setLines"; sceneId: string; lines: StoryLine[]; rev: number };

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
  if (!line) return s.backlog;
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
  if (lines.length > 0) {
    return {
      ...s,
      status: "reading",
      backlog: withBacklog(s, scene, lines[0]),
    };
  }
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
      if (s.line + 1 < lines.length) {
        return {
          ...s,
          line: s.line + 1,
          backlog: withBacklog(s, scene, lines[s.line + 1]),
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
    case "load":
      if (a.state?.v !== STORY_SAVE_VERSION) return s;
      return { ...a.state, rev: s.rev + 1 };
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
    case "setLines": {
      if (a.rev !== s.rev) return s;
      const scene = sceneOf(m, s, a.sceneId);
      if (!scene || a.lines.length === 0) return s;
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
      if (s.sceneId === a.sceneId && s.line === 0 && s.status === "reading") {
        const backlog = s.backlog.slice(0, -1);
        return {
          ...next,
          backlog: withBacklog(
            { ...next, backlog },
            replaced,
            visibleLines(replaced, s.vars)[0],
          ),
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

export function currentLine(m: StoryManifest, s: StoryState): StoryLine | null {
  const scene = sceneOf(m, s, s.sceneId);
  if (!scene) return null;
  return visibleLines(scene, s.vars)[s.line] ?? null;
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
