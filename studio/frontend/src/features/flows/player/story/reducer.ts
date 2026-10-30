// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


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
  extraScenes: Record<string, StoryScene>;
  backgrounds: Record<string, MediaValue>;
  pending: { sceneId: string; choice: string } | null;
  error: string | null;
  rev: number;
  seed: number;
  // Sounds from direction lines passed while leaving a scene; stageAt cues them at the next stop.
  cues?: string[];
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

export function isDirection(line: StoryLine | undefined): boolean {
  return Boolean(line && line.text === "" && line.stage);
}

export function pauseOf(line: StoryLine | null | undefined): number | null {
  const p = line && isDirection(line) ? line.stage?.pause : null;
  return typeof p === "number" ? p : null;
}

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

function settled(s: StoryState): StoryState {
  if (!s.cues) return s;
  const rest = { ...s };
  delete rest.cues;
  return rest;
}

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
  if (scene.generate_lines && m.entry_points?.next_scene)
    return { ...s, status: "reading" };
  return leave(m, s, scene, hops + 1, 0);
}

function leave(
  m: StoryManifest,
  here: StoryState,
  scene: StoryScene,
  hops: number,
  from: number,
): StoryState {
  const visible = visibleChoices(scene, here.vars);
  if (visible.length > 0) return { ...here, status: "choosing" };
  const passed = [
    ...(here.cues ?? []),
    ...visibleLines(scene, here.vars)
      .slice(from)
      .flatMap((l) => (l.stage?.sound ? [l.stage.sound] : [])),
  ];
  const s = passed.length > 0 ? { ...here, cues: passed } : here;
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
      if (lines.length === 0 && scene.generate_lines) return s;
      const next = restAt(lines, s.line + 1);
      if (next < lines.length) {
        return {
          ...settled(s),
          line: next,
          backlog: withBacklog(s, scene, lines[next]),
        };
      }
      return leave(m, settled(s), scene, 0, s.line + 1);
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
      return take(m, { ...settled(s), backlog: [...s.backlog, picked] }, choice, 0);
    }
    case "jump":
      return enter(m, settled(s), a.sceneId, 0);
    case "set":
      return { ...s, vars: { ...s.vars, ...a.vars } };
    case "load": {
      if (!saveFits(m, a.state)) return s;
      const loaded = { ...settled(a.state), rev: s.rev + 1 };
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
        ...settled(s),
        extraScenes: { ...s.extraScenes, [a.asId]: scene },
      };
      return enter(m, withScene, a.asId, 0);
    }
    case "generationFailed":
      if (a.rev !== s.rev || s.status !== "generating") return s;
      return fail(settled(s), a.message);
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
      if (here) {
        const old = visibleLines(scene, s.vars);
        const shown = Boolean(old[restAt(old, 0)]?.text);
        const backlog = shown ? s.backlog.slice(0, -1) : s.backlog;
        const lines = visibleLines(replaced, s.vars);
        const first = Math.min(restAt(lines, 0), Math.max(0, lines.length - 1));
        return {
          ...settled(next),
          line: first,
          backlog: withBacklog({ ...next, backlog }, replaced, lines[first]),
        };
      }
      return next;
    }
  }
}

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
  leaving: (StageSlot & { exit: StoryTransition | null })[];
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

export function stageAt(m: StoryManifest, s: StoryState): StageView {
  const scene = sceneOf(m, s, s.sceneId);
  const carried = s.cues ?? [];
  if (!scene) return { cast: [], leaving: [], sounds: carried };
  const lines = visibleLines(scene, s.vars);
  const upto = s.status === "choosing" ? lines.length - 1 : s.line;
  let prev = -1;
  const stop = (i: number) =>
    i < lines.length && (!isDirection(lines[i]) || pauseOf(lines[i]) !== null);
  if (s.status !== "reading" && stop(s.line)) {
    prev = s.line;
  } else {
    for (let i = 0; i < upto; i++) if (stop(i)) prev = i;
  }
  const before = castAfter(scene, lines, prev);
  const now = castAfter(scene, lines, upto);
  const sounds = [...carried];
  const shown = new Map<string, StoryTransition | null>();
  const hidden = new Map<string, StoryTransition | null>();
  for (let i = prev + 1; i <= upto && i < lines.length; i++) {
    const st = lines[i].stage;
    // An ending's own direction lines are already in carried.
    if (st?.sound && s.status !== "ended") sounds.push(st.sound);
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
