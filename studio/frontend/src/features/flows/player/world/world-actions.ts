// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import type { WorldSimulation, WorldSpec } from "../../types";
import {
  ACTIONS,
  type Doing,
  type Needs,
  type Offer,
  chooseAction,
  followArea,
  satisfied,
  seeded,
  tickNeeds,
} from "./needs";
import {
  activeOption,
  advanceOptions,
  critical,
  lifeBlock,
  optionDoing,
} from "./life-sim";
import type { Area, WorldGameHandle } from "./phaser-game";
import { type NpcMind, activityEmoji } from "./world-sim";
import type { WorldState } from "./world-state";

export type Steerable = Pick<
  WorldGameHandle,
  "playerTile" | "npcTile" | "setNpcPlace" | "setNpcStatus" | "say"
>;

export function placeAreas(spec: WorldSpec): Map<string, Area> {
  const out = new Map<string, Area>();
  for (const r of spec.regions ?? []) out.set(r.id, r);
  for (const p of spec.simulation?.places ?? []) out.set(p.id, p);
  return out;
}

const inside = (a: Area | undefined, t: { x: number; y: number } | null) =>
  !!a && !!t && t.x >= a.x && t.x < a.x + a.w && t.y >= a.y && t.y < a.y + a.h;

export function steerActions(
  g: Steerable,
  areas: Map<string, Area>,
  s: WorldState,
  prev: Set<string>,
): { steered: Set<string>; expired: string[] } {
  const steered = new Set<string>();
  const expired: string[] = [];
  const me = g.playerTile();
  for (const id of s.following ?? []) {
    steered.add(id);
    g.setNpcPlace(id, followArea(me));
    g.setNpcStatus(id, "👣");
  }
  for (const [id, e] of Object.entries(s.errands ?? {})) {
    if (steered.has(id)) continue;
    if (e.until !== null && s.clock >= e.until) {
      expired.push(id);
      continue;
    }
    steered.add(id);
    g.setNpcPlace(id, areas.get(e.place) ?? null);
    g.setNpcStatus(id, "📍");
  }
  for (const id of prev) {
    if (steered.has(id)) continue;
    g.setNpcPlace(id, null);
    g.setNpcStatus(id, "");
  }
  return { steered, expired };
}

export function villagerStep(opts: {
  g: Steerable;
  spec: WorldSpec;
  sim: WorldSimulation;
  npcId: string;
  mind: NpcMind;
  clock: number;
  offers: Offer[];
  areas: Map<string, Area>;
  held?: Record<string, string>;
}): (Partial<NpcMind> & { needs: Needs; doing: Doing }) | null {
  const { g, spec, sim, npcId, clock, offers, areas } = opts;
  if (!opts.mind.needs) return null;
  let doing = opts.mind.doing ?? null;
  const at = g.npcTile(npcId);
  const needs = tickNeeds(
    opts.mind.needs,
    sim.minutes_per_second,
    doing?.kind ?? null,
    inside(doing ? areas.get(doing.place) : undefined, at),
  );
  const moved = advanceOptions(opts.mind, {
    npcId,
    clock,
    needs,
    held: opts.held,
    at,
    areas,
  });
  const mind = moved ? { ...opts.mind, ...moved } : opts.mind;
  const option = activeOption(mind);
  if (!doing || clock >= doing.until || satisfied(doing, needs) || moved) {
    const block = lifeBlock(mind, clock);
    const rnd = seeded(spec.seed, npcId, Math.floor(clock));
    const next =
      option && !critical(needs)
        ? optionDoing(option, clock)
        : chooseAction({ needs, offers, clock, block, rnd });
    const lines = ACTIONS[next.kind].says;
    if (next.kind !== doing?.kind && lines.length && rnd() < 0.5)
      g.say(npcId, lines[Math.floor(rnd() * lines.length)], 3000);
    doing = next;
  }
  g.setNpcPlace(npcId, areas.get(doing.place) ?? null);
  g.setNpcStatus(
    npcId,
    doing.kind === "go_to"
      ? activityEmoji(doing.label ?? "")
      : ACTIONS[doing.kind].icon,
  );
  return { ...moved, needs, doing };
}

export function steeredNeeds(mind: NpcMind, minutes: number): Needs | null {
  return mind.needs ? tickNeeds(mind.needs, minutes, null, false) : null;
}
