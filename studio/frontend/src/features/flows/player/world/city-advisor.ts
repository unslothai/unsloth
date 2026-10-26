// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mayor mode LLM boundary: the exact numbers the model sees, and what may come back (advice, petitions, one event
// from a fixed menu, one zone name). The model never places tiles.

import type { WorldSpec } from "../../types";
import {
  type CityEvent,
  type CitySim,
  EVENTS,
  type Petition,
  cityOf,
} from "./city-sim";

export const CITY_CALLS_PER_MINUTE = 2;

export const mayorOn = (spec: WorldSpec): boolean => cityOf(spec) !== null;
/** Background calls a minute for a city world that has no village simulation budget of its own. */
export const cityQueueRate = (spec: WorldSpec): number =>
  mayorOn(spec) ? CITY_CALLS_PER_MINUTE : 0;
const MAX_ZONES = 24;
const MAX_RESIDENTS = 8;

export interface ResidentDigest {
  id: string;
  name: string;
  persona: string;
  zone: string;
  land_value: number;
  pollution: number;
  crime: number;
  traffic: number;
  power: number;
  road: number;
}

export interface CityDigest {
  title: string;
  kind: "month" | "advice";
  petitions: boolean;
  events: boolean;
  month: number;
  funds: number;
  tax: number;
  tax_income: number;
  road_cost: number;
  cash_flow: number;
  residential_pop: number;
  commercial_pop: number;
  industrial_pop: number;
  total_pop: number;
  demand_r: number;
  demand_c: number;
  demand_i: number;
  land_value_avg: number;
  pollution_avg: number;
  crime_avg: number;
  power_capacity: number;
  power_used: number;
  trips_ok: number;
  trips_failed: number;
  trips_no_road: number;
  zones: {
    id: string;
    kind: string;
    pop: number;
    power: number;
    road: number;
    name: string;
  }[];
  residents: ResidentDigest[];
  unnamed: string;
}

export function cityDigest(
  spec: WorldSpec,
  sim: CitySim,
  kind: "month" | "advice",
): CityDigest {
  const c = cityOf(spec);
  const npcs = new Map(spec.npcs.map((n) => [n.id, n]));
  const residents: ResidentDigest[] = [];
  for (const [id, zoneId] of Object.entries(sim.residents)) {
    const z = sim.zoneById(zoneId);
    const n = npcs.get(id);
    if (!z || !n || residents.length >= MAX_RESIDENTS) continue;
    const at = sim.at(z.x, z.y);
    residents.push({
      id,
      name: n.name,
      persona: (n.persona || "").slice(0, 120),
      zone: z.id,
      land_value: at.land_value,
      pollution: at.pollution,
      crime: at.crime,
      traffic: sim.zoneTraffic(z),
      power: sim.isPowered(z) ? 1 : 0,
      road: sim.hasRoad(z) ? 1 : 0,
    });
  }
  const trips = sim.monthTrips;
  return {
    title: spec.title,
    kind,
    petitions: kind === "month" && c?.petitions !== false,
    events: kind === "month" && c?.events !== false,
    month: sim.month,
    funds: sim.funds,
    tax: sim.tax,
    tax_income: sim.cash.tax,
    road_cost: sim.cash.roads,
    cash_flow: sim.cash.flow,
    residential_pop: sim.census.res,
    commercial_pop: sim.census.com,
    industrial_pop: sim.census.ind,
    total_pop: sim.census.total,
    demand_r: sim.valves.r,
    demand_c: sim.valves.c,
    demand_i: sim.valves.i,
    land_value_avg: sim.census.lv,
    pollution_avg: sim.census.pollution,
    crime_avg: sim.census.crime,
    power_capacity: sim.powerMax,
    power_used: sim.powerUsed,
    trips_ok: trips.ok,
    trips_failed: trips.failed,
    trips_no_road: trips.noRoad,
    zones: sim.zones.slice(0, MAX_ZONES).map((z) => ({
      id: z.id,
      kind: z.kind,
      pop: z.pop,
      power: sim.isPowered(z) ? 1 : 0,
      road: sim.hasRoad(z) ? 1 : 0,
      name: z.name ?? "",
    })),
    residents,
    unnamed:
      kind === "month"
        ? (sim.zones.find((z) => !z.name && z.kind !== "P" && z.pop > 0)?.id ??
          "")
        : "",
  };
}

export function mayorRequest(digest: CityDigest): Record<string, unknown> {
  return { task: "mayor", city: digest };
}

export interface MayorReply {
  advice: string;
  petitions: Omit<Petition, "month">[];
  event: { kind: CityEvent; zone: string; size: number } | null;
  name: { zone: string; name: string } | null;
}

export function parseMayorReply(raw: unknown): MayorReply | null {
  const r = (raw as { mayor?: Partial<MayorReply> } | null)?.mayor;
  if (!r || typeof r.advice !== "string") return null;
  const petitions = (Array.isArray(r.petitions) ? r.petitions : [])
    .filter(
      (p) =>
        !!p &&
        typeof p.citizen === "string" &&
        typeof p.zone === "string" &&
        typeof p.metric === "string" &&
        typeof p.value === "number" &&
        typeof p.text === "string",
    )
    .slice(0, 2);
  const e = r.event;
  const event =
    e && EVENTS.includes(e.kind) && e.kind !== "none"
      ? {
          kind: e.kind,
          zone: typeof e.zone === "string" ? e.zone : "",
          size: typeof e.size === "number" ? e.size : 1,
        }
      : null;
  const n = r.name;
  const name =
    n && typeof n.zone === "string" && typeof n.name === "string" && n.name
      ? { zone: n.zone, name: n.name.slice(0, 40) }
      : null;
  return { advice: r.advice.slice(0, 600), petitions, event, name };
}

/** Apply a validated reply; returns the lines worth a toast. */
export function applyMayorReply(
  sim: CitySim,
  digest: CityDigest,
  reply: MayorReply,
): string[] {
  const out: string[] = [];
  if (reply.advice) sim.setAdvisor(reply.advice, "model");
  if (digest.kind !== "month") return out;
  const residents = new Map(digest.residents.map((r) => [r.id, r]));
  const petitions: Petition[] = [];
  for (const p of reply.petitions) {
    const who = residents.get(p.citizen);
    if (
      !who ||
      who.zone !== p.zone ||
      petitions.some((x) => x.citizen === p.citizen)
    )
      continue;
    petitions.push({ ...p, month: digest.month });
    out.push(`${who.name}: ${p.text}`);
  }
  if (petitions.length) sim.addPetitions(petitions);
  if (reply.event && digest.events) {
    const text = sim.applyEvent(
      reply.event.kind,
      reply.event.zone,
      reply.event.size,
    );
    if (text) out.push(text);
  }
  if (reply.name && reply.name.zone === digest.unnamed)
    sim.nameZone(reply.name.zone, reply.name.name);
  return out;
}

/** The advisor without a model: the same numbers, in a fixed sentence. */
export function localAdvice(d: CityDigest): string {
  const tips: string[] = [];
  const unpowered = d.zones.filter((z) => z.kind !== "P" && !z.power);
  const noRoad = d.zones.filter((z) => z.kind !== "P" && !z.road);
  if (!d.zones.some((z) => z.kind === "P"))
    tips.push("Build a power plant: nothing grows without power.");
  else if (unpowered.length)
    tips.push(
      `${unpowered.length} zone${unpowered.length > 1 ? "s have" : " has"} no power (${unpowered
        .slice(0, 3)
        .map((z) => z.id)
        .join(", ")}).`,
    );
  if (noRoad.length)
    tips.push(
      `${noRoad.length} zone${noRoad.length > 1 ? "s have" : " has"} no road (${noRoad
        .slice(0, 3)
        .map((z) => z.id)
        .join(", ")}).`,
    );
  const top = [
    ["residential", d.demand_r],
    ["commercial", d.demand_c],
    ["industrial", d.demand_i],
  ].sort((a, b) => (b[1] as number) - (a[1] as number))[0];
  if ((top[1] as number) > 0)
    tips.push(`Demand is highest for ${top[0]} (${top[1]}).`);
  tips.push(
    `Population ${d.total_pop}, funds ${d.funds}, cash flow ${d.cash_flow} a month.`,
  );
  return tips.join(" ");
}
