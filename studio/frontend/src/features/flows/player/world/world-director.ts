// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Prng, seedFrom } from "../../expr";
import type { WorldItem, WorldNpc, WorldQuest, WorldSpec } from "../../types";
import { type Point, type WorldMap, isSolid, reachable } from "./procgen";
import { MAX_ADDED_ROLE, type NpcRole, ROLES, autonomyOf } from "./roles";
import type { WorldState } from "./world-state";
import { freshLife, lifeOf } from "./life-sim";
import { type NpcMind, initialMinds } from "./world-sim";

export const MAX_OPEN_QUESTS = 3;
export const MAX_ADDED_NPCS = 4;
export const MAX_ADDED_ITEMS = 12;
export const MAX_ADDED_QUESTS = 24;

type Placed<T> = T & { x: number; y: number };

export interface WorldAdditions {
	npcs: Placed<WorldNpc>[];
	items: Placed<WorldItem>[];
	quests: WorldQuest[];
}

export interface DirectorPatch {
	quest: WorldQuest;
	npcs?: (WorldNpc & { region?: string })[];
	items?: (WorldItem & { region?: string })[];
}

export const noAdditions = (): WorldAdditions => ({
	npcs: [],
	items: [],
	quests: [],
});

export function withAdditions(
	spec: WorldSpec,
	add: WorldAdditions | undefined,
): WorldSpec {
	if (!add || (!add.npcs.length && !add.items.length && !add.quests.length))
		return spec;
	return {
		...spec,
		npcs: [...spec.npcs, ...add.npcs],
		items: [...spec.items, ...add.items],
		quests: [...spec.quests, ...add.quests],
	};
}

const openQuests = (spec: WorldSpec, s: WorldState) =>
	spec.quests.filter((q) => !s.completedQuests.includes(q.id));

export function directorDue(
	spec: WorldSpec,
	s: WorldState,
	justCompleted: boolean,
): boolean {
	if ((s.additions?.quests.length ?? 0) >= MAX_ADDED_QUESTS) return false;
	const open = openQuests(spec, s).length;
	return open === 0 || (justCompleted && open < MAX_OPEN_QUESTS);
}

export function directorRequest(
	spec: WorldSpec,
	s: WorldState,
	events: string[],
) {
	return {
		task: "director",
		world: spec,
		inventory: s.inventory,
		picked: s.picked,
		completed_quests: s.completedQuests,
		events: events.slice(-6),
		allow_npc: (s.additions?.npcs.length ?? 0) < MAX_ADDED_NPCS,
		allow_item: (s.additions?.items.length ?? 0) < MAX_ADDED_ITEMS,
		allow_rival: roleAllowed(spec, s, "rival"),
		allow_hostile: roleAllowed(spec, s, "hostile"),
	};
}

export function roleAllowed(spec: WorldSpec, s: WorldState, role: NpcRole) {
	const flags = autonomyOf(spec);
	const on = role === "rival" ? flags.rivals : flags.enemies;
	const added = (s.additions?.npcs ?? []).filter((n) => n.role === role);
	return on && added.length < MAX_ADDED_ROLE;
}

export function placeFree(
	map: WorldMap,
	occupied: Point[],
	area: { x: number; y: number; w: number; h: number } | null,
	seed: number,
): Point | null {
	const reach = reachable(map, map.spawn);
	const ok = (x: number, y: number) =>
		!isSolid(map, x, y) &&
		reach[y * map.width + x] === 1 &&
		!(x === map.spawn.x && y === map.spawn.y) &&
		occupied.every((p) => Math.abs(p.x - x) + Math.abs(p.y - y) > 1);
	const rng = new Prng(seed);
	const pick = (cells: Point[]) =>
		cells.length ? cells[Math.floor(rng.random() * cells.length)] : null;
	const inArea: Point[] = [];
	const anywhere: Point[] = [];
	for (let y = 1; y < map.height - 1; y++)
		for (let x = 1; x < map.width - 1; x++) {
			if (!ok(x, y)) continue;
			anywhere.push({ x, y });
			if (
				area &&
				x >= area.x &&
				x < area.x + area.w &&
				y >= area.y &&
				y < area.y + area.h
			)
				inArea.push({ x, y });
		}
	return pick(inArea) ?? pick(anywhere);
}

const taken = (spec: WorldSpec) =>
	new Set(
		[...spec.npcs, ...spec.items, ...spec.quests, ...spec.regions].map(
			(x) => x.id,
		),
	);

export function applyPatch(
	spec: WorldSpec,
	map: WorldMap,
	s: WorldState,
	patch: DirectorPatch,
	occupied: Point[],
): {
	state: WorldState;
	npcs: Placed<WorldNpc>[];
	items: Placed<WorldItem>[];
} | null {
	const add = s.additions ?? noAdditions();
	const ids = taken(spec);
	const q = patch?.quest;
	if (!q || typeof q.id !== "string" || ids.has(q.id) || !q.title) return null;
	if (openQuests(spec, s).length >= MAX_OPEN_QUESTS) return null;
	const newNpcs = (patch.npcs ?? []).slice(0, 1);
	const newItems = (patch.items ?? []).slice(0, 1);
	if (add.npcs.length + newNpcs.length > MAX_ADDED_NPCS) return null;
	if (add.items.length + newItems.length > MAX_ADDED_ITEMS) return null;
	const fresh = new Set<string>([q.id]);
	for (const x of [...newNpcs, ...newItems]) {
		if (
			typeof x?.id !== "string" ||
			!x.name ||
			ids.has(x.id) ||
			fresh.has(x.id)
		)
			return null;
		fresh.add(x.id);
	}
	if (
		!spec.npcs.some((n) => n.id === q.giver) &&
		!newNpcs.some((n) => n.id === q.giver)
	)
		return null;
	if (
		!q.requires_item ||
		(!spec.items.some((i) => i.id === q.requires_item) &&
			!newItems.some((i) => i.id === q.requires_item))
	)
		return null;
	if (
		spec.quests.some((x) => x.requires_item === q.requires_item) ||
		s.inventory.includes(q.requires_item) ||
		s.picked.includes(q.requires_item)
	)
		return null;
	const spots = [...occupied];
	const areaOf = (region?: string) =>
		spec.regions.find((r) => r.id === region) ?? null;
	const place = <T extends { id: string; region?: string }>(x: T) => {
		const at = placeFree(
			map,
			spots,
			areaOf(x.region),
			seedFrom("add", spec.seed, x.id),
		);
		if (!at) return null;
		spots.push(at);
		const { region: _, ...rest } = x;
		return { ...rest, ...at };
	};
	const npcs: Placed<WorldNpc>[] = [];
	for (const n of newNpcs) {
		const p = place(n);
		if (!p) return null;
		npcs.push({
			id: p.id,
			name: String(p.name).slice(0, 60),
			persona: String(p.persona ?? "").slice(0, 600),
			greeting: String(p.greeting ?? "").slice(0, 300) || "Hello!",
			...(ROLES.includes(p.role as NpcRole) &&
			roleAllowed(spec, s, p.role as NpcRole)
				? { role: p.role as NpcRole }
				: {}),
			x: p.x,
			y: p.y,
		});
	}
	const items: Placed<WorldItem>[] = [];
	for (const i of newItems) {
		const p = place(i);
		if (!p) return null;
		items.push({
			id: p.id,
			name: String(p.name).slice(0, 60),
			description: String(p.description ?? "").slice(0, 300),
			x: p.x,
			y: p.y,
		});
	}
	const quest: WorldQuest = {
		id: q.id,
		title: String(q.title).slice(0, 120),
		description: String(q.description ?? "").slice(0, 500),
		giver: q.giver,
		requires_item: q.requires_item,
		reward_text: String(q.reward_text ?? "").slice(0, 200),
	};
	return {
		state: {
			...s,
			additions: {
				npcs: [...add.npcs, ...npcs],
				items: [...add.items, ...items],
				quests: [...add.quests, quest],
			},
			npc: { ...newMinds(spec, npcs, s.clock), ...s.npc },
			activeQuests: [...s.activeQuests, quest.id],
		},
		npcs,
		items,
	};
}

const placed = (v: unknown): v is { x: number; y: number } =>
	!!v &&
	typeof (v as Point).x === "number" &&
	typeof (v as Point).y === "number";

export function additionsOf(raw: unknown, spec: WorldSpec): WorldAdditions {
	const r = (raw ?? {}) as Partial<WorldAdditions>;
	const ids = taken(spec);
	const fresh = (x: { id?: unknown; name?: unknown } | null | undefined) => {
		if (!x || typeof x.id !== "string" || !x.name || ids.has(x.id))
			return false;
		ids.add(x.id);
		return true;
	};
	const inside = (p: Point) =>
		p.x > 0 && p.y > 0 && p.x < spec.width - 1 && p.y < spec.height - 1;
	const npcs = (Array.isArray(r.npcs) ? r.npcs : [])
		.filter((n) => placed(n) && inside(n) && fresh(n))
		.slice(0, MAX_ADDED_NPCS);
	const items = (Array.isArray(r.items) ? r.items : [])
		.filter((i) => placed(i) && inside(i) && fresh(i))
		.slice(0, MAX_ADDED_ITEMS);
	const npcIds = new Set([...spec.npcs, ...npcs].map((n) => n.id));
	const itemIds = new Set([...spec.items, ...items].map((i) => i.id));
	const quests = (Array.isArray(r.quests) ? r.quests : [])
		.filter(
			(q) =>
				typeof q?.title === "string" &&
				npcIds.has(q.giver) &&
				(!q.requires_item || itemIds.has(q.requires_item)) &&
				fresh({ id: q.id, name: q.title }),
		)
		.slice(0, MAX_ADDED_QUESTS);
	return { npcs, items, quests };
}

/** Minds for villagers the director adds, with the same life fields as the starting cast. */
function newMinds(
	spec: WorldSpec,
	npcs: WorldSpec["npcs"],
	clock: number,
): Record<string, NpcMind> {
	const life = freshLife(lifeOf(spec));
	const out = initialMinds({ ...spec, npcs }, clock);
	for (const id of Object.keys(out)) out[id] = { ...out[id], ...life };
	return out;
}
