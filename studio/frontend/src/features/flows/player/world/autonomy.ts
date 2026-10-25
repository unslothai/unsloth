// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Villagers acting on their own: making and trading goods, finding a home, racing the player for quest items and,
// for hostile ones, chasing and robbing them. One beat a second, decided here with no model call and seeded by
// world, villager and beat, so a run replays the same; the renderers only carry out the commands it returns.

import { seedFrom } from "../../expr";
import type { WorldSpec } from "../../types";
import { followArea, seeded } from "./needs";
import type { Area } from "./phaser-game";
import type { Point, WorldMap } from "./procgen";
import { autonomyOf, hostilesOf, rivalFor } from "./roles";
import { placeFree } from "./world-director";
import type { WorldState } from "./world-state";

export const GOODS = {
  bread: { name: "Bread", icon: "🍞", price: 3, food: true },
  fish: { name: "Fish", icon: "🐟", price: 4, food: true },
  apples: { name: "Apples", icon: "🍎", price: 2, food: true },
  wood: { name: "Wood", icon: "🪵", price: 3, food: false },
  stone: { name: "Stone", icon: "🪨", price: 4, food: false },
  tools: { name: "Tools", icon: "🔨", price: 8, food: false },
  flowers: { name: "Flowers", icon: "💐", price: 2, food: false },
} as const;
export type Good = keyof typeof GOODS;
export const GOOD_IDS = Object.keys(GOODS) as Good[];
const isGood = (g: unknown): g is Good =>
  typeof g === "string" && Object.hasOwn(GOODS, g);

/** What a villager makes, read from their name and persona; anyone else gets a good picked by seed. */
const MAKES: [RegExp, Good][] = [
  [/bak|bread|cafe|cook|chef/, "bread"],
  [/fish|pier|sail|boat|harbo/, "fish"],
  [/farm|orchard|garden|apple|harvest/, "apples"],
  [/wood|carpent|lumber|forest|logg/, "wood"],
  [/stone|mason|quarr|mine|miner/, "stone"],
  [/smith|forge|tool|tinker|craft/, "tools"],
  [/flower|florist|herb|bloom/, "flowers"],
];
export const HOME_RE = /\b(home|house|cottage|hut|cabin|lodge)\b/i;
export const SAFE_RE =
  /\b(home|house|cottage|inn|tavern|square|plaza|temple|church|chapel|shrine)\b/i;

export const PLAYER = "player";
const NPC_COINS = 20;
const PLAYER_COINS = 15;
const STOCK_START = 3;
const STOCK_CAP = 5;
/** Beats between a villager making one more of their good, and between shopping trips. */
const MAKE_EVERY = 20;
const SHOP_EVERY = 25;
const SHOP_GIVE_UP = 30;
/** Beats of head start the player gets on a rival. */
const RACE_DELAY = 25;
const CHASE = 6;
const LEASH = 12;
const COOLDOWN = 40;
const MAX_CHASERS = 2;
export const ITEM_PRICE = 8;
const HOME_COST: Partial<Record<Good, number>> = { wood: 2, stone: 1 };

export type Stock = Partial<Record<Good, number>>;

export interface AutoState {
  beat: number;
  /** "player" and villager ids. */
  coins: Record<string, number>;
  goods: Record<string, Stock>;
  /** A world item a villager got to first -> that villager. */
  held: Record<string, string>;
  homes: Record<string, { place?: string; at?: Point }>;
  /** A villager on their way to buy something. */
  shop: Record<string, { seller: string; good: Good; until: number }>;
  /** Quest id -> the beat its rival sets off. */
  race: Record<string, number>;
  /** Villager -> beat their next encounter may start. */
  cool: Record<string, number>;
  /** What each hostile villager took from the player, given back when the player stands up to them. */
  stolen: Record<string, { coins: number; goods: Stock }>;
  encounter: string | null;
}

export type Cmd =
  | { do: "place"; id: string; area: Area | null }
  | { do: "status"; id: string; icon: string }
  | { do: "say"; id: string; text: string }
  | { do: "remove"; id: string }
  | { do: "marker"; id: string; at: Point; icon: string; label: string };

export function makesOf(spec: WorldSpec, npcId: string): Good {
  const n = spec.npcs.find((x) => x.id === npcId);
  const text = `${n?.name ?? ""} ${n?.persona ?? ""}`.toLowerCase();
  for (const [re, g] of MAKES) if (re.test(text)) return g;
  return GOOD_IDS[seedFrom(spec.seed, "makes", npcId) % GOOD_IDS.length];
}

/** Every villager (including ones added while playing) has a purse and a stock; idempotent. */
function withVillagers(spec: WorldSpec, a: AutoState): AutoState {
  const missing = spec.npcs.filter((n) => !(n.id in a.coins));
  if (!missing.length && PLAYER in a.coins) return a;
  const coins: Record<string, number> = { [PLAYER]: PLAYER_COINS, ...a.coins };
  const goods: Record<string, Stock> = { [PLAYER]: {}, ...a.goods };
  for (const n of missing) {
    coins[n.id] = NPC_COINS;
    goods[n.id] = { [makesOf(spec, n.id)]: STOCK_START };
  }
  return { ...a, coins, goods };
}

export function initialAuto(spec: WorldSpec): AutoState {
  return withVillagers(spec, {
    beat: 0,
    coins: {},
    goods: {},
    held: {},
    homes: {},
    shop: {},
    race: {},
    cool: {},
    stolen: {},
    encounter: null,
  });
}

const count = (v: unknown) =>
  typeof v === "number" && Number.isFinite(v) && v > 0 ? Math.floor(v) : 0;

function stockOf(raw: unknown): Stock {
  const out: Stock = {};
  for (const [g, v] of Object.entries((raw ?? {}) as Record<string, unknown>))
    if (isGood(g) && count(v)) out[g] = count(v);
  return out;
}

const pointOf = (v: unknown): Point | undefined => {
  const p = v as Point | null;
  return p && typeof p.x === "number" && typeof p.y === "number"
    ? { x: p.x, y: p.y }
    : undefined;
};

/** A save's autonomy state; anything that does not fit the world is dropped. */
export function autoOf(raw: unknown, spec: WorldSpec): AutoState {
  const r = (raw ?? {}) as Partial<AutoState>;
  const npcs = new Set(spec.npcs.map((n) => n.id));
  const items = new Set(spec.items.map((i) => i.id));
  const holders = new Set([...npcs, PLAYER]);
  const a = initialAuto(spec);
  a.beat = count(r.beat);
  for (const [id, c] of Object.entries(r.coins ?? {}))
    if (holders.has(id)) a.coins[id] = count(c);
  for (const [id, g] of Object.entries(r.goods ?? {}))
    if (holders.has(id)) a.goods[id] = stockOf(g);
  for (const [it, by] of Object.entries(r.held ?? {}))
    if (items.has(it) && npcs.has(by)) a.held[it] = by;
  for (const [id, h] of Object.entries(r.homes ?? {}))
    if (npcs.has(id) && h)
      a.homes[id] = {
        ...(typeof h.place === "string" ? { place: h.place } : {}),
        ...(pointOf(h.at) ? { at: pointOf(h.at) } : {}),
      };
  for (const [id, b] of Object.entries(r.race ?? {})) a.race[id] = count(b);
  for (const [id, b] of Object.entries(r.cool ?? {}))
    if (npcs.has(id)) a.cool[id] = count(b);
  for (const [id, st] of Object.entries(r.stolen ?? {}))
    if (npcs.has(id) && st)
      a.stolen[id] = { coins: count(st.coins), goods: stockOf(st.goods) };
  for (const [id, t] of Object.entries(r.shop ?? {}))
    if (npcs.has(id) && t && npcs.has(t.seller) && isGood(t.good))
      a.shop[id] = { seller: t.seller, good: t.good, until: count(t.until) };
  // An encounter open when saving is dropped: the player loads into a calm world.
  return a;
}

/** Bounded scarcity: a seller low on stock asks more, one with plenty asks less (half to double the base). */
export function priceOf(good: Good, sellerStock: number): number {
  const base = GOODS[good].price;
  const p = Math.round(base * (1 + 0.15 * (2 - sellerStock)));
  return Math.max(Math.max(1, Math.ceil(base / 2)), Math.min(base * 2, p));
}

const have = (a: AutoState, id: string, g: Good) => a.goods[id]?.[g] ?? 0;

function addGood(a: AutoState, id: string, g: Good, n: number): AutoState {
  const left = have(a, id, g) + n;
  const stock = { ...(a.goods[id] ?? {}) };
  if (left > 0) stock[g] = left;
  else delete stock[g];
  return { ...a, goods: { ...a.goods, [id]: stock } };
}

function pay(a: AutoState, from: string, to: string, n: number): AutoState {
  return {
    ...a,
    coins: {
      ...a.coins,
      [from]: (a.coins[from] ?? 0) - n,
      [to]: (a.coins[to] ?? 0) + n,
    },
  };
}

/** One unit of a good changes hands for coins; null when the seller has none or the buyer cannot pay. */
export function trade(
  a: AutoState,
  seller: string,
  buyer: string,
  good: Good,
): { auto: AutoState; price: number } | null {
  if (seller === buyer || have(a, seller, good) < 1) return null;
  const price = priceOf(good, have(a, seller, good));
  if ((a.coins[buyer] ?? 0) < price) return null;
  return {
    auto: addGood(
      addGood(pay(a, buyer, seller, price), seller, good, -1),
      buyer,
      good,
      1,
    ),
    price,
  };
}

const npcName = (spec: WorldSpec, id: string) =>
  spec.npcs.find((n) => n.id === id)?.name ?? id;
const itemName = (spec: WorldSpec, id: string) =>
  spec.items.find((i) => i.id === id)?.name ?? id;

/** The player buys one unit from a villager (a trade panel button). */
export function playerBuy(
  spec: WorldSpec,
  s: WorldState,
  npcId: string,
  good: Good,
): { state: WorldState; note: string } | null {
  if (!s.auto) return null;
  const t = trade(s.auto, npcId, PLAYER, good);
  return t
    ? {
        state: { ...s, auto: t.auto },
        note: `Bought ${GOODS[good].name} from ${npcName(spec, npcId)} · ${t.price} coins`,
      }
    : null;
}

export function playerSell(
  spec: WorldSpec,
  s: WorldState,
  npcId: string,
  good: Good,
): { state: WorldState; note: string } | null {
  if (!s.auto) return null;
  const t = trade(s.auto, PLAYER, npcId, good);
  return t
    ? {
        state: { ...s, auto: t.auto },
        note: `Sold ${GOODS[good].name} to ${npcName(spec, npcId)} · ${t.price} coins`,
      }
    : null;
}

/** What buying back an item from the villager holding it costs: never more than the player has. */
export const buyBackPrice = (s: WorldState) =>
  Math.min(ITEM_PRICE, s.auto?.coins[PLAYER] ?? 0);

/** A lost race always has a way through: the holder sells the item back for whatever the player can pay. */
export function buyBack(
  spec: WorldSpec,
  s: WorldState,
  npcId: string,
  itemId: string,
): { state: WorldState; note: string } | null {
  if (!s.auto || s.auto.held[itemId] !== npcId) return null;
  const price = buyBackPrice(s);
  const { [itemId]: _, ...held } = s.auto.held;
  return {
    state: {
      ...s,
      inventory: [...s.inventory, itemId],
      auto: { ...pay(s.auto, PLAYER, npcId, price), held },
    },
    note: `Bought back ${itemName(spec, itemId)} from ${npcName(spec, npcId)} · ${price} coins`,
  };
}

export interface Ground {
  spec: WorldSpec;
  map: WorldMap;
  /** Places and regions by id. */
  areas: Map<string, Area>;
  /** Where each world item lies. */
  items: Record<string, Point>;
  /** Where each villager started, which a hostile one never strays far from. */
  homes: Record<string, Point>;
}

export interface Seen {
  player: Point;
  npcs: Record<string, Point>;
}

const dist = (a: Point, b: Point) =>
  Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
const inside = (a: Area, p: Point) =>
  p.x >= a.x && p.x < a.x + a.w && p.y >= a.y && p.y < a.y + a.h;
const around = (p: Point, r: number): Area => ({
  x: Math.max(0, p.x - r),
  y: Math.max(0, p.y - r),
  w: r * 2 + 1,
  h: r * 2 + 1,
});

function namedAreas(g: Ground): { id: string; name: string; area: Area }[] {
  const out: { id: string; name: string; area: Area }[] = [];
  for (const p of [
    ...(g.spec.simulation?.places ?? []),
    ...(g.spec.regions ?? []),
  ]) {
    const area = g.areas.get(p.id);
    if (area && !out.some((o) => o.id === p.id))
      out.push({ id: p.id, name: p.name, area });
  }
  return out;
}

/** Homes, the inn, the square and the like, plus every home a villager built: no hostile follows the player in. */
export function safeAt(g: Ground, a: AutoState, p: Point): boolean {
  if (namedAreas(g).some((n) => SAFE_RE.test(n.name) && inside(n.area, p)))
    return true;
  return Object.values(a.homes).some((h) => h.at && dist(h.at, p) <= 1);
}

/**
 * One beat of village life. Pure: the same world, state and positions give the same result, and the renderer is
 * only told what to do through the returned commands. `steered` lists the villagers this beat walks somewhere.
 */
export function autoStep(
  g: Ground,
  s: WorldState,
  seen: Seen,
): { state: WorldState; cmds: Cmd[]; notes: string[]; steered: string[] } {
  const { spec } = g;
  const flags = autonomyOf(spec);
  let a = withVillagers(spec, s.auto ?? initialAuto(spec));
  const beat = a.beat + 1;
  a = { ...a, beat };
  let st: WorldState = s;
  const cmds: Cmd[] = [];
  const notes: string[] = [];
  const steered: string[] = [];
  const hostile = new Set(hostilesOf(spec));
  const villagers = spec.npcs.filter((n) => n.id in seen.npcs);
  const at = (id: string) => seen.npcs[id];

  // Making: each villager adds one of their good now and then, up to a small stock.
  if (flags.trade || flags.homes)
    for (const n of villagers) {
      if ((beat + (seedFrom(n.id) % MAKE_EVERY)) % MAKE_EVERY) continue;
      const good = makesOf(spec, n.id);
      if (have(a, n.id, good) < STOCK_CAP) a = addGood(a, n.id, good, 1);
    }

  // Homes: claim a free house; without one, build on a free plot once the materials are in hand.
  if (flags.homes) {
    const houses = namedAreas(g).filter((n) => HOME_RE.test(n.name));
    const claimed = new Set(
      Object.values(a.homes)
        .map((h) => h.place)
        .filter(Boolean),
    );
    for (const n of villagers) {
      if (a.homes[n.id] || hostile.has(n.id)) continue;
      const free = houses.find((h) => !claimed.has(h.id));
      if (free) {
        a = { ...a, homes: { ...a.homes, [n.id]: { place: free.id } } };
        notes.push(`${n.name} moved into ${free.name}`);
        cmds.push({ do: "say", id: n.id, text: "Home sweet home!" });
        break;
      }
      if (
        !Object.entries(HOME_COST).every(
          ([good, need]) => have(a, n.id, good as Good) >= (need ?? 0),
        )
      )
        continue;
      const taken = [
        ...Object.values(seen.npcs),
        ...Object.values(g.items),
        seen.player,
        ...Object.values(a.homes).flatMap((h) => (h.at ? [h.at] : [])),
      ];
      const plot = placeFree(
        g.map,
        taken,
        around(at(n.id), 4),
        seedFrom(spec.seed, "home", n.id),
      );
      if (!plot) continue;
      for (const [good, need] of Object.entries(HOME_COST))
        a = addGood(a, n.id, good as Good, -(need ?? 0));
      a = { ...a, homes: { ...a.homes, [n.id]: { at: plot } } };
      cmds.push(homeMarker(n.id, n.name, plot));
      notes.push(`${n.name} built a home`);
      break;
    }
  }

  // Shopping: a villager who wants something walks to the nearest seller and buys it there.
  if (flags.trade || flags.homes) {
    const shop = { ...a.shop };
    for (const n of villagers) {
      if (hostile.has(n.id)) continue;
      const trip = shop[n.id];
      if (trip) {
        const there = at(trip.seller);
        if (beat > trip.until || !there) {
          delete shop[n.id];
          continue;
        }
        if (dist(at(n.id), there) <= 2) {
          const t = trade(a, trip.seller, n.id, trip.good);
          delete shop[n.id];
          if (!t) continue;
          a = t.auto;
          const food = GOODS[trip.good].food;
          if (food) {
            a = addGood(a, n.id, trip.good, -1);
            st = feed(st, n.id);
          }
          notes.push(
            `${n.name} bought ${GOODS[trip.good].name.toLowerCase()} from ${npcName(spec, trip.seller)} · ${t.price} coins`,
          );
          cmds.push({
            do: "say",
            id: n.id,
            text: food
              ? "Just what I needed!"
              : `Thanks for the ${GOODS[trip.good].name.toLowerCase()}!`,
          });
          continue;
        }
        steered.push(n.id);
        cmds.push({ do: "place", id: n.id, area: followArea(there) });
        cmds.push({ do: "status", id: n.id, icon: "🛒" });
        continue;
      }
      if ((beat + (seedFrom("shop", n.id) % SHOP_EVERY)) % SHOP_EVERY) continue;
      const want = wantOf(g, a, st, n.id, flags.homes);
      if (!want) continue;
      const sellers = villagers
        .filter(
          (m) =>
            m.id !== n.id &&
            !hostile.has(m.id) &&
            have(a, m.id, want) > 0 &&
            (a.coins[n.id] ?? 0) >= priceOf(want, have(a, m.id, want)),
        )
        .sort(
          (x, y) =>
            dist(at(n.id), at(x.id)) - dist(at(n.id), at(y.id)) ||
            x.id.localeCompare(y.id),
        );
      if (sellers[0])
        shop[n.id] = {
          seller: sellers[0].id,
          good: want,
          until: beat + SHOP_GIVE_UP,
        };
    }
    a = { ...a, shop };
  }

  // Rivals: after a head start, a rival walks to an open quest's item; whoever reaches it first keeps it.
  if (flags.quests)
    for (const q of spec.quests) {
      const it = q.requires_item;
      if (
        !it ||
        !st.activeQuests.includes(q.id) ||
        st.completedQuests.includes(q.id) ||
        st.picked.includes(it) ||
        a.held[it] ||
        !g.items[it]
      )
        continue;
      const rival = rivalFor(spec, q);
      if (!rival || !at(rival) || steered.includes(rival)) continue;
      if (!(q.id in a.race)) {
        a = { ...a, race: { ...a.race, [q.id]: beat + RACE_DELAY } };
        notes.push(
          `${npcName(spec, rival)} is after the ${itemName(spec, it)} too!`,
        );
        cmds.push({
          do: "say",
          id: rival,
          text: `That ${itemName(spec, it)} will be mine!`,
        });
        continue;
      }
      if (beat < a.race[q.id]) continue;
      if (dist(at(rival), g.items[it]) <= 1) {
        a = { ...a, held: { ...a.held, [it]: rival } };
        st = { ...st, picked: [...st.picked, it] };
        cmds.push({ do: "remove", id: it });
        cmds.push({ do: "say", id: rival, text: "Too slow!" });
        notes.push(
          `${npcName(spec, rival)} got the ${itemName(spec, it)} first. Talk to them to buy it back.`,
        );
        continue;
      }
      steered.push(rival);
      cmds.push({ do: "place", id: rival, area: around(g.items[it], 0) });
      cmds.push({ do: "status", id: rival, icon: "🏁" });
    }

  // Hostiles: chase a player who comes close, never far from home and never into a safe place.
  if (flags.enemies && !a.encounter) {
    const safe = safeAt(g, a, seen.player);
    let chasers = 0;
    for (const id of [...hostile].sort()) {
      const me = at(id);
      const home = g.homes[id] ?? me;
      if (!me) continue;
      steered.push(id);
      const chase =
        !safe &&
        chasers < MAX_CHASERS &&
        (a.cool[id] ?? 0) <= beat &&
        dist(me, seen.player) <= CHASE &&
        dist(home, seen.player) <= LEASH;
      if (!chase) {
        cmds.push({ do: "place", id, area: around(home, 1) });
        cmds.push({ do: "status", id, icon: "😠" });
        continue;
      }
      chasers++;
      if (dist(me, seen.player) <= 2) {
        a = { ...a, encounter: id, cool: { ...a.cool, [id]: beat + COOLDOWN } };
        cmds.push({ do: "say", id, text: "Hand it over!" });
        notes.push(`${npcName(spec, id)} blocks your way!`);
        break;
      }
      cmds.push({ do: "place", id, area: followArea(seen.player) });
      cmds.push({ do: "status", id, icon: "💢" });
    }
  }
  // A villager following the player or on an errand they asked for keeps walking where the player sent them.
  const directed = new Set([
    ...(s.following ?? []),
    ...Object.keys(s.errands ?? {}),
  ]);
  return {
    state: { ...st, auto: a },
    cmds: cmds.filter(
      (c) => !(c.do === "place" || c.do === "status") || !directed.has(c.id),
    ),
    notes,
    steered: steered.filter((id) => !directed.has(id)),
  };
}

export const homeMarker = (id: string, name: string, at: Point): Cmd => ({
  do: "marker",
  id: `home:${id}`,
  at,
  icon: "🏠",
  label: `${name}'s home`,
});

function feed(s: WorldState, npcId: string): WorldState {
  const mind = s.npc[npcId];
  if (!mind?.needs) return s;
  return {
    ...s,
    npc: {
      ...s.npc,
      [npcId]: {
        ...mind,
        needs: { ...mind.needs, hunger: Math.min(100, mind.needs.hunger + 35) },
      },
    },
  };
}

/** What a villager goes out to buy: food when hungry, materials for a home, else now and then something nice. */
function wantOf(
  g: Ground,
  a: AutoState,
  s: WorldState,
  npcId: string,
  homes: boolean,
): Good | null {
  const rnd = seeded(g.spec.seed, "want", npcId, a.beat);
  const own = makesOf(g.spec, npcId);
  const hunger = s.npc[npcId]?.needs?.hunger;
  if (typeof hunger === "number" && hunger < 40) {
    const food = GOOD_IDS.filter((x) => GOODS[x].food && x !== own);
    return food[Math.floor(rnd() * food.length)] ?? null;
  }
  const houses = namedAreas(g).filter((n) => HOME_RE.test(n.name)).length;
  if (homes && !a.homes[npcId] && Object.keys(a.homes).length >= houses)
    for (const [good, need] of Object.entries(HOME_COST))
      if (have(a, npcId, good as Good) < (need ?? 0) && good !== own)
        return good as Good;
  if (rnd() < 0.3) {
    const other = GOOD_IDS.filter((x) => x !== own);
    return other[Math.floor(rnd() * other.length)];
  }
  return null;
}

export type Stand = "evade" | "defend" | "negotiate";

/**
 * The player's answer to a hostile villager, resolved here: slipping away or standing firm can fail and cost
 * coins or a good (never a quest item); standing firm gets back what they took before; paying always ends it.
 */
export function resolveEncounter(
  spec: WorldSpec,
  s: WorldState,
  stand: Stand,
): { state: WorldState; note: string } {
  const a = s.auto;
  const id = a?.encounter;
  if (!a || !id) return { state: s, note: "" };
  const name = npcName(spec, id);
  const rnd = seeded(spec.seed, "encounter", id, a.beat, stand);
  let next: AutoState = { ...a, encounter: null };
  if (stand === "negotiate") {
    const n = Math.min(3, a.coins[PLAYER] ?? 0);
    next = pay(next, PLAYER, id, n);
    return {
      state: { ...s, auto: next },
      note: n
        ? `You paid ${name} ${n} coins to leave you alone`
        : `${name} sees you have nothing and lets you go`,
    };
  }
  const odds = stand === "evade" ? 0.6 : have(a, PLAYER, "tools") ? 0.7 : 0.5;
  if (rnd() < odds) {
    if (stand === "evade")
      return {
        state: { ...s, auto: next },
        note: `You slipped away from ${name}`,
      };
    const took = next.stolen[id];
    if (took) {
      next = pay(next, id, PLAYER, Math.min(took.coins, next.coins[id] ?? 0));
      for (const [good, n] of Object.entries(took.goods)) {
        const back = Math.min(n ?? 0, have(next, id, good as Good));
        next = addGood(
          addGood(next, id, good as Good, -back),
          PLAYER,
          good as Good,
          back,
        );
      }
      const { [id]: _, ...stolen } = next.stolen;
      next = { ...next, stolen };
    }
    return {
      state: { ...s, auto: next },
      note: took
        ? `You stood up to ${name} and got your things back`
        : `You stood up to ${name}, who backs off`,
    };
  }
  // Lost: coins first, else one good. Quest items are never taken.
  const took = next.stolen[id] ?? { coins: 0, goods: {} };
  const coins = Math.min(5, next.coins[PLAYER] ?? 0);
  if (coins) {
    next = pay(next, PLAYER, id, coins);
    next = {
      ...next,
      stolen: { ...next.stolen, [id]: { ...took, coins: took.coins + coins } },
    };
    return { state: { ...s, auto: next }, note: `${name} took ${coins} coins` };
  }
  const good = GOOD_IDS.find((x) => have(next, PLAYER, x) > 0);
  if (good) {
    next = addGood(addGood(next, PLAYER, good, -1), id, good, 1);
    next = {
      ...next,
      stolen: {
        ...next.stolen,
        [id]: {
          ...took,
          goods: { ...took.goods, [good]: (took.goods[good] ?? 0) + 1 },
        },
      },
    };
    return {
      state: { ...s, auto: next },
      note: `${name} took your ${GOODS[good].name.toLowerCase()}`,
    };
  }
  return {
    state: { ...s, auto: next },
    note: `${name} finds nothing worth taking`,
  };
}

/** Everything the player and villagers own, for conservation checks. */
export function totals(a: AutoState): { coins: number; goods: Stock } {
  let coins = 0;
  const goods: Stock = {};
  for (const c of Object.values(a.coins)) coins += c;
  for (const st of Object.values(a.goods))
    for (const [good, n] of Object.entries(st))
      goods[good as Good] = (goods[good as Good] ?? 0) + (n ?? 0);
  return { coins, goods };
}
