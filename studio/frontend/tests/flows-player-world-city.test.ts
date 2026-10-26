// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const C = await import("../src/features/flows/player/world/city-sim.ts");
const A = await import("../src/features/flows/player/world/city-advisor.ts");
const W = await import("../src/features/flows/player/world/world-state.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;
type Sim = InstanceType<typeof C.CitySim>;

function spec(over: Partial<Spec> = {}): Spec {
  return {
    title: "Testville",
    seed: 11,
    width: 32,
    height: 32,
    tile_size: 32,
    biome: "meadow",
    tileset: {},
    player: { name: "Pip" },
    npcs: [
      { id: "ada", name: "Ada", persona: "baker", greeting: "hi" },
      { id: "bo", name: "Bo", persona: "mason", greeting: "hey" },
    ],
    items: [],
    quests: [],
    regions: [],
    vars: {},
    intro_text: "",
    city: { funds: 100000 },
    ...over,
  };
}

function sim(s: Spec = spec()): Sim {
  return new C.CitySim(C.newCity(s), null);
}

function placed(c: Sim, tool: string, x: number, y: number): void {
  const r = c.place(tool as never, x, y);
  assert.ok(r.ok, `${tool} at ${x},${y}: ${r.reason}`);
}

// A plant wired to a block of R, C and I on one road; runs until a villager has moved in.
function town(s: Spec): Sim {
  const c = sim(s);
  placed(c, "P", 3, 3);
  for (let x = 5; x <= 14; x++) placed(c, "wire", x, 3);
  placed(c, "R", 16, 3);
  placed(c, "C", 19, 3);
  placed(c, "I", 22, 3);
  for (let x = 14; x <= 24; x++) placed(c, "road", x, 5);
  for (let k = 0; k < C.CYCLES_PER_MONTH * 6 && !c.residents.ada; k++) {
    c.step();
    c.moveIn(["ada"]);
  }
  assert.ok(c.residents.ada, "someone moved in");
  while (c.month < 1) c.step();
  return c;
}

// Census fixtures fed straight into SetValves; expected values computed by hand from s_sim.c.
function valvesFor(census: {
  res: number;
  com: number;
  ind: number;
  his: [number, number, number];
  tax: number;
  valves?: { r: number; c: number; i: number };
}) {
  const st = C.newCity(spec());
  st.census = {
    ...st.census,
    res: census.res,
    com: census.com,
    ind: census.ind,
  };
  st.his = {
    res: [0, census.his[0]],
    com: [0, census.his[1]],
    ind: [0, census.his[2]],
  };
  st.tax = census.tax;
  st.valves = census.valves ?? { r: 0, c: 0, i: 0 };
  const c = new C.CitySim(st, null);
  c.setValves();
  return { ...c.valves, total: c.census.total };
}

test("SetValves matches Micropolis on census fixtures", () => {
  // NormResPop 50, jobs 74: employment 1.48, migration 24, births 1, so PjResPop 75 and R ratio 1.5 -> +300.
  // LaborBase 37 / 74 = 0.5; IntMarket 58 / 3.7 x 0.5 / 4 = 1.9595 -> C +575. PjIndPop 2.4 floors at 5: 5 / 4 -> +150.
  const fixture = {
    res: 400,
    com: 4,
    ind: 4,
    his: [37, 40, 34] as [number, number, number],
  };
  assert.deepEqual(valvesFor({ ...fixture, tax: 7 }), {
    r: 300,
    c: 575,
    i: 150,
    total: 58,
  });
  // Tax 0 adds TaxTable[0] = 200 to each ratio; tax 20 adds -600 (valves truncate toward zero, like a C short).
  assert.deepEqual(valvesFor({ ...fixture, tax: 0 }), {
    r: 500,
    c: 775,
    i: 350,
    total: 58,
  });
  assert.deepEqual(valvesFor({ ...fixture, tax: 20 }), {
    r: -300,
    c: -24,
    i: -450,
    total: 58,
  });
  // An empty city: R ratio 1.3 (+180), no commercial market (-600), industry floor 5 capped at 2 (+600).
  assert.deepEqual(
    valvesFor({ res: 0, com: 0, ind: 0, his: [0, 0, 0], tax: 7 }),
    {
      r: 180,
      c: -600,
      i: 600,
      total: 0,
    },
  );
  // Valves saturate at +-2000 (R) and +-1500 (C, I).
  assert.deepEqual(
    valvesFor({
      res: 0,
      com: 0,
      ind: 0,
      his: [0, 0, 0],
      tax: 7,
      valves: { r: 1990, c: -1400, i: 1499 },
    }),
    { r: 2000, c: -1500, i: 1500, total: 0 },
  );
});

test("power flows from a plant through wires and zones, not roads", () => {
  const c = sim();
  placed(c, "P", 3, 3);
  placed(c, "R", 12, 3);
  for (let x = 5; x <= 10; x++) placed(c, "wire", x, 3);
  const r = c.zoneOf(12, 3);
  assert.ok(r && c.isPowered(r));
  placed(c, "bulldoze", 8, 3);
  assert.equal(c.isPowered(r), false);
  placed(c, "road", 8, 3);
  assert.equal(c.isPowered(r), false, "a plain road does not conduct");
  placed(c, "wire", 8, 3);
  assert.equal(c.kind[3 * 32 + 8], C.ROADWIRE);
  assert.ok(c.isPowered(r), "a road with a power line does");
  placed(c, "C", 15, 3);
  assert.ok(
    c.isPowered(c.zoneOf(15, 3)!),
    "zones pass power to their neighbours",
  );
});

test("a plant powers at most 700 tiles", () => {
  const s = spec({ width: 40, height: 40 });
  const c = sim(s);
  placed(c, "P", 20, 20);
  for (let y = 0; y < 40; y++)
    for (let x = 0; x < 40; x++) if (!c.zoneOf(x, y)) c.place("wire", x, y);
  let n = 0;
  for (const p of c.powered) n += p;
  assert.equal(n, C.POWER_PER_PLANT);
  assert.equal(c.powerUsed, 700);
});

test("traffic: a random-walk trip passes, fails at a dead end, or finds no road", () => {
  const c = sim();
  placed(c, "R", 3, 10);
  for (let x = 5; x <= 8; x++) placed(c, "road", x, 10);
  placed(c, "C", 10, 10);
  const r = c.zoneOf(3, 10)!;
  assert.equal(c.makeTraf(r, 0), 1, "R reaches C");
  assert.ok(
    c.traffic.some((v) => v > 0),
    "a passed trip leaves traffic",
  );

  const d = sim();
  placed(d, "R", 3, 10);
  for (let x = 5; x <= 8; x++) placed(d, "road", x, 10);
  placed(d, "R", 10, 10);
  assert.equal(d.makeTraf(d.zoneOf(3, 10)!, 0), 0, "R to R is no trip");
  assert.ok(
    d.traffic.every((v) => v === 0),
    "a failed trip leaves none",
  );

  const far = sim(spec({ width: 64, height: 16 }));
  placed(far, "R", 3, 8);
  for (let x = 5; x <= 40; x++) placed(far, "road", x, 8);
  placed(far, "C", 42, 8);
  assert.equal(far.makeTraf(far.zoneOf(3, 8)!, 0), 0, "beyond MAXDIS 30");

  const lone = sim();
  placed(lone, "R", 3, 10);
  assert.equal(lone.makeTraf(lone.zoneOf(3, 10)!, 0), -1);
  // Industry only drives to developed homes.
  const e = sim();
  placed(e, "I", 3, 10);
  for (let x = 5; x <= 8; x++) placed(e, "road", x, 10);
  placed(e, "R", 10, 10);
  assert.equal(e.makeTraf(e.zoneOf(3, 10)!, 2), 0);
  e.zoneOf(10, 10)!.pop = 3;
  assert.equal(e.makeTraf(e.zoneOf(3, 10)!, 2), 1);
});

test("overlays: land value from distance to the centre, crime from it, pollution from industry", () => {
  const c = sim();
  // No zones: the centre of mass is Micropolis's fallback (HWLDX, HWLDY) >> 1 = (8, 8) in half tiles.
  placed(c, "road", 16, 16);
  placed(c, "road", 0, 0);
  c.step();
  const mid = c.at(16, 16);
  const corner = c.at(0, 0);
  assert.equal(mid.land_value, 34 * 4, "(34 - 0) << 2");
  assert.equal(corner.land_value, (34 - 16) * 4);
  assert.equal(mid.crime, 0, "128 - 136 clamps to 0");
  assert.equal(corner.crime, 128 - 72);
  assert.equal(
    c.at(8, 24).land_value,
    0,
    "undeveloped ground has no land value",
  );
  assert.equal(c.census.lv, Math.trunc((136 + 72) / 2));

  const p = sim();
  placed(p, "I", 10, 10);
  p.step();
  assert.equal(
    p.at(10, 10).pollution,
    0,
    "an empty industrial zone does not pollute",
  );
  p.zoneOf(10, 10)!.pop = 2;
  p.step();
  const near = p.at(10, 10).pollution;
  assert.ok(near > 0);
  assert.ok(near > p.at(24, 24).pollution);
  placed(p, "P", 25, 25);
  p.step();
  assert.ok(p.at(25, 25).pollution > 0, "a plant pollutes");
});

test("budget: CollectTax's yearly funds, paid each month", () => {
  // TaxFund = (50 x 120 / 120) x 7 x 1.4 = 490 a year; RoadFund = 40 x 0.7 = 28; x 16/48 cycles.
  assert.deepEqual(C.budget(50, 120, 7, 40), { tax: 163, roads: 9, flow: 154 });
  assert.deepEqual(C.budget(0, 120, 7, 40), { tax: 0, roads: 0, flow: 0 });
  // trunc(7 x 61 / 120) = 3; 3 x 9 x 1.4 = 37.8 -> 37 a year -> 12 a month; roads trunc(2.1) = 2 -> 0.
  assert.deepEqual(C.budget(7, 61, 9, 3), { tax: 12, roads: 0, flow: 12 });
  const c = sim();
  const before = c.funds;
  placed(c, "road", 5, 5);
  assert.equal(c.funds, before - C.COSTS.road);
  const st = c.snapshot();
  st.census = { ...st.census, total: 50, lv: 120, roads: 40 };
  const d = new C.CitySim(st, null);
  d.collectTax();
  assert.equal(d.funds, before - C.COSTS.road + 154);
  assert.deepEqual(d.cash, { tax: 163, roads: 9, flow: 154 });
  assert.deepEqual(c.place("P", 20, 20).ok, true);
  const broke = new C.CitySim({ ...c.snapshot(), funds: 50 }, null);
  assert.equal(broke.place("R", 10, 25).ok, false);
});

test("powered zones grow and pay off; unpowered ones never do", () => {
  const c = sim();
  placed(c, "P", 3, 3);
  for (let x = 5; x <= 14; x++) placed(c, "wire", x, 3);
  placed(c, "R", 16, 3);
  placed(c, "C", 19, 3);
  placed(c, "I", 22, 3);
  placed(c, "R", 16, 12);
  for (let x = 14; x <= 24; x++) placed(c, "road", x, 5);
  const home = c.zoneOf(16, 3)!;
  let months = 0;
  let grew = 0;
  let moved = 0;
  for (let k = 0; k < C.CYCLES_PER_MONTH * 3; k++) {
    if (c.step()) months++;
    grew = Math.max(grew, home.pop);
    moved += c.moveIn(["ada", "bo"]).length;
    assert.equal(c.zoneOf(16, 12)!.pop, 0, "no power, no houses");
  }
  assert.equal(months, 3);
  assert.equal(c.month, 3);
  assert.ok(grew > 0, "the powered zone grew houses");
  assert.ok(moved > 0);
  assert.equal(c.residents.ada, home.id);
});

test("the simulation is deterministic and a save round-trips exactly", () => {
  const a = town(spec());
  const b = town(spec());
  for (const c of [a, b]) {
    c.zoneOf(19, 3)!.pop = 3;
    c.zoneOf(22, 3)!.pop = 4;
  }
  for (let k = 0; k < 40; k++) {
    a.step();
    b.step();
  }
  for (let k = 0; k < 64 && !a.traffic.some((v) => v > 0); k++) {
    a.step();
    b.step();
  }
  assert.deepEqual(a.snapshot(), b.snapshot());
  assert.ok(
    a.traffic.some((v) => v > 0),
    "the fixture has traffic to carry over",
  );
  assert.ok(a.pollution.some((v) => v > 0));
  const saved = JSON.parse(JSON.stringify(a.snapshot()));
  const parsed = C.parseCity(saved, spec());
  assert.deepEqual(parsed, a.snapshot());
  const r = new C.CitySim(parsed!, null);
  for (const key of [
    "traffic",
    "pollution",
    "land",
    "crime",
    "density",
    "terrain",
    "market",
    "powered",
  ] as const)
    assert.deepEqual([...r[key]], [...a[key]], key);
  for (let k = 0; k < 40; k++) {
    a.step();
    r.step();
  }
  assert.deepEqual(r.snapshot(), a.snapshot());
  assert.equal(
    C.parseCity(saved, spec({ width: 40 })),
    null,
    "another map size",
  );
  assert.equal(C.parseCity({ ...saved, grid: "x" }, spec()), null);
});

test("the city rides in the world save; worlds without it are unchanged", () => {
  const s = spec();
  const c = sim(s);
  placed(c, "R", 6, 6);
  c.step();
  const ws = { ...W.initialWorldState(s), city: c.snapshot() };
  const back = W.deserialize(s, JSON.parse(JSON.stringify(W.serialize(ws))));
  assert.deepEqual(back?.city, c.snapshot());
  const plain: Spec = { ...s };
  delete plain.city;
  const old = W.initialWorldState(plain);
  assert.equal("city" in old, false);
  const reloaded = W.deserialize(
    plain,
    JSON.parse(JSON.stringify({ ...old, city: c.snapshot() })),
  );
  assert.equal(
    reloaded && "city" in reloaded,
    false,
    "no city block, no city state",
  );
  assert.deepEqual(W.deserialize(plain, JSON.parse(JSON.stringify(old))), old);
  const off = spec({ city: { enabled: false } });
  assert.equal(C.cityOf(off), null);
});

test("the advisor digest carries the exact simulation numbers and no tile coordinates", () => {
  const s = spec();
  const c = town(s);
  const d = A.cityDigest(s, c, "month");
  assert.equal(d.funds, c.funds);
  assert.equal(d.demand_r, c.valves.r);
  assert.equal(d.residential_pop, c.census.res);
  assert.equal(d.land_value_avg, c.census.lv);
  assert.equal(d.power_capacity, 700);
  const home = c.zoneById(c.residents.ada)!;
  const at = c.at(home.x, home.y);
  assert.deepEqual(
    d.residents.map((r) => [
      r.id,
      r.zone,
      r.land_value,
      r.pollution,
      r.crime,
      r.power,
    ]),
    [["ada", home.id, at.land_value, at.pollution, at.crime, 1]],
  );
  const req = A.mayorRequest(d);
  assert.equal(req.task, "mayor");
  const keys = JSON.stringify(req).match(/"(x|y)":/g);
  assert.equal(keys, null, "the model is never told where tiles are");
  assert.equal(A.cityDigest(s, c, "advice").petitions, false);
});

test("a mayor reply only lands what the digest supports", () => {
  const s = spec();
  const c = town(s);
  const d = A.cityDigest(s, c, "month");
  const ada = d.residents[0];
  const reply = A.parseMayorReply({
    mayor: {
      advice: "Build more homes.",
      petitions: [
        {
          citizen: "ada",
          zone: "C99",
          metric: "crime",
          value: 1,
          text: "Wrong home.",
        },

        {
          citizen: "ada",
          zone: ada.zone,
          metric: "crime",
          value: ada.crime,
          text: "Too much crime.",
        },
        {
          citizen: "bo",
          zone: ada.zone,
          metric: "crime",
          value: 1,
          text: "Not a resident.",
        },
      ],
      event: { kind: "festival", zone: "", size: 2 },
      name: { zone: d.unnamed, name: "Rose Row" },
    },
  });
  assert.ok(reply);
  const before = c.valves.r;
  const lines = A.applyMayorReply(c, d, reply);
  assert.equal(c.petitions.length, 1);
  assert.equal(c.petitions[0].value, ada.crime);
  assert.equal(c.valves.r, Math.min(2000, before + 200));
  assert.equal(c.advisor?.text, "Build more homes.");
  assert.ok(lines.some((l) => l.startsWith("Ada:")));
  if (d.unnamed) assert.equal(c.zoneById(d.unnamed)?.name, "Rose Row");
  const local = A.localAdvice(A.cityDigest(s, sim(s), "advice"));
  assert.match(local, /power plant/);
  assert.equal(A.parseMayorReply({ text: "hi" }), null);
});

test("events come from the fixed menu and act deterministically", () => {
  const c = sim();
  placed(c, "R", 6, 6);
  placed(c, "I", 12, 6);
  c.zoneOf(6, 6)!.pop = 5;
  c.zoneOf(12, 6)!.pop = 3;
  assert.ok(c.applyEvent("fire", c.zoneOf(6, 6)!.id, 2));
  assert.equal(c.zoneOf(6, 6)!.pop, 3);
  assert.ok(c.applyEvent("strike", c.zoneOf(12, 6)!.id, 1));
  assert.equal(c.zoneOf(12, 6)!.pop, 2);
  assert.equal(c.applyEvent("strike", c.zoneOf(6, 6)!.id, 1), null);
  assert.equal(c.applyEvent("none", "", 1), null);
  assert.deepEqual(
    [...C.EVENTS],
    ["none", "festival", "fire", "strike", "flood", "boom"],
  );
});

test("tax and a month's petitions apply once across a save and reload at the month boundary", () => {
  const s = spec();
  const live = town(s);
  while (live.cycle % C.CYCLES_PER_MONTH !== C.CYCLES_PER_MONTH - 1)
    live.step();
  const before = JSON.parse(JSON.stringify(live.snapshot()));
  live.step();
  const after = JSON.parse(JSON.stringify(live.snapshot()));
  assert.equal(after.taxed, after.month, "the month just ended was taxed");
  // Reload just before the boundary: crossing it taxes exactly as the live run did.
  const early = new C.CitySim(C.parseCity(before, s)!, null);
  early.step();
  assert.deepEqual(early.snapshot(), live.snapshot());
  // Reload just after it: the next month end is the only tax.
  const late = new C.CitySim(C.parseCity(after, s)!, null);
  const funds = late.funds;
  for (let k = 1; k < C.CYCLES_PER_MONTH; k++) late.step();
  assert.equal(late.funds, funds, "no tax again mid-month");
  assert.equal(late.snapshot().taxed, after.month);
  // A stale save with a month already taxed does not collect it twice.
  const busy = {
    ...before,
    census: { ...before.census, res: 400, com: 4, ind: 4, lv: 120 },
  };
  const taxed = new C.CitySim(C.parseCity(busy, s)!, null);
  const taxedFunds = taxed.funds;
  taxed.step();
  assert.ok(
    taxed.cash.tax > 0 && taxed.funds !== taxedFunds,
    "the fixture pays tax",
  );
  const stale = new C.CitySim(
    C.parseCity({ ...busy, taxed: before.month + 1 }, s)!,
    null,
  );
  const staleFunds = stale.funds;
  stale.step();
  assert.equal(
    stale.funds,
    staleFunds,
    "a month already taxed is not taxed again",
  );

  const d = A.cityDigest(s, live, "month");
  const ada = d.residents[0];
  const reply = A.parseMayorReply({
    mayor: {
      advice: "Fine.",
      petitions: [
        {
          citizen: "ada",
          zone: ada.zone,
          metric: "crime",
          value: ada.crime,
          text: "Crime.",
        },
      ],
      event: { kind: "festival", zone: "", size: 1 },
      name: null,
    },
  })!;
  const r0 = live.valves.r;
  A.applyMayorReply(live, d, reply);
  const saved = JSON.parse(JSON.stringify(live.snapshot()));
  A.applyMayorReply(live, d, reply);
  const reloaded = new C.CitySim(C.parseCity(saved, s)!, null);
  A.applyMayorReply(reloaded, d, reply);
  for (const c of [live, reloaded]) {
    assert.equal(c.petitions.length, 1, "one petition, not two");
    assert.equal(c.valves.r, Math.min(2000, r0 + 100), "the festival once");
  }
});

test("old saves (v1, v2) still load; a city world opens them without new state", () => {
  const s = spec();
  const plain: Spec = { ...s };
  delete plain.city;
  for (const v of [1, 2]) {
    const old = { ...W.initialWorldState(plain), v };
    const raw = JSON.parse(JSON.stringify(old));
    const asCity = W.deserialize(s, raw);
    assert.ok(asCity, `v${v} loads`);
    assert.equal(
      "city" in (asCity ?? {}),
      false,
      "no city state until the mayor builds",
    );
    assert.deepEqual(
      raw,
      JSON.parse(JSON.stringify(old)),
      "loading does not rewrite it",
    );
    assert.equal("city" in (W.deserialize(plain, raw) ?? {}), false);
  }
  // A city save from before the period fields counts its months as applied.
  const c = town(s);
  const legacy = JSON.parse(JSON.stringify(c.snapshot()));
  delete legacy.taxed;
  delete legacy.petitioned;
  const parsed = C.parseCity(legacy, s)!;
  assert.equal(parsed.taxed, parsed.month);
  assert.equal(parsed.petitioned, parsed.month);
});
