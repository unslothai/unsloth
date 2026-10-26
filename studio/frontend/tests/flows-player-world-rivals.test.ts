// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const R = await import("../src/features/flows/player/world/rivals.ts");
const P = await import("../src/features/flows/player/world/procgen.ts");
const W = await import("../src/features/flows/player/world/world-state.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;
type Rivals = typeof import("../src/features/flows/player/world/rivals.ts");
type State = ReturnType<Rivals["initialRivals"]>;
type Plan = NonNullable<ReturnType<Rivals["planOf"]>>;

const BASE: Spec = {
  title: "Three Villages",
  seed: 11,
  width: 48,
  height: 36,
  tile_size: 32,
  biome: "meadow",
  tileset: {},
  player: { name: "Pip" },
  npcs: [
    { id: "bea", name: "Bea", persona: "baker", greeting: "" },
    { id: "bo", name: "Bo", persona: "stone mason", greeting: "" },
    { id: "ada", name: "Ada", persona: "carpenter", greeting: "" },
  ],
  items: [],
  quests: [],
  regions: [
    { id: "north", name: "North", x: 2, y: 2, w: 10, h: 8, on_enter_text: "" },
    { id: "east", name: "East", x: 34, y: 3, w: 10, h: 8, on_enter_text: "" },
    {
      id: "south",
      name: "South",
      x: 18,
      y: 26,
      w: 10,
      h: 8,
      on_enter_text: "",
    },
  ],
  vars: {},
  intro_text: "",
};
const SPEC: Spec = {
  ...BASE,
  rivals: {
    factions: [
      { id: "us", name: "Us", player: true, region: "north", leader: "bea" },
      { id: "oak", name: "Oakvale", region: "east", leader: "bo" },
      { id: "ash", name: "Ashford", region: "south", leader: "ada" },
    ],
    max_turns: 30,
    hysteresis: 0.15,
  },
};
const RIVALS = SPEC.rivals as NonNullable<Spec["rivals"]>;
const MAP = P.generateMap(SPEC);
const B = R.boardOf(SPEC, MAP);
const fresh = () => R.initialRivals(SPEC, MAP, B);
const MILITARY = {
  economy: 0.2,
  expansion: 0.2,
  military: 0.5,
  diplomacy: 0.1,
};
const ECONOMY = { economy: 0.6, expansion: 0.2, military: 0.1, diplomacy: 0.1 };
const plan = (
  posture: Plan["posture"],
  target: string | null = null,
): Plan => ({
  posture,
  target,
  goals: ["hold on"],
});

function invariants(s: State) {
  for (const [id, f] of Object.entries(s.factions)) {
    for (const r of R.RESOURCES) {
      assert.ok(Number.isInteger(f.stock[r]), `${id}.${r} is an integer`);
      assert.ok(
        f.stock[r] >= 0 && f.stock[r] <= R.STOCK_CAP,
        `${id}.${r} in range`,
      );
    }
    assert.ok(f.workers >= 1 && f.soldiers >= 0);
    assert.ok(f.queue.length <= R.MAX_QUEUE);
    assert.ok(
      Math.abs(R.POSTURE_KEYS.reduce((a, k) => a + f.posture[k], 0) - 1) < 0.01,
    );
  }
  for (const [k, owner] of Object.entries(s.plots)) {
    assert.ok(owner in s.factions, `plot ${k} has a known owner`);
    const [px, py] = k.split(",").map(Number);
    assert.ok(px >= 0 && py >= 0 && px < B.cols && py < B.rows);
  }
  assert.ok(s.log.length <= R.MAX_LOG && s.offers.length <= R.MAX_OFFERS);
}

test("the board gives every village distinct home plots and its own villagers", () => {
  const s = fresh();
  const homes = Object.values(B.home).flat();
  assert.equal(new Set(homes).size, homes.length);
  for (const f of B.factions) assert.ok(B.home[f.id].length >= 1);
  assert.equal(B.player, "us");
  assert.ok(
    s.factions.oak.villagers.includes("bo"),
    "the leader works for their village",
  );
  invariants(s);
});

test("managers execute a posture deterministically, and the posture decides what they do", () => {
  const run = (posture: Plan["posture"]) => {
    let s = fresh();
    s = R.applyPlan(B, s, "oak", plan(posture, "ash"));
    s = R.runManagers(B, s, "oak");
    return s;
  };
  const a = run(MILITARY);
  assert.deepEqual(run(MILITARY), a, "same state and posture, same actions");
  const military = R.wants(
    B,
    R.applyPlan(B, fresh(), "oak", plan(MILITARY, "ash")),
    "oak",
  );
  const economy = R.wants(
    B,
    R.applyPlan(B, fresh(), "oak", plan(ECONOMY)),
    "oak",
  );
  const top = (w: typeof military) =>
    w[0].action.kind === "build" ? w[0].action.building : w[0].action.kind;
  assert.ok(["barracks", "train"].includes(top(military)), top(military));
  assert.ok(["farm", "lumber", "quarry"].includes(top(economy)), top(economy));
  assert.equal(
    a.factions.oak.actionsLeft,
    0,
    "the managers spend the day's actions",
  );
  assert.ok(
    a.factions.oak.queue.some((q) => q.kind === "barracks") ||
      a.factions.oak.soldiers > 1,
  );
  const ws = R.assignWorkers(
    SPEC,
    R.applyPlan(B, fresh(), "oak", plan(MILITARY)),
    "oak",
  );
  const jobs = ws.factions.oak.jobs;
  assert.equal(
    R.RESOURCES.reduce((n, r) => n + jobs[r], 0),
    ws.factions.oak.workers,
  );
  assert.ok(jobs.food >= 1, "workers keep the village fed first");
});

test("hysteresis holds the posture until the change passes the threshold", () => {
  let s = R.applyPlan(B, fresh(), "oak", plan(MILITARY, "ash"));
  assert.deepEqual(
    s.factions.oak.posture,
    R.normPosture(MILITARY),
    "the first plan always lands",
  );
  const near = {
    economy: 0.25,
    expansion: 0.2,
    military: 0.45,
    diplomacy: 0.1,
  };
  const held = R.applyPlan(B, s, "oak", plan(near, "us"));
  assert.equal(held, s, "a small change is ignored, target included");
  s = R.applyPlan(B, s, "oak", plan(ECONOMY));
  assert.deepEqual(s.factions.oak.posture, R.normPosture(ECONOMY));
  let flips = 0;
  let t = R.applyPlan(B, fresh(), "oak", plan(MILITARY));
  for (let i = 0; i < 10; i++) {
    const next = R.applyPlan(B, t, "oak", plan(i % 2 ? MILITARY : near));
    if (next !== t) flips++;
    t = next;
  }
  assert.equal(flips, 0, "a dithering strategist does not move the village");
  let loose = R.applyPlan(B, fresh(), "oak", plan(MILITARY), 0);
  let moves = 0;
  for (let i = 0; i < 10; i++) {
    const next = R.applyPlan(B, loose, "oak", plan(i % 2 ? MILITARY : near), 0);
    if (next !== loose) moves++;
    loose = next;
  }
  assert.equal(moves, 10, "with no threshold every flip lands (control)");
});

test("the strategist digest stays bounded however long the game runs", () => {
  const spec: Spec = {
    ...SPEC,
    rivals: {
      ...RIVALS,
      factions: RIVALS.factions.map((f) => ({
        ...f,
        name: f.name.padEnd(60, "x"),
      })),
    },
  };
  const b = R.boardOf(spec, MAP);
  let s = R.initialRivals(spec, MAP, b);
  const sizes: number[] = [];
  for (let i = 0; i < 60; i++) {
    s = R.endTurn(spec, b, s, {
      oak: plan(MILITARY, "ash"),
      ash: plan(ECONOMY),
    });
    s = { ...s, log: [...s.log, { turn: s.turn, text: "y".repeat(500) }] };
    s.over = null;
    const req = R.strategistRequest(spec, b, s, "oak");
    sizes.push(JSON.stringify(req.digest).length);
    assert.ok(req.digest.recent.length <= R.DIGEST_LOG);
  }
  assert.ok(
    Math.max(...sizes) <= R.DIGEST_MAX_CHARS,
    `digest ${Math.max(...sizes)} chars`,
  );
  assert.ok(
    Math.max(...sizes) - Math.min(...sizes) < 400,
    "size does not grow with the turn count",
  );
});

test("a fake strategist drives a 10-turn game to a stable state, one call per AI village per turn", async () => {
  const play = async () => {
    let s = fresh();
    const calls: {
      fid: string;
      req: { task: string; digest: { turn: number } };
    }[] = [];
    const postures: string[] = [];
    for (let turn = 0; turn < 10; turn++) {
      const out = await R.playTurn(SPEC, B, s, async (fid, req) => {
        calls.push({ fid, req: req as (typeof calls)[number]["req"] });
        const wobble = (turn % 3) * 0.02;
        return fid === "oak"
          ? plan({ ...MILITARY, economy: MILITARY.economy + wobble }, "ash")
          : plan({ ...ECONOMY, military: ECONOMY.military + wobble });
      });
      assert.equal(out.calls, 2);
      s = out.state;
      invariants(s);
      postures.push(
        JSON.stringify([s.factions.oak.posture, s.factions.ash.posture]),
      );
    }
    return { s, calls, postures };
  };
  const a = await play();
  assert.equal(a.calls.length, 20, "2 AI villages x 10 turns");
  assert.ok(
    a.calls.every((c) => c.req.task === "rivals_strategist" && c.fid !== "us"),
  );
  assert.deepEqual(
    a.calls.map((c) => c.req.digest.turn),
    Array.from({ length: 10 }, (_, i) => [i + 1, i + 1]).flat(),
  );
  assert.equal(a.s.turn, 11);
  assert.equal(
    new Set(a.postures).size,
    1,
    "postures settle on turn 1 and stay",
  );
  const b = await play();
  assert.deepEqual(b.s, a.s, "the same strategist answers give the same game");
  const built = R.BUILDING_IDS.reduce(
    (n, k) => n + a.s.factions.ash.buildings[k],
    0,
  );
  assert.ok(built >= 3, "the economy village builds");
  assert.ok(R.plotsOf(a.s, "ash").length > B.home.ash.length, "and expands");
  const silent = await R.playTurn(SPEC, B, fresh(), null);
  assert.equal(
    silent.calls,
    0,
    "no chat model: the managers play on their own",
  );
  const failing = await R.playTurn(SPEC, B, fresh(), async () => {
    throw new Error("offline");
  });
  assert.equal(failing.state.turn, 2, "a failed call keeps the current plan");
});

test("diplomacy offers become engine actions only when the engine would do them", () => {
  const s = fresh();
  const trade = R.offerAction(B, s, "oak", {
    kind: "trade",
    give: "stone",
    give_amount: 2,
    want: "food",
    want_amount: 3,
  });
  assert.deepEqual(trade, {
    kind: "trade",
    with: "oak",
    give: "food",
    giveAmount: 3,
    want: "stone",
    wantAmount: 2,
  });
  const done = R.act(B, s, "us", trade as NonNullable<typeof trade>);
  assert.ok(done.ok);
  assert.equal(done.state.factions.us.stock.food, s.factions.us.stock.food - 3);
  assert.equal(
    done.state.factions.oak.stock.stone,
    s.factions.oak.stock.stone - 2,
  );
  const big = {
    kind: "trade" as const,
    give: "gold",
    give_amount: 50,
    want: "food",
    want_amount: 1,
  };
  assert.equal(
    R.offerAction(B, s, "oak", big),
    null,
    "more than the leader has",
  );
  assert.equal(
    R.offerAction(B, s, "oak", { kind: "peace" }),
    null,
    "not at war",
  );
  assert.deepEqual(R.offerAction(B, s, "oak", { kind: "ally" }), {
    kind: "ally",
    with: "oak",
  });
  const hostile = R.applyPlan(B, s, "oak", plan(MILITARY, "us"));
  assert.equal(
    R.offerAction(B, hostile, "oak", { kind: "ally" }),
    null,
    "against its intent",
  );
  assert.equal(R.offerAction(B, s, "oak", { kind: "none" }), null);
  assert.equal(R.offerAction(B, s, "oak", null), null);
  const war = R.act(B, s, "us", { kind: "raid", target: "oak" }).state;
  assert.equal(R.relationOf(war, "us", "oak"), "war");
  assert.deepEqual(R.offerAction(B, war, "oak", { kind: "peace" }), {
    kind: "peace",
    with: "oak",
  });
  assert.equal(R.offerAction(B, war, "oak", big), null);
});

test("AI proposals to the player run through the same gate when accepted", () => {
  let s = R.applyPlan(
    B,
    fresh(),
    "oak",
    plan({ economy: 0.2, expansion: 0.1, military: 0.1, diplomacy: 0.6 }),
  );
  s = R.runManagers(B, s, "oak");
  const offer = s.offers.find((o) => o.from === "oak");
  assert.ok(offer && offer.action.kind === "ally");
  const r = R.acceptOffer(B, s, offer.id);
  assert.ok(r.ok);
  assert.equal(R.relationOf(r.state, "us", "oak"), "ally");
  assert.equal(
    R.blocked(B, r.state, "us", { kind: "raid", target: "oak" }),
    "you are allies",
  );
  assert.equal(R.declineOffer(s, offer.id).offers.length, s.offers.length - 1);
  let tired = s;
  tired = R.act(B, tired, "us", { kind: "build", building: "farm" }).state;
  tired = R.act(B, tired, "us", { kind: "claim" }).state;
  assert.equal(tired.factions.us.actionsLeft, 0);
  assert.equal(
    R.acceptOffer(B, tired, offer.id).ok,
    false,
    "accepting costs an action",
  );
});

test("a game saves and loads through the world save, and junk starts fresh", () => {
  let s = fresh();
  for (let i = 0; i < 4; i++)
    s = R.endTurn(SPEC, B, s, { oak: plan(MILITARY, "ash") });
  const ws = { ...W.initialWorldState(SPEC), rivals: s };
  const back = W.deserialize(SPEC, JSON.parse(JSON.stringify(W.serialize(ws))));
  assert.deepEqual(back?.rivals, s);
  const junk = W.deserialize(SPEC, {
    ...W.serialize(ws),
    rivals: { turn: "x", plots: 5 },
  });
  assert.deepEqual(junk?.rivals, fresh());
  const odd = R.rivalsOf(
    {
      ...s,
      plots: { ...s.plots, "99,99": "oak", "1,1": "ghost" },
      factions: { ...s.factions, ghost: s.factions.oak },
    },
    SPEC,
    MAP,
  );
  assert.ok(
    !("ghost" in odd.factions) &&
      !("99,99" in odd.plots) &&
      !("1,1" in odd.plots && odd.plots["1,1"] === "ghost"),
  );
});

test("worlds without rivals are unchanged", () => {
  assert.equal(R.rivalsOn(BASE), false);
  const st = W.initialWorldState(BASE);
  assert.ok(!("rivals" in st));
  const back = W.deserialize(BASE, { ...W.serialize(st), rivals: fresh() });
  assert.ok(
    back && !("rivals" in back),
    "a stray rivals save on a plain world is dropped",
  );
  assert.deepEqual(
    W.deserialize(BASE, JSON.parse(JSON.stringify(W.serialize(st)))),
    st,
  );
});

test("the game ends: domination or the last day", () => {
  let s = fresh();
  const land = R.landPlots(B);
  const plots = { ...s.plots };
  let n = Object.values(plots).filter((o) => o === "ash").length;
  for (
    let i = 0;
    i < B.cols * B.rows && n < Math.ceil(land * R.WIN_SHARE);
    i++
  ) {
    const k = R.plotKey(i % B.cols, Math.floor(i / B.cols));
    if (!plots[k] && B.land[i]) {
      plots[k] = "ash";
      n++;
    }
  }
  s = R.endTurn(SPEC, B, { ...s, plots }, {});
  assert.equal(s.over?.winner, "ash");
  const short: Spec = { ...SPEC, rivals: { ...RIVALS, max_turns: 5 } };
  let t = R.initialRivals(short, MAP, B);
  for (let i = 0; i < 6; i++) t = R.endTurn(short, B, t, {});
  assert.ok(t.over && /after 5 days/.test(t.over.reason));
  assert.equal(R.endTurn(short, B, t, {}), t, "nothing moves after the end");
});
