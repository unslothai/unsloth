// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const L = await import("../src/features/flows/player/world/life-sim.ts");
const S = await import("../src/features/flows/player/world/world-sim.ts");
const W = await import("../src/features/flows/player/world/world-state.ts");
const A = await import("../src/features/flows/player/world/world-actions.ts");
const { budgetSpec, runBudget } = await import("./helpers/life-budget.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;
type Mind = import("../src/features/flows/player/world/world-sim.ts").NpcMind;
type Option = import("../src/features/flows/player/world/world-sim.ts").PlanOption;

const LIFE = {
  options: true,
  plan_cache: 2,
  reflect_at: 20,
  rate_batch: 3,
  relationships: true,
};
const spec = (life: Record<string, unknown> = LIFE) =>
  budgetSpec(3, life) as unknown as Spec;
const areas = () => A.placeAreas(spec());
const NEEDS = { energy: 80, hunger: 80, social: 80, fun: 80 };

const opt = (
  until: Option["until"],
  kind: Option["kind"] = "work",
): Option => ({
  activity: "Work",
  place: "forge",
  kind,
  until,
});

function ctx(over: Partial<Parameters<typeof L.untilMet>[1]> = {}) {
  return {
    npcId: "v0",
    clock: 600,
    needs: NEEDS,
    held: {},
    at: { x: 0, y: 0 },
    areas: areas(),
    ...over,
  };
}

test("each termination condition fires only when met", () => {
  assert.equal(L.untilMet({ type: "time", at: 601 }, ctx()), false);
  assert.equal(L.untilMet({ type: "time", at: 600 }, ctx()), true);
  const need = { type: "need", need: "hunger", above: 85 } as const;
  assert.equal(L.untilMet(need, ctx()), false);
  assert.equal(
    L.untilMet(need, ctx({ needs: { ...NEEDS, hunger: 85 } })),
    true,
  );
  assert.equal(L.untilMet(need, ctx({ needs: undefined })), false);
  const item = { type: "item", item: "pie" } as const;
  assert.equal(L.untilMet(item, ctx({ held: { pie: "v1" } })), false);
  assert.equal(L.untilMet(item, ctx({ held: { pie: "v0" } })), true);
  const place = { type: "place", place: "forge" } as const;
  assert.equal(L.untilMet(place, ctx()), false);
  assert.equal(L.untilMet(place, ctx({ at: { x: 23, y: 21 } })), true);
});

test("options run in code until they end; stale non-time options give up, timed ones do not", () => {
  const mind: Mind = {
    ...S.emptyMind(),
    options: [
      opt({ type: "time", at: 700 }),
      opt({ type: "need", need: "fun", above: 99 }, "wander"),
      opt({ type: "place", place: "forge" }),
    ],
    optionAt: 0,
    optionSince: 0,
  };
  assert.equal(L.advanceOptions(mind, ctx({ clock: 650 })), null);
  assert.deepEqual(L.advanceOptions(mind, ctx({ clock: 700 })), {
    optionAt: 1,
    optionSince: 700,
  });
  const second = { ...mind, optionAt: 1, optionSince: 700 };
  assert.equal(L.advanceOptions(second, ctx({ clock: 700 + 239 })), null);
  assert.deepEqual(L.advanceOptions(second, ctx({ clock: 700 + 240 })), {
    optionAt: 2,
    optionSince: 940,
  });
  assert.equal(L.advanceOptions(S.emptyMind(), ctx()), null);
  const block = L.lifeBlock(mind, 650);
  assert.deepEqual(block, {
    start: 0,
    minutes: 700,
    activity: "Work",
    place: "forge",
  });
});

function stepper(mind: Mind, clock: number, needs = NEEDS) {
  const said: string[] = [];
  const g = {
    playerTile: () => ({ x: 0, y: 0 }),
    npcTile: () => ({ x: 0, y: 0 }),
    setNpcPlace: () => {},
    setNpcStatus: () => {},
    say: (_: string, t: string) => said.push(t),
  };
  const sp = spec();
  return A.villagerStep({
    g,
    spec: sp,
    sim: sp.simulation as NonNullable<Spec["simulation"]>,
    npcId: "v0",
    mind: { ...mind, needs },
    clock,
    offers: [],
    areas: areas(),
  });
}

test("villagers do the running option, fall back to needs when one is critical, then move on", () => {
  const mind: Mind = {
    ...S.emptyMind(),
    options: [
      opt({ type: "time", at: 700 }),
      opt({ type: "time", at: 900 }, "eat"),
    ],
    optionAt: 0,
    optionSince: 600,
  };
  const out = stepper(mind, 610);
  assert.equal(out?.doing.kind, "work");
  assert.equal(out?.doing.label, "Work");
  const starving = stepper(mind, 610, { ...NEEDS, hunger: 5 });
  assert.notEqual(starving?.doing.kind, "work");
  const next = stepper({ ...mind, doing: out?.doing }, 700);
  assert.equal(next?.optionAt, 1);
  assert.equal(next?.doing.kind, "eat");
});

test("plan due: classic is once a day; options replan when exhausted early, a notable event lands, never late", () => {
  const off = L.lifeOf(spec({}));
  const on = L.lifeOf(spec());
  const m: Mind = { ...S.emptyMind(), planDay: 0 };
  assert.equal(L.planDue(off, m, 600), false);
  assert.equal(L.planDue(off, m, 1440), true);
  const done: Mind = {
    ...m,
    options: [opt({ type: "time", at: 500 })],
    optionAt: 1,
    planAt: 500,
  };
  assert.equal(L.planDue(on, done, 540), false, "gap since the last plan");
  assert.equal(L.planDue(on, done, 600), true);
  assert.equal(
    L.planDue(on, done, L.LATE_REPLAN),
    false,
    "evening runs on needs alone",
  );
  assert.equal(L.planDue(off, done, 600), false);
  const busy: Mind = { ...done, optionAt: 0 };
  assert.equal(L.planDue(on, busy, 600), false);
  const noted = S.remember(
    busy,
    [{ text: "Fire at the mill!", importance: 9 }],
    600,
  );
  assert.equal(noted.replan, true);
  assert.equal(L.planDue(on, noted, 600), true);
  assert.equal(
    L.planDue(on, S.remember(busy, [{ text: "meh", importance: 3 }], 600), 600),
    false,
  );
  assert.equal(
    S.remember(S.emptyMind(), [{ text: "x", importance: 9 }], 0).replan,
    undefined,
  );
});

test("a notable event during an in-flight plan call stays pending after the answer or a failure", () => {
  const on = L.lifeOf(spec());
  const sent: Mind = { ...S.emptyMind(), planDay: 0, options: [], planAt: 500 };
  const askedSeq = sent.replanSeq ?? 0;
  const noted = S.remember(
    sent,
    [{ text: "Go to the pier now, the boat is leaving.", importance: 10 }],
    600,
  );
  assert.equal(noted.replan, true);
  const answer = { options: [opt({ type: "time", at: 1200 })] };
  const planned: Mind = {
    ...noted,
    ...L.lifePlanned(noted, answer, 610, askedSeq),
  };
  assert.equal(planned.options?.length, 1);
  assert.equal(planned.replan, true);
  assert.equal(L.planDue(on, planned, 700), true);
  const failed: Mind = {
    ...noted,
    ...L.lifePlanned(noted, null, 610, askedSeq),
  };
  assert.equal(failed.replan, true);
  assert.equal(L.planDue(on, failed, 700), true);
  const seen: Mind = {
    ...planned,
    ...L.lifePlanned(planned, answer, 700, planned.replanSeq ?? 0),
  };
  assert.equal(seen.replan, false);
  assert.equal(L.planDue(on, seen, 800), false);
});

test("plan cache: miss, store, personal hit, LRU bound and replan bypass", () => {
  const sp = spec();
  const cfg = L.lifeOf(sp);
  let s = W.initialWorldState(sp);
  assert.deepEqual(s.life, { cache: [] });
  assert.equal(L.cachedPlan(sp, cfg, s, "v0", 600, "forge"), null);
  const out = {
    plan: [{ start: 600, minutes: 60, activity: "Work", place: "forge" }],
    options: [
      opt({ type: "time", at: 720 }),
      opt({ type: "need", need: "fun", above: 70 }, "wander"),
    ],
  };
  const key = L.planCacheKey(cfg, sp, "v0", s.npc.v0, 600, "forge");
  assert.equal(key, `baker|ok|morning|forge`);
  s = L.storePlan(s, cfg, key, out, 600);
  assert.equal(s.life?.cache[0].options?.[0].until.type, "time");
  const later = 1440 * 2 + 600;
  const hit = L.cachedPlan(sp, cfg, s, "v0", later, "forge");
  assert.ok(hit);
  const after = hit(s).npc.v0;
  assert.equal(after.planDay, 2);
  assert.equal(after.optionAt, 0);
  const at = (after.options?.[0].until as { at: number }).at;
  assert.ok(Math.abs(at - (later + 120)) <= 20 && at > later);
  assert.equal(after.memory.at(-1)?.kind, "plan");
  const again = L.cachedPlan(sp, cfg, s, "v0", later, "forge");
  assert.deepEqual(
    again?.(s).npc.v0.options,
    after.options,
    "seeded, so stable",
  );
  assert.equal(
    L.cachedPlan(sp, cfg, s, "v1", later, "forge"),
    null,
    "other role",
  );
  const noted = W.withMind(s, "v0", (m) => ({ ...m, replan: true }));
  assert.equal(L.cachedPlan(sp, cfg, noted, "v0", later, "forge"), null);
  assert.equal(
    L.planCacheKey(cfg, sp, "v0", noted.npc.v0, later, "forge"),
    null,
  );

  let life = undefined;
  for (const k of ["a", "b", "c"])
    life = L.cachePut(life, 2, { key: k, plan: [] });
  assert.deepEqual(
    life?.cache.map((e) => e.key),
    ["b", "c"],
  );
  life = L.cachePut(life, 2, { key: "b", plan: [] });
  life = L.cachePut(life, 2, { key: "d", plan: [] });
  assert.deepEqual(
    life?.cache.map((e) => e.key),
    ["b", "d"],
  );
  assert.equal(L.storePlan(s, L.lifeOf(spec({})), key, out, 600), s);
});

test("plan cache, options, pair memory and importance sums survive a save round trip", () => {
  const sp = spec();
  const cfg = L.lifeOf(sp);
  let s = W.initialWorldState(sp);
  s = L.storePlan(
    s,
    cfg,
    "k",
    { plan: [{ start: 600, minutes: 60, activity: "x", place: "forge" }] },
    600,
  );
  s = W.withMind(s, "v0", (m) => ({
    ...S.remember(
      m,
      [{ text: "heard it", importance: 5, source: ["v1", "v2"] }],
      600,
    ),
    options: [opt({ type: "item", item: "pie" })],
    optionAt: 0,
    optionSince: 600,
    planAt: 600,
    rel: { v1: { affinity: 3, facts: ["bakes"], talked: 600 } },
  }));
  const back = W.deserialize(sp, JSON.parse(JSON.stringify(W.serialize(s))));
  assert.deepEqual(back, s);
  assert.equal(back?.npc.v0.since, 5);
});

test("old saves and worlds without the new fields load and behave as before", () => {
  const classic = spec({});
  const fresh = W.initialWorldState(classic);
  assert.equal(fresh.life, undefined);
  for (const m of Object.values(fresh.npc)) {
    assert.deepEqual(Object.keys(m).sort(), [
      "lastChat",
      "memory",
      "needs",
      "nextId",
      "plan",
      "planDay",
    ]);
  }
  const old = JSON.parse(JSON.stringify(W.serialize(fresh)));
  assert.deepEqual(W.deserialize(classic, old), fresh);
  const upgraded = W.deserialize(spec(), old);
  assert.deepEqual(upgraded?.life, { cache: [] });
  assert.equal(upgraded?.npc.v0.since, 0);
  assert.deepEqual(upgraded?.npc.v0.rel, {});
  const cfg = L.lifeOf(classic);
  assert.deepEqual(cfg, {
    options: false,
    cache: 0,
    reflectAt: 0,
    rateBatch: 8,
    relationships: false,
  });
  assert.deepEqual(L.planAsk(cfg, fresh.npc.v0), {});
  assert.deepEqual(L.chatAsk(cfg, fresh, "v0", "v1"), {});
  assert.deepEqual(
    L.lifeJobs(
      classic,
      cfg,
      fresh,
      async () => ({}),
      () => {},
    ),
    [],
  );
  assert.equal(L.lifeOf(spec({ options: true, needs: false })).options, false);
});

test("reflection waits for summed importance to cross the threshold, then resets", async () => {
  const sp = spec();
  const cfg = L.lifeOf(sp);
  let s = W.initialWorldState(sp);
  s = W.withMind(s, "v0", (m) =>
    S.remember(
      m,
      [
        { text: "a", importance: 9 },
        { text: "b", importance: 9 },
      ],
      600,
    ),
  );
  s = W.withMind(s, "v0", (m) =>
    S.remember(m, [{ text: "guess", importance: 9, guess: true }], 600),
  );
  assert.equal(s.npc.v0.since, 18);
  assert.deepEqual(
    L.lifeJobs(
      sp,
      cfg,
      s,
      async () => ({}),
      () => {},
    ),
    [],
  );
  s = W.withMind(s, "v0", (m) =>
    S.remember(m, [{ text: "c", importance: 2 }], 600),
  );
  const failing = L.lifeJobs(
    sp,
    cfg,
    s,
    async () => {
      throw new Error("model down");
    },
    (fn) => {
      s = fn(s);
    },
  );
  await assert.rejects(failing[0].run(new AbortController().signal));
  assert.equal(s.npc.v0.since, 20, "a failed reflection keeps the sum");
  const asked: Record<string, unknown>[] = [];
  const jobs = L.lifeJobs(
    sp,
    cfg,
    s,
    async (state) => {
      asked.push(state);
      return { insights: [{ text: "I am busy", importance: 7 }] };
    },
    (fn) => {
      s = fn(s);
    },
  );
  assert.deepEqual(
    jobs.map((j) => j.key),
    ["reflect:v0"],
  );
  await jobs[0].run(new AbortController().signal);
  assert.equal(asked.length, 1);
  assert.equal(asked[0].task, "reflect");
  assert.ok(
    !(asked[0].memory as { text: string }[]).some((r) => r.text === "guess"),
  );
  assert.equal(s.npc.v0.since, 0);
  assert.equal(s.npc.v0.memory.at(-1)?.kind, "reflection");
  assert.equal(s.npc.v0.memory.at(-1)?.text, "I am busy");
  const late = L.lifeJobs(
    sp,
    cfg,
    W.withMind(s, "v0", (m) => ({ ...m, since: 25 })),
    async () => {
      s = W.withMind(s, "v0", (m) =>
        S.remember(m, [{ text: "d", importance: 10 }], 600),
      );
      return { insights: [] };
    },
    (fn) => {
      s = fn(s);
    },
  );
  s = W.withMind(s, "v0", (m) => ({ ...m, since: 25 }));
  await late[0].run(new AbortController().signal);
  assert.equal(
    s.npc.v0.since,
    10,
    "notes added during the call wait for the next reflection",
  );
});

test("importance is rated in one batched call across villagers", async () => {
  const sp = spec();
  const cfg = L.lifeOf(sp);
  let s = W.initialWorldState(sp);
  const g = (t: string) => ({ text: t, importance: 3, guess: true });
  s = W.withMind(s, "v0", (m) => S.remember(m, [g("w1"), g("w2")], 600));
  assert.deepEqual(
    L.lifeJobs(
      sp,
      cfg,
      s,
      async () => ({}),
      () => {},
    ),
    [],
  );
  s = W.withMind(s, "v1", (m) => S.remember(m, [g("w3"), g("w4")], 600));
  const calls: Record<string, unknown>[] = [];
  const jobs = L.lifeJobs(
    sp,
    cfg,
    s,
    async (state) => {
      calls.push(state);
      return { ratings: [8, 2, 10] };
    },
    (fn) => {
      s = fn(s);
    },
  );
  assert.deepEqual(
    jobs.map((j) => j.kind),
    ["rate"],
  );
  await jobs[0].run(new AbortController().signal);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].texts, ["w1", "w2", "w3"]);
  const v0 = s.npc.v0.memory;
  assert.deepEqual(
    v0.map((r) => [r.text, r.importance, r.guess]),
    [
      ["w1", 8, undefined],
      ["w2", 2, undefined],
    ],
  );
  assert.equal(s.npc.v0.since, 10);
  assert.equal(s.npc.v1.memory[0].importance, 10);
  assert.equal(
    s.npc.v1.memory[1].guess,
    true,
    "beyond the batch, waits for the next call",
  );
  assert.equal(s.npc.v1.since, 10);
});

test("pair memory: chats and gifts move affinity and keep a few facts", () => {
  const sp = spec();
  const cfg = L.lifeOf(sp);
  let s = W.initialWorldState(sp);
  s = { ...s, clock: 700 };
  s = L.afterChat(cfg, s, "v0", "v1", {
    a_feels: { affinity: 2, fact: "fishes at dawn" },
    b_feels: { affinity: -1, fact: "" },
  });
  assert.deepEqual(s.npc.v0.rel?.v1, {
    affinity: 2,
    facts: ["fishes at dawn"],
    talked: 700,
  });
  assert.deepEqual(s.npc.v1.rel?.v0, { affinity: -1, facts: [], talked: 700 });
  let m = s.npc.v0;
  for (const f of ["a", "b", "c", "d"]) m = L.feel(m, "v1", 5, f);
  assert.deepEqual(m.rel?.v1, {
    affinity: 10,
    facts: ["b", "c", "d"],
    talked: 700,
  });
  const given = W.applyReply(sp, { ...s, inventory: ["pie"] }, "v2", {
    text: "Thanks!",
    take_item: "pie",
  });
  assert.deepEqual(given.state.npc.v2.rel?.player, {
    affinity: 2,
    facts: ["gave me Pie"],
  });
  const off = spec({});
  const classic = W.applyReply(
    off,
    { ...W.initialWorldState(off), inventory: ["pie"] },
    "v2",
    { text: "x", take_item: "pie" },
  );
  assert.equal(classic.state.npc.v2.rel, undefined);
  assert.equal(
    L.afterChat(L.lifeOf(off), s, "v0", "v1", { a_feels: { affinity: 3 } }),
    s,
  );
});

test("rumours carry their source chain and answer who told you", () => {
  const sp = spec();
  let s = W.initialWorldState(sp);
  s = W.withMind(s, "v2", (m) =>
    S.remember(
      m,
      [{ text: "Mill fire", importance: 8, source: ["player"] }],
      600,
    ),
  );
  const fromV2 = L.rumours(s, "v2", [
    { text: "Mill fire", importance: 8, from: "m0" },
  ]);
  assert.deepEqual(fromV2[0].source, ["v2", "player"]);
  s = W.withMind(s, "v1", (m) => S.remember(m, fromV2, 610));
  const fromV1 = L.rumours(s, "v1", [
    { text: "Mill fire", importance: 8, from: "m0" },
    { text: "own news", importance: 3 },
  ]);
  assert.deepEqual(
    fromV1.map((n) => n.source),
    [["v1", "v2", "player"], ["v1"]],
  );
  assert.ok(!("from" in fromV1[0]));
  s = W.withMind(s, "v0", (m) => S.remember(m, fromV1, 620));
  assert.deepEqual(L.whoToldMe(sp, s, "v0", "m0"), [
    "Villager 1",
    "Villager 2",
    "Pip",
  ]);
  const long = L.rumours(
    {
      ...s,
      npc: {
        ...s.npc,
        v1: {
          ...s.npc.v1,
          memory: [{ ...s.npc.v1.memory[0], source: ["a", "b", "c", "d"] }],
        },
      },
    },
    "v1",
    [{ text: "x", importance: 1, from: "m0" }],
  );
  assert.equal(long[0].source?.length, 4);
});

test("small talk replaces a model chat only for pairs who already talked today with nothing notable", () => {
  const sp = spec();
  const cfg = L.lifeOf(sp);
  let s = W.initialWorldState(sp);
  const said: string[] = [];
  const talk = (st: typeof s, clock: number) =>
    L.smallTalk({
      cfg,
      spec: sp,
      s: st,
      a: "v0",
      b: "v1",
      clock,
      say: (_, t) => said.push(t),
      update: (fn) => {
        s = fn(s);
      },
    });
  assert.equal(talk(s, 600), false, "first meeting goes to the model");
  s = L.afterChat(cfg, { ...s, clock: 600 }, "v0", "v1", {});
  assert.equal(talk(s, 700), true);
  assert.equal(said.length, 1);
  assert.equal(talk(s, 1440 + 600), false, "a new day goes to the model");
  const news = W.withMind(s, "v1", (m) =>
    S.remember(m, [{ text: "Fire!", importance: 9 }], 650),
  );
  assert.equal(talk(news, 700), false, "notable news goes to the model");
  const heard = W.withMind(s, "v1", (m) =>
    S.remember(m, [{ text: "Told in that chat", importance: 9 }], 600),
  );
  assert.equal(talk(heard, 700), true, "notes from that chat are not news");
  assert.equal(
    L.smallTalk({
      cfg: L.lifeOf(spec({})),
      spec: sp,
      s,
      a: "v0",
      b: "v1",
      clock: 700,
      say: () => {},
      update: () => {},
    }),
    false,
  );
});

test("the queue counts background calls by kind and refunds jobs answered without a call", async () => {
  let now = 0;
  const q = new S.BackgroundQueue(2, () => now);
  const done: string[] = [];
  q.push({
    key: "a",
    kind: "plan",
    run: async () => {
      done.push("a");
      return false;
    },
  });
  q.push({
    key: "b",
    kind: "chat",
    run: async () => {
      done.push("b");
    },
  });
  q.push({
    key: "c",
    kind: "rate",
    run: async () => {
      done.push("c");
    },
  });
  q.pump();
  for (let i = 0; i < 10; i++) await new Promise((r) => setImmediate(r));
  assert.deepEqual(done, ["a", "b", "c"]);
  assert.deepEqual(q.calls, { plan: 0, chat: 1, rate: 1 });
  assert.equal(q.callsLabel(), "2 background calls (plan 0, chat 1, rate 1)");
  now = 1;
  assert.equal(q.allows(), false);
});

test("budget: 12 villagers over a simulated day stay under 1 background call/s, and fewer than classic", async () => {
  const life = {
    options: true,
    plan_cache: 64,
    reflect_at: 150,
    relationships: true,
  };
  const classic = await runBudget({ perMinute: Number.POSITIVE_INFINITY });
  const head = await runBudget({ life, perMinute: Number.POSITIVE_INFINITY });
  assert.equal(classic.seconds, 720);
  assert.ok(head.perSecond < 1, `${head.perSecond}/s`);
  assert.ok(
    head.total < classic.total / 2,
    `${head.total} vs ${classic.total}`,
  );
  assert.ok((head.calls.chat ?? 0) < (classic.calls.chat ?? 0));
  assert.ok((head.calls.plan ?? 0) <= (classic.calls.plan ?? 0));
  assert.ok(
    head.smallTalk > 0 && head.cacheHits > 0 && (head.calls.rate ?? 0) > 0,
  );
  const throttled = await runBudget({ life });
  assert.ok(throttled.perSecond <= 4 / 60 + 1e-9);
});

test("a save with option plans drops them when the world turns option planning off", () => {
  const sp = spec();
  const s = W.withMind(W.initialWorldState(sp), "v0", (m) => ({
    ...m,
    options: [opt({ type: "item", item: "pie" })],
    optionAt: 0,
    optionSince: 600,
    planAt: 600,
    replan: true,
  }));
  const saved = JSON.parse(JSON.stringify(W.serialize(s)));
  const off = W.deserialize(spec({ ...LIFE, options: false }), saved);
  assert.equal(off?.npc.v0.options, undefined);
  assert.equal(off?.npc.v0.optionAt, undefined);
  assert.equal(off?.npc.v0.replan, undefined);
  assert.equal(off?.npc.v0.planAt, 600);
  assert.equal(W.deserialize(sp, saved)?.npc.v0.options?.length, 1);
});

test("a cached plan with options installs only the plan when option planning is off", () => {
  const sp = spec();
  const cfg = L.lifeOf(sp);
  let s = W.initialWorldState(sp);
  const key = L.planCacheKey(cfg, sp, "v0", s.npc.v0, 600, "forge");
  s = L.storePlan(
    s,
    cfg,
    key,
    {
      plan: [{ start: 600, minutes: 60, activity: "Work", place: "forge" }],
      options: [opt({ type: "time", at: 720 })],
    },
    600,
  );
  const off = { ...cfg, options: false };
  const hit = L.cachedPlan(sp, off, s, "v0", 1440 * 2 + 600, "forge");
  assert.ok(hit);
  const after = hit(s).npc.v0;
  assert.equal(after.planDay, 2);
  assert.equal(after.options, undefined);
});
