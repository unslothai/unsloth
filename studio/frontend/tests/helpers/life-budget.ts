// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Headless copy of the world player's village tick with a fake model, to count background calls.
// Imports only modules that exist before life-sim.ts, so the same file also measures older trees.

const WORLD = "../../src/features/flows/player/world/";

export interface BudgetResult {
  seconds: number;
  calls: Record<string, number>;
  total: number;
  perSecond: number;
  /** Real seconds until every villager had a plan for the first day. */
  allPlannedAt: number | null;
  chats: number;
  smallTalk: number;
  cacheHits: number;
}

const JOBS = ["baker", "farmer", "fisher", "smith", "merchant", "teacher"];

export function budgetSpec(villagers: number, life: Record<string, unknown>) {
  const places = [
    { id: "home", name: "Cottages", x: 2, y: 2, w: 6, h: 4 },
    { id: "cafe", name: "Cafe", x: 12, y: 2, w: 5, h: 4 },
    { id: "market", name: "Market", x: 22, y: 2, w: 6, h: 4 },
    { id: "square", name: "Town square", x: 12, y: 12, w: 6, h: 5 },
    { id: "park", name: "Park", x: 2, y: 20, w: 8, h: 6 },
    { id: "forge", name: "Forge", x: 22, y: 20, w: 5, h: 4 },
  ];
  return {
    title: "Budget Vale",
    seed: 3,
    width: 32,
    height: 28,
    tile_size: 32,
    biome: "town",
    tileset: {},
    player: { name: "Pip" },
    npcs: Array.from({ length: villagers }, (_, i) => ({
      id: `v${i}`,
      name: `Villager ${i}`,
      persona: `a cheerful ${JOBS[i % JOBS.length]}`,
      greeting: "hi",
    })),
    items: [{ id: "pie", name: "Pie", description: "" }],
    quests: [],
    regions: [],
    vars: {},
    intro_text: "",
    entry_points: { npc_reply: "r", npc_plan: "p", npc_chat: "c" },
    simulation: {
      enabled: true,
      minutes_per_second: 2,
      start_minute: 480,
      places,
      chat_cooldown_min: 120,
      calls_per_minute: 4,
      needs: true,
      ...life,
    },
  };
}

export function fakeModel(counts: Record<string, number>) {
  let n = 0;
  return async (
    entry: string,
    state: Record<string, any>,
  ): Promise<Record<string, unknown>> => {
    n++;
    if (entry === "c") {
      counts.chat = (counts.chat ?? 0) + 1;
      return {
        chat: {
          a: state.a,
          b: state.b,
          lines: [
            { npc: state.a, text: "Hello there." },
            { npc: state.b, text: "Hi!" },
          ],
          a_learns:
            n % 3 === 0
              ? [
                  {
                    text: `News ${n} from ${state.b}`,
                    importance: 6,
                    from: state.memory_b?.[0]?.id,
                  },
                ]
              : [],
          b_learns: [],
          a_recalled: [],
          b_recalled: [],
          a_feels: { affinity: 1, fact: "likes bread" },
          b_feels: { affinity: 1, fact: "" },
        },
      };
    }
    const task = state.task ?? "plan";
    counts[task] = (counts[task] ?? 0) + 1;
    if (task === "rate")
      return {
        plan: {
          ratings: state.texts.map((_: string, i: number) => 3 + (i % 5)),
        },
      };
    if (task === "reflect")
      return {
        plan: { insights: [{ text: "I value my neighbours", importance: 6 }] },
      };
    const clock = state.clock as number;
    const day0 = Math.floor(clock / 1440) * 1440;
    if (state.options) {
      const at = (m: number) => ({ type: "time", at: day0 + m });
      return {
        plan: {
          plan: [
            { start: clock, minutes: 120, activity: "Work", place: "forge" },
          ],
          options: [
            { activity: "Work", place: "forge", kind: "work", until: at(720) },
            {
              activity: "Lunch",
              place: "cafe",
              kind: "eat",
              until: { type: "need", need: "hunger", above: 85 },
            },
            {
              activity: "Chat",
              place: "square",
              kind: "socialise",
              until: at(1020),
            },
            {
              activity: "Walk",
              place: "park",
              kind: "wander",
              until: { type: "place", place: "park" },
            },
            { activity: "Rest", place: "home", kind: "sleep", until: at(1320) },
          ].filter(
            (o) =>
              o.until.type !== "time" || (o.until as { at: number }).at > clock,
          ),
        },
      };
    }
    return {
      plan: {
        plan: [
          { start: day0 + 420, minutes: 120, activity: "Work", place: "forge" },
          { start: day0 + 600, minutes: 60, activity: "Lunch", place: "cafe" },
          {
            start: day0 + 720,
            minutes: 180,
            activity: "Chat",
            place: "square",
          },
          { start: day0 + 960, minutes: 120, activity: "Walk", place: "park" },
          { start: day0 + 1140, minutes: 180, activity: "Rest", place: "home" },
        ],
      },
    };
  };
}

const flush = () => new Promise((r) => setImmediate(r));

/** One simulated game day (or `days`) of `villagers` NPCs, ticked once per real second like the player. */
export async function runBudget(opts: {
  villagers?: number;
  life?: Record<string, unknown>;
  perMinute?: number;
  days?: number;
}): Promise<BudgetResult> {
  const S: any = await import(`${WORLD}world-sim.ts`);
  const N: any = await import(`${WORLD}needs.ts`);
  const A: any = await import(`${WORLD}world-actions.ts`);
  const W: any = await import(`${WORLD}world-state.ts`);
  const L: any = await import(`${WORLD}life-sim.ts`).catch(() => null);
  const spec: any = budgetSpec(opts.villagers ?? 12, opts.life ?? {});
  if (opts.perMinute !== undefined)
    spec.simulation.calls_per_minute = opts.perMinute;
  const sim = S.simOf(spec);
  const cfg = L ? L.lifeOf(spec) : null;
  const counts: Record<string, number> = {};
  const model = fakeModel(counts);
  let now = 0;
  const queue = new S.BackgroundQueue(
    opts.perMinute === Number.POSITIVE_INFINITY ? 1e9 : sim.calls_per_minute,
    () => now,
  );
  let ws: any = W.initialWorldState(spec);
  const setWs = (fn: (s: any) => any) => {
    ws = fn(ws);
  };
  const tiles: Record<string, { x: number; y: number }> = {};
  const target: Record<string, any> = {};
  spec.npcs.forEach((n: any, i: number) => {
    tiles[n.id] = { x: 2 + ((i * 5) % 28), y: 2 + ((i * 7) % 24) };
  });
  const g = {
    npcTile: (id: string) => tiles[id] ?? null,
    npcTiles: () => tiles,
    playerTile: () => ({ x: 14, y: 14 }),
    setNpcPlace: (id: string, a: any) => {
      target[id] = a;
    },
    setNpcStatus: () => {},
    say: () => {},
  };
  const walk = () => {
    for (const [id, t] of Object.entries(tiles)) {
      const a = target[id];
      if (!a) continue;
      const rnd = N.seeded(spec.seed, id, Math.floor(now / 1000));
      const gx = a.x + Math.floor(rnd() * a.w);
      const gy = a.y + Math.floor(rnd() * a.h);
      t.x += Math.max(-2, Math.min(2, gx - t.x));
      t.y += Math.max(-2, Math.min(2, gy - t.y));
    }
  };
  const busy = new Set<string>();
  const offers = N.advertisements(sim.places);
  const areas = A.placeAreas(spec);
  const ticks = Math.round(((opts.days ?? 1) * 1440) / sim.minutes_per_second);
  let allPlannedAt: number | null = null;
  let small = 0;
  let hits = 0;
  let chats = 0;
  const planDay = async (npcId: string, day: number) => {
    const s = ws;
    const here =
      S.placeAt(sim.places, g.npcTile(npcId) ?? { x: -1, y: -1 })?.id ?? "";
    const mind = s.npc[npcId] ?? S.emptyMind();
    if (L && mind.planDay === day && !L.planDue(cfg, mind, s.clock))
      return false;
    const hit = L ? L.cachedPlan(spec, cfg, s, npcId, s.clock, here) : null;
    if (hit) {
      hits++;
      setWs(hit);
      return false;
    }
    const key = L
      ? L.planCacheKey(cfg, spec, npcId, s.npc[npcId], s.clock, here)
      : null;
    const out: any = (
      await model("p", {
        world: spec,
        npc_id: npcId,
        memory: s.npc[npcId]?.memory ?? [],
        clock: s.clock,
        ...(L ? L.planAsk(cfg, s.npc[npcId]) : {}),
      })
    ).plan;
    const blocks = out?.plan ?? [];
    setWs((st: any) => {
      const next = W.withMind(st, npcId, (m: any) =>
        S.remember(
          {
            ...m,
            plan: blocks,
            planDay: day,
            ...(L ? L.lifePlanned(m, out, st.clock) : {}),
          },
          blocks.length
            ? [{ text: `My plan for day ${day + 1}`, importance: 4 }]
            : [],
          st.clock,
          "plan",
        ),
      );
      return L ? L.storePlan(next, cfg, key, out, st.clock) : next;
    });
  };
  const chat = async (a: string, b: string) => {
    const s = ws;
    busy.add(a);
    busy.add(b);
    chats++;
    const c: any = (
      await model("c", {
        world: spec,
        a,
        b,
        clock: s.clock,
        memory_a: s.npc[a]?.memory ?? [],
        memory_b: s.npc[b]?.memory ?? [],
        ...(L ? L.chatAsk(cfg, s, a, b) : {}),
      })
    ).chat;
    setWs((st: any) => {
      const heard = (from: string, notes: any[]) =>
        cfg?.relationships ? L.rumours(st, from, notes) : notes;
      const learn = (other: string, notes: any[]) => (m: any) =>
        S.remember(
          m,
          [...notes, { text: `Chatted with ${other}`, importance: 2 }],
          st.clock,
        );
      const next = W.withMind(
        W.withMind(st, a, learn(b, heard(b, c.a_learns ?? []))),
        b,
        learn(a, heard(a, c.b_learns ?? [])),
      );
      return L ? L.afterChat(cfg, next, a, b, c) : next;
    });
    const release = now + c.lines.length * 2400;
    holds.push([release, a, b]);
  };
  const holds: [number, string, string][] = [];
  for (let t = 0; t < ticks; t++) {
    now = t * 1000;
    for (let i = holds.length - 1; i >= 0; i--)
      if (holds[i][0] <= now) {
        busy.delete(holds[i][1]);
        busy.delete(holds[i][2]);
        holds.splice(i, 1);
      }
    walk();
    const s = ws;
    const clock = s.clock + sim.minutes_per_second;
    setWs((st: any) => ({ ...st, clock: st.clock + sim.minutes_per_second }));
    const day = S.dayOf(clock);
    if (t % 60 === 30) {
      const me = g.playerTile();
      for (const [id, at] of Object.entries(tiles))
        if (Math.abs(at.x - me.x) + Math.abs(at.y - me.y) <= 6)
          setWs((st: any) =>
            W.withMind(st, id, (m: any) =>
              S.remember(
                m,
                [
                  {
                    text: `Pip waved at ${t}`,
                    importance: 3,
                    guess: (cfg?.reflectAt ?? 0) > 0,
                  },
                ],
                st.clock,
              ),
            ),
          );
    }
    const lives: Record<string, any> = {};
    for (const n of spec.npcs) {
      const mind = ws.npc[n.id] ?? S.emptyMind();
      if (L) {
        const cached = L.planDue(cfg, mind, clock)
          ? L.cachedPlan(
              spec,
              cfg,
              ws,
              n.id,
              clock,
              S.placeAt(sim.places, tiles[n.id])?.id ?? "",
            )
          : undefined;
        if (cached) {
          hits++;
          setWs(cached);
        } else if (cached === null)
          queue.push({
            key: `plan:${n.id}`,
            kind: "plan",
            run: () => planDay(n.id, day),
          });
      } else if (mind.planDay !== day)
        queue.push({
          key: `plan:${n.id}`,
          kind: "plan",
          run: () => planDay(n.id, day),
        });
      const life = A.villagerStep({
        g,
        spec,
        sim,
        npcId: n.id,
        mind: ws.npc[n.id] ?? mind,
        clock,
        offers,
        areas,
        held: ws.auto?.held,
      });
      if (life) lives[n.id] = life;
    }
    setWs((st: any) => {
      const npc = { ...st.npc };
      for (const [id, life] of Object.entries(lives))
        if (npc[id]) npc[id] = { ...npc[id], ...life };
      return { ...st, npc };
    });
    if (!queue.busy && !queue.pending && queue.allows()) {
      const pair = S.findEncounter(
        tiles,
        busy,
        ws.npc,
        clock,
        sim.chat_cooldown_min,
      );
      if (pair) {
        const before = ws;
        setWs((st: any) => ({
          ...st,
          npc: S.markChat(st.npc, pair[0], pair[1], st.clock),
        }));
        const talked =
          L &&
          L.smallTalk({
            cfg,
            spec,
            s: before,
            a: pair[0],
            b: pair[1],
            clock,
            say: () => {},
            update: setWs,
          });
        if (talked) small++;
        else
          queue.push({
            key: `chat:${pair[0]}:${pair[1]}`,
            kind: "chat",
            run: () => chat(pair[0], pair[1]),
          });
      }
    }
    if (L)
      for (const job of L.lifeJobs(
        spec,
        cfg,
        ws,
        async (state: any) => (await model("p", state)).plan,
        setWs,
      ))
        queue.push(job);
    queue.pump();
    for (let k = 0; k < 6; k++) await flush();
    if (
      allPlannedAt === null &&
      spec.npcs.every((n: any) => ws.npc[n.id]?.planDay === S.dayOf(480))
    )
      allPlannedAt = t + 1;
  }
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return {
    seconds: ticks,
    calls: counts,
    total,
    perSecond: total / ticks,
    allPlannedAt,
    chats,
    smallTalk: small,
    cacheHits: hits,
  };
}
