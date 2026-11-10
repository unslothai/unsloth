// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { NodeSpec } from "../src/features/flows/types.ts";
import { specs } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { checkConnection, inLoopFrame, wouldCreateCycle, inHandle, outHandle } = await import(
  "../src/features/flows/editor/connection-rules.ts"
);
const { loopView } = await import("../src/features/flows/editor/loop-status.ts");

const spec = (type: string) => specs[type];
const bench: NodeSpec = {
  ...spec("logic.loop"),
  type: "test.bench",
  outputs: [
    { id: "score", type: "number", label: "Score" },
    { id: "failures", type: "json", label: "Failures" },
  ],
};

const round = (n: number, score: number) =>
  ({ round: n, test_score: score, probe_score: score, rows_from_failures: 0, probe_failures: 0 });
/** The Loop card's view of a run: idle with no output unless `over` says otherwise. */
type LoopInput = Parameters<typeof loopView>[0];
const view = (params: Record<string, unknown>, over: Partial<LoopInput> = {}) =>
  loopView({ status: "idle", params, outputs: null, partial: null, partialStep: null, ...over } as LoopInput);
const history = (rounds: unknown[], stopped = "max_rounds") => ({ history: { rounds, stopped } });
const toSeed = (src: string, srcSpec: NodeSpec, loop?: boolean) =>
  checkConnection(src, srcSpec, outHandle("failures"), "gen", spec("recipe.generate"), inHandle("seed"), loop);

test("Loop keeps rounds, target and feedback on the card, the rest under Advanced", () => {
  const simple = spec("logic.loop")
    .params.filter((p: { advanced?: boolean }) => !p.advanced)
    .map((p: { id: string }) => p.id);
  assert.deepEqual(simple, ["rounds", "target", "feedback", "continue_rounds"]);
});

test("a score step's failures into a dataset seed is a feedback wire, and never a cycle", () => {
  assert.deepEqual(
    toSeed("bench", bench, true),
    { ok: true, kind: "feedback", sourcePort: "failures", targetPort: "seed" },
  );
  // Outside a Loop the same wire is ordinary data: a pipeline that makes more rows from failures.
  assert.deepEqual(
    toSeed("bench", spec("train.benchmark")),
    { ok: true, kind: "data", sourcePort: "failures", targetPort: "seed" },
  );
  const edges = [
    { id: "a", source: "gen", source_port: "dataset", target: "train", target_port: "dataset", kind: "data" },
    { id: "b", source: "train", source_port: "holdout", target: "bench", target_port: "dataset", kind: "data" },
    { id: "c", source: "bench", source_port: "failures", target: "gen", target_port: "seed", kind: "feedback" },
  ] as const;
  assert.equal(wouldCreateCycle(edges, "train", "bench"), false);
  // A Loop dropped into an ordinary frame makes it the Loop's frame too.
  const graph = {
    nodes: {
      loop: { type: "logic.loop", group: "f" },
      bench: { type: "train.benchmark", group: "f" },
      other: { type: "train.benchmark", group: "g" },
      free: { type: "train.benchmark" },
    },
    groups: { f: {}, g: {}, l: { kind: "loop" } },
  };
  assert.equal(inLoopFrame(graph, "bench"), true);
  assert.equal(inLoopFrame(graph, "other"), false);
  assert.equal(inLoopFrame(graph, "free"), false);
  assert.equal(inLoopFrame({ ...graph, nodes: { x: { type: "t", group: "l" } } }, "x"), true);
  // Without a score output, failures is just json and cannot go into a dataset.
  const plain = { ...bench, outputs: [bench.outputs[1]] };
  assert.equal(toSeed("x", plain, true).ok, false);
});

test("the Loop card says which round it is on and plots the test score per round", () => {
  const params = { rounds: 3, target: 90, feedback: true };
  assert.equal(view(params).headline, "Ready");
  const step1 = { partialStep: { step: 1, total: 3 } };
  const running = view(params, { status: "running", partial: { history: [round(1, 0)] }, ...step1 });
  assert.equal(running.headline, "Round 2 of 3");
  assert.deepEqual(running.scores, [0]);
  const done = view(params, { status: "done", outputs: history([round(1, 0), round(2, 88), round(3, 81)]) });
  assert.equal(done.headline, "Best 88% in round 2");
  assert.deepEqual(done.scores, [0, 88, 81]);
  assert.match(view({ ...params, feedback: false }).detail, /feedback off/);
});

test("a rerun shows only its own rounds, and an errored test round never shows as best", () => {
  const params = { rounds: 3, target: 90 };
  const old = history([round(1, 50), round(2, 70), round(3, 95)], "target");
  const rerun = view(params, {
    status: "running",
    outputs: old,
    partial: { history: [round(1, 60)] },
    partialStep: { step: 1, total: 3 },
  });
  assert.equal(rerun.headline, "Round 2 of 3");
  assert.deepEqual(rerun.scores, [60]);
  const errored = { ...round(1, 100), test_errored: 1 };
  const best = (rounds: unknown[]) => view(params, { status: "done", outputs: history(rounds) }).headline;
  assert.equal(best([errored, round(2, 50)]), "Best 50% in round 2");
  assert.equal(best([errored]), "No round scored all its test rows");
});
