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

const round = (n: number, score: number) => ({
  round: n,
  test_score: score,
  probe_score: score,
  rows_from_failures: 0,
  probe_failures: 0,
});

test("Loop keeps rounds, target and feedback on the card, the rest under Advanced", () => {
  const simple = spec("logic.loop")
    .params.filter((p: { advanced?: boolean }) => !p.advanced)
    .map((p: { id: string }) => p.id);
  assert.deepEqual(simple, ["rounds", "target", "feedback", "continue_rounds"]);
});

test("a score step's failures into a dataset seed is a feedback wire, and never a cycle", () => {
  const check = checkConnection(
    "bench",
    bench,
    outHandle("failures"),
    "gen",
    spec("recipe.generate"),
    inHandle("seed"),
    true,
  );
  assert.deepEqual(check, {
    ok: true,
    kind: "feedback",
    sourcePort: "failures",
    targetPort: "seed",
  });
  // Outside a Loop the same wire is ordinary data: a pipeline that makes more rows from failures.
  assert.deepEqual(
    checkConnection("bench", spec("train.benchmark"), outHandle("failures"), "gen", spec("recipe.generate"), inHandle("seed")),
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
  assert.equal(
    checkConnection("x", plain, outHandle("failures"), "gen", spec("recipe.generate"), inHandle("seed"), true).ok,
    false,
  );
});

test("the Loop card says which round it is on and plots the test score per round", () => {
  const params = { rounds: 3, target: 90, feedback: true };
  assert.equal(
    loopView({ status: "idle", params, outputs: null, partial: null, partialStep: null }).headline,
    "Ready",
  );
  const running = loopView({
    status: "running",
    params,
    outputs: null,
    partial: { history: [round(1, 0)] },
    partialStep: { step: 1, total: 3 },
  });
  assert.equal(running.headline, "Round 2 of 3");
  assert.deepEqual(running.scores, [0]);
  const done = loopView({
    status: "done",
    params,
    outputs: { history: { rounds: [round(1, 0), round(2, 88), round(3, 81)], stopped: "max_rounds" } },
    partial: null,
    partialStep: null,
  });
  assert.equal(done.headline, "Best 88% in round 2");
  assert.deepEqual(done.scores, [0, 88, 81]);
  assert.match(
    loopView({ status: "idle", params: { ...params, feedback: false }, outputs: null, partial: null, partialStep: null }).detail,
    /feedback off/,
  );
});

test("a rerun shows only its own rounds, and an errored test round never shows as best", () => {
  const params = { rounds: 3, target: 90 };
  const old = { history: { rounds: [round(1, 50), round(2, 70), round(3, 95)], stopped: "target" } };
  const rerun = loopView({
    status: "running",
    params,
    outputs: old,
    partial: { history: [round(1, 60)] },
    partialStep: { step: 1, total: 3 },
  });
  assert.equal(rerun.headline, "Round 2 of 3");
  assert.deepEqual(rerun.scores, [60]);
  const errored = { ...round(1, 100), test_errored: 1 };
  const done = loopView({
    status: "done",
    params,
    outputs: { history: { rounds: [errored, round(2, 50)], stopped: "max_rounds" } },
    partial: null,
    partialStep: null,
  });
  assert.equal(done.headline, "Best 50% in round 2");
  const none = loopView({
    status: "done",
    params,
    outputs: { history: { rounds: [errored], stopped: "max_rounds" } },
    partial: null,
    partialStep: null,
  });
  assert.equal(none.headline, "No round scored all its test rows");
});
