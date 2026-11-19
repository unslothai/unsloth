// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/export-store-resolver.mjs", import.meta.url);

const { useExportRuntimeStore } = await import(
  "../src/features/export/stores/export-runtime-store.ts"
);

const CAP = 4000;
const store = () => useExportRuntimeStore.getState();
type Entry = {
  stream: "stdout" | "stderr" | "status";
  line: string;
  ts: number | null;
};
const line = (i: number): Entry => ({
  stream: "stdout",
  line: `tensor ${i}`,
  ts: null,
});
// The row key the log box renders for each visible line.
const keyOf = (i: number) => store().logBase + i;

test("row keys stay attached to the same line when the head is trimmed", () => {
  store().reset();
  for (let i = 0; i < CAP; i++) store().appendLog(line(i), i + 1);
  assert.equal(store().logLines.length, CAP);
  assert.equal(store().logBase, 0);

  const before = new Map(store().logLines.map((e, i) => [e, keyOf(i)]));
  store().appendLog(line(CAP), CAP + 1);
  store().appendLogs(
    [CAP + 1, CAP + 2].map((n) => ({ ...line(n), seq: n + 1 })),
  );
  store().appendLogBatch(
    [CAP + 3, CAP + 4].map((n) => ({ entry: line(n), seq: n + 1 })),
  );

  assert.equal(store().logLines.length, CAP);
  assert.equal(store().logBase, 5, "five lines were trimmed off the head");
  assert.equal(store().logLines[0].line, "tensor 5");
  assert.equal(store().logLines[CAP - 1].line, `tensor ${CAP + 4}`);
  let kept = 0;
  store().logLines.forEach((e, i) => {
    if (!before.has(e)) return;
    kept += 1;
    assert.equal(keyOf(i), before.get(e), `key moved for ${e.line}`);
  });
  assert.equal(kept, CAP - 5, "surviving lines keep their entry objects");
  assert.equal(new Set(store().logLines.map((_, i) => keyOf(i))).size, CAP);
});

test("a new run and reset start the keys from zero", () => {
  store().reset();
  for (let i = 0; i < CAP + 10; i++) store().appendLog(line(i));
  assert.equal(store().logBase, 10);
  store().reset();
  assert.equal(store().logBase, 0);
  assert.deepEqual(store().logLines, []);
});

test("seq dedupe: a stream batch matches one appendLog per event", () => {
  const events: { entry: Entry; seq?: number }[] = [
    { entry: line(1), seq: 1 },
    { entry: line(2), seq: 2 },
    { entry: line(2), seq: 2 },
    { entry: line(99), seq: 1 },
    { entry: line(3) },
    { entry: { stream: "status", line: "Quantizing", ts: null }, seq: 3 },
    { entry: line(4), seq: 5 },
    { entry: line(5), seq: 4 },
    { entry: line(6) },
  ];
  const snapshot = () => ({
    lines: store().logLines.map((e) => `${e.stream}:${e.line}`),
    lastSeq: store().lastSeq,
    stage: store().stage,
    logBase: store().logBase,
  });

  store().reset();
  for (const { entry, seq } of events) store().appendLog(entry, seq);
  const oneByOne = snapshot();

  store().reset();
  let notified = 0;
  const unsubscribe = useExportRuntimeStore.subscribe(() => {
    notified += 1;
  });
  try {
    store().appendLogBatch(events);
  } finally {
    unsubscribe();
  }

  assert.deepEqual(snapshot(), oneByOne);
  assert.deepEqual(oneByOne.lines, [
    "stdout:tensor 1",
    "stdout:tensor 2",
    "stdout:tensor 3",
    "status:Quantizing",
    "stdout:tensor 4",
    "stdout:tensor 6",
  ]);
  assert.equal(oneByOne.lastSeq, 5);
  assert.equal(oneByOne.stage, "Quantizing");
  assert.equal(notified, 1, "the whole batch is one store update");
});

test("seq dedupe: an all-duplicate append leaves the state object untouched", () => {
  store().reset();
  store().appendLog(line(1), 7);
  const state = store();
  store().appendLog(line(2), 7);
  store().appendLogBatch([{ entry: line(3), seq: 6 }]);
  store().appendLogs([{ ...line(4), seq: 7 }]);
  store().appendLogBatch([]);
  assert.equal(store(), state);
});

test("seq dedupe: the poll path still drops entries at or below lastSeq", () => {
  store().reset();
  store().appendLog(line(1), 2);
  store().appendLogs([
    { ...line(10), seq: 1 },
    { ...line(11), seq: 2 },
    { ...line(12), seq: 3 },
  ]);
  assert.deepEqual(
    store().logLines.map((e) => e.line),
    ["tensor 1", "tensor 12"],
  );
  assert.equal(store().lastSeq, 3);
});

const { groupExportLogLines } = await import(
  "../src/features/export/lib/log-style.ts"
);

test("log blocks follow absolute line numbers, so a trim only touches the first", () => {
  store().reset();
  for (let i = 0; i < CAP; i++) store().appendLog(line(i), i + 1);
  const before = groupExportLogLines(store().logLines, store().logBase);
  assert.equal(before.length, CAP / 100);

  store().appendLogBatch(
    [CAP, CAP + 1, CAP + 2].map((n) => ({ entry: line(n), seq: n + 1 })),
  );
  const after = groupExportLogLines(store().logLines, store().logBase);

  assert.deepEqual(
    after.flatMap((g) => g.entries),
    store().logLines,
    "blocks cover every visible line once, in order",
  );
  assert.deepEqual(
    after.map((g) => [g.key, g.start, g.entries.length]).slice(0, 2),
    [
      [0, 3, 97],
      [1, 100, 100],
    ],
  );
  assert.deepEqual(after.at(-1)?.key, CAP / 100);
  assert.equal(after.at(-1)?.entries.length, 3);
  for (let k = 1; k < before.length; k++) {
    const a = before[k];
    const b = after.find((g) => g.key === a.key);
    assert.ok(b, `block ${a.key} still rendered`);
    assert.equal(b.start, a.start);
    assert.deepEqual(b.entries, a.entries);
    assert.equal(b.entries[0], a.entries[0], "same entry objects, so the block memo holds");
  }
  for (const g of after) {
    assert.ok(g.entries.length <= 100);
    assert.equal(Math.floor(g.start / 100), g.key);
  }
});
