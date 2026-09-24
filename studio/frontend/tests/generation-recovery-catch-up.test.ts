// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A follower attaches to a run that is already under way, so most of what it reads was written
// before the tab that reads it existed. Those frames are the reply AS IT STANDS, not a stream to
// watch -- but publishing every one of them re-typed the whole answer in front of the reader, one
// awaited storage write per frame, so reopening a closed browser replayed a generation it had
// missed instead of opening on where that generation now is. These pin the gate that separates the
// two: history folds silently and shows once, caught up; only frames past the live edge stream.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { createRecoveryCatchUpGate, generationIsSettled } = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);
const { createRecoveryReplay } = await import(
  "../src/features/chat/utils/chat-generation-replay.ts"
);

const read = (relative: string) =>
  readFileSync(new URL(relative, import.meta.url), "utf8");

/** One frame of a run, as the follower's loop sees it: how far IT has folded, and how far the RUN
 *  had got when the tab attached. */
const update = (cursor: number, lastEventSeq: number, status = "running") => ({
  cursor,
  status: status as never,
  lastEventSeq,
});

/** The whole follower loop: fold each frame, ask the gate whether it is worth showing. */
function follow(
  frames: string[],
  { watermark, status = "running" }: { watermark: number; status?: string },
) {
  const replay = createRecoveryReplay("");
  const gate = createRecoveryCatchUpGate();
  const published: string[] = [];
  let cursor = 0;
  for (const frame of frames) {
    const changed = replay.applyChunk({
      choices: [{ delta: { content: frame } }],
    });
    cursor += 1;
    // The snapshot the loop yields first arms the watermark with the run's own edge at attach time.
    if (cursor === 1) gate.shouldPublish(update(0, watermark), false);
    if (gate.shouldPublish(update(cursor, watermark, status), changed)) {
      published.push(replay.rawText());
    }
  }
  return { published, final: replay.rawText() };
}

test("what was written before this tab existed is shown once, already finished", () => {
  // The report: close the browser mid-reply, reopen it, and instead of the reply where it now is,
  // watch the whole thing type itself out again.
  const missed = Array.from({ length: 40 }, (_, i) => `w${i} `);
  const { published, final } = follow(missed, { watermark: 40 });

  assert.equal(published.length, 1, "the backlog is one write, not forty");
  assert.equal(
    published[0],
    final,
    "and it carries the WHOLE reply as it stands, not a prefix",
  );
});

test("a run that settled while nobody was watching shows its history in one publish", () => {
  // Nothing lives above the watermark here, so `settled` has to override the fold: the whole
  // history IS what the reader must see, and they should see it at once.
  const missed = ["All ", "done. "];
  const { published, final } = follow(missed, {
    watermark: 2,
    status: "completed",
  });
  assert.equal(published.length, 1);
  assert.equal(published[0], final);
  assert.ok(generationIsSettled("completed", 2, 2));
});

test("past the live edge the reply streams again, one frame at a time", () => {
  // What the reader watches from here on is what is being produced NOW: four frames they missed
  // fold into the first caught-up publish, and the four that arrive after it show as they land.
  const frames = ["a1 ", "a2 ", "a3 ", "a4 ", "b1 ", "b2 ", "b3 ", "b4 "];
  const { published } = follow(frames, { watermark: 4 });
  assert.equal(
    published.length,
    5,
    "one for the backlog, then one per live frame",
  );
  assert.ok(
    published[0].endsWith("a4 "),
    "the catch-up publish carries what was missed",
  );
  assert.equal(
    published[1],
    "a1 a2 a3 a4 b1 ",
    "and the live tail grows from there",
  );
});

test("an update that changes nothing is still not worth a write", () => {
  // Past the watermark the gate is exactly what it was before: changed, settled, or status moved.
  const gate = createRecoveryCatchUpGate();
  gate.shouldPublish(update(10, 10), false);
  assert.equal(gate.shouldPublish(update(11, 12, "running"), false), false);
  assert.equal(
    gate.shouldPublish(update(12, 12, "cancelling"), false),
    true,
    "a run that moved states is worth showing even when no token arrived",
  );
});

test("the live edge is where this tab attached, not wherever the next snapshot says", () => {
  // A reconnect hands the loop a fresher `lastEventSeq`. Re-arming on it would push the live edge
  // past frames that are genuinely live and silently swallow them.
  const gate = createRecoveryCatchUpGate();
  gate.shouldPublish(update(0, 20), false); // the snapshot every follow starts with arms the gate
  assert.equal(
    gate.shouldPublish(update(19, 20), true),
    false,
    "still behind the edge it attached at: fold, do not replay",
  );
  assert.equal(
    gate.shouldPublish(update(20, 20), true),
    true,
    "the frame AT the edge is where showing starts, carrying everything folded behind it",
  );
  assert.equal(
    gate.shouldPublish(update(21, 25, "running"), true),
    true,
    "a fresher snapshot must not drag the live edge backwards over live frames",
  );
});

test("the follower routes its publishes through the gate", () => {
  // The rule lives in one place and is applied at exactly one call site per publish path; pinning
  // the wiring is what stops a future edit from restoring the per-frame write.
  const provider = read("../src/features/chat/runtime-provider.tsx");
  const gate = provider.indexOf("const catchUp = createRecoveryCatchUpGate();");
  const loop = provider.indexOf(
    "for await (const update of followChatGenerationRun(",
  );
  assert.ok(gate > 0, "the follower no longer builds a catch-up gate");
  assert.ok(
    loop > gate,
    "the gate must exist before the follow loop consults it",
  );
  assert.equal(
    provider.match(/catchUp\.shouldPublish\(/g)?.length,
    2,
    "both publish paths (the reasoning-summary early continue and the frame path) must ask the gate",
  );
  assert.ok(
    !provider.includes("lastPublishedStatus"),
    "the per-publish status comparison the gate replaced is back",
  );
});
