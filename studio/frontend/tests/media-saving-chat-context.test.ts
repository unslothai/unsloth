// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// While a media load POST is pending the backend may save the chat context first, and the toast
// says so. Every other phase then describes the PREVIOUS model, so only the save is rendered.

import assert from "node:assert/strict";
import test from "node:test";

import * as save from "../src/lib/swap-save-phase.ts";
import { readSrc } from "./helpers/kit.ts";

const tick = () => new Promise((resolve) => setTimeout(resolve, 5));
const FAST = { intervalMs: 1, firstDelayMs: 0 };
const SAVE = save.SAVING_CHAT_CONTEXT;

test("only the saving phase reaches the toast, and a failed read keeps watching", async () => {
  const phases = ["ready", null, "boom", SAVE, "downloading", "error", SAVE];
  let saving = 0;
  let reads = 0;
  const watch = save.watchSavingPhase(
    async () => {
      const phase = phases[reads++] ?? "ready";
      if (phase === "boom") throw new Error("offline");
      return { phase };
    },
    () => saving++,
    FAST,
  );
  while (reads < phases.length) await tick();
  watch.stop();
  assert.equal(saving, 2);
});

test("a read still in flight at stop() is dropped, and stop() is idempotent", async () => {
  let release: (value: { phase: unknown }) => void = () => {};
  let aborted = false;
  let saving = 0;
  let reads = 0;
  const read = (signal: AbortSignal) => {
    reads++;
    signal.addEventListener("abort", () => {
      aborted = true;
    });
    return new Promise<{ phase: unknown }>((resolve) => {
      release = resolve;
    });
  };
  const watch = save.watchSavingPhase(read, () => saving++, FAST);
  while (reads === 0) await tick();
  watch.stop();
  watch.stop();
  release({ phase: SAVE });
  await tick();
  assert.deepEqual(
    { aborted, saving, reads },
    { aborted: true, saving: 0, reads: 1 },
  );
});

for (const [path, read] of [
  ["features/images/images-page.tsx", "getDiffusionLoadProgress"],
  ["features/video/video-page.tsx", "getVideoLoadProgress"],
] as const) {
  test(`${path} shows the save on its own load toast, only while the POST is pending`, () => {
    const source = readSrc(path);
    for (const pin of [
      'if (p.phase === "saving_chat_context") {',
      'title="Saving chat context…"',
      `message="Saving the chat model's context so it can resume after this load."`,
      // The same toast id, keeping Cancel, fenced on a cancel or a newer load.
      "if (startSeq === cancelSeq.current && startLoad === loadSeq.current && loadToastId.current != null) {",
      "toast(null, loadToastArgs(SAVING_PROGRESS, loadToastId.current, cancelLoadFromToast));",
    ]) {
      assert.ok(source.includes(pin), `missing: ${pin}`);
    }
    const load = source.slice(
      source.indexOf("const handleLoad = useCallback("),
    );
    const order = [
      "loadToastId.current = toast(",
      `const savePoll = watchSavingPhase(${read}, () => {`,
      "const answer = await startRequest;",
      "savePoll.stop();",
      "void pollLoadProgress();",
    ].map((pin) => load.indexOf(pin));
    assert.ok(
      order.every((at, i) => at > 0 && (i === 0 || at > order[i - 1])),
      `out of order: ${order}`,
    );
  });
}
