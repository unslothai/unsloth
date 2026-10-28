// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  GRAMMAR_SAMPLES,
  PREWARM_GUARD_MS,
  PREWARM_RETRY_MS,
  createGrammarPrewarm,
} from "../src/components/assistant-ui/grammar-prewarm.ts";

const harness = (options: { quiet?: () => boolean; skip?: (language: string) => boolean; sync?: boolean } = {}) => {
  const idle: (() => void)[] = [];
  const waits: { callback: () => void; ms: number }[] = [];
  const calls: { code: string; language: string }[] = [];
  const lates: (() => void)[] = [];
  const prewarm = createGrammarPrewarm(
    (code, language, late) => {
      calls.push({ code, language });
      if (options.sync === false) {
        lates.push(late);
        return null;
      }
      return {};
    },
    {
      idle: (callback) => {
        idle.push(callback);
        return () => {
          const at = idle.indexOf(callback);
          if (at >= 0) idle.splice(at, 1);
        };
      },
      wait: (callback, ms) => {
        waits.push({ callback, ms });
        return () => {
          const at = waits.findIndex((entry) => entry.callback === callback);
          if (at >= 0) waits.splice(at, 1);
        };
      },
      quiet: options.quiet ?? (() => true),
      skip: options.skip ?? (() => false),
    },
    [
      { language: "python", lines: ["# a", "import os"] },
      { language: "css", lines: ["/* b */", ".x { }", ".y { }"] },
    ],
  );
  return { prewarm, idle, waits, calls, lates };
};

test("every sample starts with its marker line and names a grammar the plugin keys by", () => {
  for (const sample of GRAMMAR_SAMPLES) {
    assert.ok(sample.lines[0].includes("unsloth grammar warm"), sample.language);
    assert.equal(sample.language, sample.language.toLowerCase());
    assert.ok(sample.lines.length >= 2);
  }
});

test("one step per idle task, each tokenizing one more line, sample by sample", () => {
  const h = harness();
  h.prewarm.start();
  assert.equal(h.idle.length, 1);
  h.idle.shift()!();
  assert.deepEqual(h.calls.map((call) => call.code), ["# a\n"]);
  h.idle.shift()!();
  h.idle.shift()!();
  assert.deepEqual(h.calls.map((call) => [call.language, call.code]), [
    ["python", "# a\n"],
    ["python", "# a\nimport os\n"],
    ["css", "/* b */\n"],
  ]);
  h.idle.shift()!();
  h.idle.shift()!();
  assert.equal(h.calls.length, 5);
  h.idle.shift()!();
  assert.equal(h.idle.length, 0, "the chain ends after the last line");
  h.prewarm.start();
  assert.equal(h.idle.length, 0, "and never restarts");
});

test("a stream or recent input defers the step without tokenizing", () => {
  let quiet = false;
  const h = harness({ quiet: () => quiet });
  h.prewarm.start();
  h.idle.shift()!();
  assert.equal(h.calls.length, 0);
  assert.equal(h.waits.length, 1);
  assert.equal(h.waits[0].ms, PREWARM_RETRY_MS);
  quiet = true;
  h.waits.shift()!.callback();
  assert.equal(h.calls.length, 1);
});

test("a grammar still loading advances when its late callback fires, or after the guard", () => {
  const h = harness({ sync: false });
  h.prewarm.start();
  h.idle.shift()!();
  assert.equal(h.calls.length, 1);
  assert.equal(h.idle.length, 0);
  assert.equal(h.waits.length, 1);
  assert.equal(h.waits[0].ms, PREWARM_GUARD_MS);
  h.lates.shift()!();
  assert.equal(h.idle.length, 1, "the late result schedules the next step");
  assert.equal(h.waits.length, 0, "and cancels the guard");
  h.idle.shift()!();
  h.waits.shift()!.callback();
  assert.equal(h.idle.length, 1, "the guard alone also advances");
});

test("cancel during a grammar load stops the chain, and a restart runs exactly one chain", () => {
  const h = harness({ sync: false });
  h.prewarm.start();
  h.idle.shift()!();
  assert.equal(h.calls.length, 1);
  assert.equal(h.waits.length, 1, "the guard is armed while the grammar loads");
  h.prewarm.cancel();
  assert.equal(h.waits.length, 0, "cancel disarms the guard");
  h.lates.shift()!();
  assert.equal(h.idle.length, 0, "a late result for the cancelled step does not resume the chain");
  h.prewarm.start();
  assert.equal(h.idle.length, 1, "start schedules one chain");
});

test("languages the main thread already warmed are skipped and cancel stops the chain", () => {
  const h = harness({ skip: (language) => language === "python" });
  h.prewarm.start();
  h.idle.shift()!();
  assert.deepEqual(h.calls.map((call) => call.language), ["css"]);
  h.prewarm.cancel();
  assert.equal(h.idle.length, 0);
});
