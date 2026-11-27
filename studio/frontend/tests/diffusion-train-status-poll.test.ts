// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

function body(status: string, active: boolean, step = 4000): string {
  return JSON.stringify({
    active,
    job_id: "job-1",
    status,
    step,
    metric_history: { steps: [1, 2, step], loss: [0.3, 0.2, 0.1] },
  });
}

function setup(reply: () => string) {
  const target = new EventTarget();
  const doc = {
    hidden: false,
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    setHidden(hidden: boolean) {
      doc.hidden = hidden;
      target.dispatchEvent(new Event("visibilitychange"));
    },
  };
  const g = globalThis as { document?: unknown; window?: unknown };
  g.document = doc;
  g.window = globalThis;
  const cleanups: (() => void)[] = [];
  const applied: { status: string; step: number }[] = [];
  let requests = 0;
  const mod = loadWithStubs<{
    useDiffusionTrainingStatus: (active: boolean) => unknown;
  }>(
    new URL(
      "../src/features/images/train/use-diffusion-training-status.ts",
      import.meta.url,
    ),
    {
      react: {
        useState: <T>(initial: T) => [
          initial,
          (next: { status: string; step: number }) =>
            applied.push({ status: next.status, step: next.step }),
        ],
        useRef: <T>(initial: T) => ({ current: initial }),
        useCallback: <T>(fn: T) => fn,
        useEffect: (effect: () => (() => void) | undefined) => {
          const cleanup = effect();
          if (cleanup) cleanups.push(cleanup);
        },
      },
      "../api": {
        getDiffusionTrainingStatusText: async () => {
          requests += 1;
          return reply();
        },
      },
    },
  );
  mod.useDiffusionTrainingStatus(true);
  return {
    doc,
    applied,
    requests: () => requests,
    unmount: () => {
      for (const cleanup of cleanups) cleanup();
    },
  };
}

test("an unchanged finished status is applied once while the tab is visible", async (t) => {
  t.mock.timers.enable({ apis: ["setInterval"] });
  const done = body("completed", false);
  const s = setup(() => done);
  try {
    await settle();
    for (let i = 0; i < 40; i++) {
      t.mock.timers.tick(1500);
      await settle();
    }
    assert.equal(s.requests(), 41, "a visible tab keeps the 1.5 s cadence");
    assert.deepEqual(s.applied, [{ status: "completed", step: 4000 }]);
  } finally {
    s.unmount();
  }
});

test("a hidden tab skips polls once settled and polls once on return", async (t) => {
  t.mock.timers.enable({ apis: ["setInterval"] });
  let reply = body("completed", false);
  const s = setup(() => reply);
  try {
    await settle();
    s.doc.setHidden(true);
    t.mock.timers.tick(60_000);
    await settle();
    assert.equal(s.requests(), 1, "no polls while hidden and settled");

    reply = body("running", true, 1);
    s.doc.setHidden(false);
    await settle();
    assert.equal(s.requests(), 2, "returning polls at once");
    assert.equal(s.applied.at(-1)?.status, "running");

    s.doc.setHidden(true);
    s.doc.setHidden(false);
    await settle();
    assert.equal(
      s.requests(),
      2,
      "a switch with no skipped tick sends nothing",
    );
  } finally {
    s.unmount();
  }
});

test("a live run keeps polling while hidden, so completion lands on time", async (t) => {
  t.mock.timers.enable({ apis: ["setInterval"] });
  let tick = 0;
  const s = setup(() =>
    tick < 10 ? body("running", true, tick) : body("completed", false),
  );
  try {
    await settle();
    s.doc.setHidden(true);
    for (tick = 1; tick <= 20; tick++) {
      t.mock.timers.tick(1500);
      await settle();
    }
    assert.equal(s.requests(), 11, "polls every tick until the run settles");
    assert.equal(s.applied.length, 11, "each changed reply is applied");
    assert.equal(s.applied.at(-1)?.status, "completed");
    assert.equal(s.applied.filter((a) => a.status === "completed").length, 1);
  } finally {
    s.unmount();
  }
});
