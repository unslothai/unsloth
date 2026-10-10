import assert from "node:assert/strict";
import test from "node:test";

// The POST only starts a background load; the notice settles from load-progress. Two loading
// announcements per load are deliberate: before the POST, and after the arbiter evicts.
class FakeWindow extends EventTarget {}

const originalWindow = (globalThis as { window?: unknown }).window;
(globalThis as { window?: unknown }).window = new FakeWindow();

const { subscribeModelLifecycle, withBackgroundLoadNotice } = await import(
  "../src/lib/model-lifecycle-events.ts"
);

const TIMING = { pollMs: 1, readTimeoutMs: 25, stallMs: 5000 };

type Seen = { runtime: string; loading: boolean; model: string | null };

function record(): { seen: Seen[]; settled: Promise<void>; stop: () => void } {
  const seen: Seen[] = [];
  let onSettled: () => void = () => {};
  const settled = new Promise<void>((resolve) => {
    onSettled = resolve;
  });
  const stop = subscribeModelLifecycle((detail) => {
    seen.push({
      runtime: detail.runtime,
      loading: detail.loading,
      model: detail.model,
    });
    if (!detail.loading) onSettled();
  });
  return { seen, settled, stop };
}

test.after(() => {
  (globalThis as { window?: unknown }).window = originalWindow;
});

test("the notice outlives the POST and settles when the load reports ready", async () => {
  const { seen, settled, stop } = record();
  const phases: ("downloading" | "ready")[] = [
    "downloading",
    "downloading",
    "ready",
  ];
  let read = 0;

  const result = await withBackgroundLoadNotice(
    "image",
    "unsloth/flux",
    async () => "started",
    async () => {
      const phase = phases[Math.min(read++, phases.length - 1)];
      if (phase !== "ready") assert.equal(seen.length, 2);
      return phase;
    },
    TIMING,
  );

  assert.equal(result, "started");
  assert.deepEqual(seen, [
    { runtime: "image", loading: true, model: "unsloth/flux" },
    { runtime: "image", loading: true, model: "unsloth/flux" },
  ]);

  await settled;
  assert.equal(read, 3);
  assert.deepEqual(seen, [
    { runtime: "image", loading: true, model: "unsloth/flux" },
    { runtime: "image", loading: true, model: "unsloth/flux" },
    { runtime: "image", loading: false, model: "unsloth/flux" },
  ]);
  stop();
});

test("an errored load settles the notice too", async () => {
  const { seen, settled, stop } = record();
  await withBackgroundLoadNotice(
    "video",
    "unsloth/wan",
    async () => null,
    async () => "error",
    TIMING,
  );
  await settled;
  assert.deepEqual(seen, [
    { runtime: "video", loading: true, model: "unsloth/wan" },
    { runtime: "video", loading: true, model: "unsloth/wan" },
    { runtime: "video", loading: false, model: "unsloth/wan" },
  ]);
  stop();
});

test("a load that never started settles at once, not from the poll", async () => {
  const { seen, stop } = record();
  let polled = false;

  await assert.rejects(
    withBackgroundLoadNotice(
      "image",
      "unsloth/flux",
      async () => {
        throw new Error("422 unsupported model kind");
      },
      async () => {
        polled = true;
        return "ready";
      },
      TIMING,
    ),
    /unsupported model kind/,
  );

  assert.deepEqual(seen, [
    { runtime: "image", loading: true, model: "unsloth/flux" },
    { runtime: "image", loading: false, model: "unsloth/flux" },
  ]);
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.equal(polled, false);
  assert.equal(seen.length, 2);
  stop();
});

test("an unreadable progress read does not end a live load", async () => {
  const { seen, settled, stop } = record();
  const answers: (Error | "downloading" | "ready")[] = [
    new Error("backend restarting"),
    "downloading",
    "ready",
  ];
  let read = 0;

  await withBackgroundLoadNotice(
    "image",
    "unsloth/flux",
    async () => null,
    async () => {
      const answer = answers[Math.min(read++, answers.length - 1)];
      assert.equal(seen.length, 2);
      if (answer instanceof Error) throw answer;
      return answer;
    },
    TIMING,
  );

  await settled;
  assert.equal(read, 3);
  assert.deepEqual(seen.at(-1), {
    runtime: "image",
    loading: false,
    model: "unsloth/flux",
  });
  stop();
});

test("a null phase is terminal, since it means the load left nothing behind", async () => {
  const { seen, settled, stop } = record();
  let read = 0;

  await withBackgroundLoadNotice(
    "video",
    "unsloth/wan",
    async () => null,
    async () => {
      read += 1;
      return null;
    },
    TIMING,
  );

  // After an eject or eviction, load-progress reports null for good, which is terminal.
  await settled;
  assert.equal(read, 1);
  assert.deepEqual(seen, [
    { runtime: "video", loading: true, model: "unsloth/wan" },
    { runtime: "video", loading: true, model: "unsloth/wan" },
    { runtime: "video", loading: false, model: "unsloth/wan" },
  ]);
  stop();
});

test("only downloading and finalizing keep the row up", async () => {
  const { seen, settled, stop } = record();
  const phases: ("downloading" | "finalizing" | "ready")[] = [
    "downloading",
    "finalizing",
    "ready",
  ];
  let read = 0;

  await withBackgroundLoadNotice(
    "image",
    "unsloth/flux",
    async () => null,
    async () => {
      const phase = phases[Math.min(read++, phases.length - 1)];
      if (phase !== "ready") assert.equal(seen.length, 2);
      return phase;
    },
    TIMING,
  );

  await settled;
  assert.equal(read, 3);
  assert.equal(seen.length, 3);
  assert.equal(seen[2].loading, false);
  stop();
});

test("a hung read is abandoned, so the deadline still bounds the loop", async () => {
  const { seen, settled, stop } = record();
  let aborts = 0;
  let read = 0;

  await withBackgroundLoadNotice(
    "image",
    "unsloth/flux",
    async () => null,
    (signal) =>
      new Promise<never>((_resolve, reject) => {
        read += 1;
        signal.addEventListener("abort", () => {
          aborts += 1;
          reject(new Error("aborted"));
        });
      }),
    { pollMs: 1, readTimeoutMs: 10, stallMs: 60 },
  );

  await settled;
  assert.ok(read >= 2, `expected repeated reads, got ${read}`);
  assert.equal(aborts, read);
  assert.deepEqual(seen.at(-1), {
    runtime: "image",
    loading: false,
    model: "unsloth/flux",
  });
  stop();
});

test("the read signal is not aborted when the read answers in time", async () => {
  const { settled, stop } = record();
  let aborted = false;

  await withBackgroundLoadNotice(
    "video",
    "unsloth/wan",
    async () => null,
    async (signal) => {
      signal.addEventListener("abort", () => {
        aborted = true;
      });
      return "ready";
    },
    TIMING,
  );

  await settled;
  await new Promise((resolve) => setTimeout(resolve, 60));
  assert.equal(aborted, false);
  stop();
});

test("a long but healthy download is never abandoned", async () => {
  const { seen, settled, stop } = record();
  let read = 0;

  await withBackgroundLoadNotice(
    "video",
    "unsloth/wan",
    async () => null,
    async () => {
      read += 1;
      return read < 12 ? "downloading" : "ready";
    },
    { pollMs: 1, readTimeoutMs: 25, stallMs: 4 },
  );

  await settled;
  assert.equal(read, 12);
  assert.deepEqual(seen, [
    { runtime: "video", loading: true, model: "unsloth/wan" },
    { runtime: "video", loading: true, model: "unsloth/wan" },
    { runtime: "video", loading: false, model: "unsloth/wan" },
  ]);
  stop();
});

const STALL_MS = 400;

/** One poll run; healthy for `healthyAfterMs`, then unreadable. Timed in ms, not read counts. */
async function settlingRun(healthyAfterMs: number): Promise<{
  totalMs: number;
  resetAtMs: number | null;
  maxGapMs: number;
}> {
  const { settled, stop } = record();
  const began = Date.now();
  let reported = false;
  let resetAtMs: number | null = null;
  let maxGapMs = 0;
  let lastRead = began;
  await withBackgroundLoadNotice(
    "image",
    "unsloth/flux",
    async () => null,
    async () => {
      const now = Date.now();
      maxGapMs = Math.max(maxGapMs, now - lastRead);
      lastRead = now;
      if (!reported && now - began >= healthyAfterMs) {
        reported = true;
        resetAtMs = now - began;
        return "downloading";
      }
      throw new Error("backend restarting");
    },
    { pollMs: 1, readTimeoutMs: 5_000, stallMs: STALL_MS },
  );
  await settled;
  stop();
  return { totalMs: Date.now() - began, resetAtMs, maxGapMs };
}

test("a healthy read resets the stall window", async (t) => {
  // One run, not two subtracted: scheduling noise then only grows totalMs, so the bound is one-sided.
  const run = await settlingRun(STALL_MS / 2);
  assert.ok(run.resetAtMs !== null, "the fixture never got to report progress");

  // The deadline is checked at poll boundaries, so subtract the observed gap; a huge gap voids the run.
  if (run.maxGapMs > STALL_MS / 8) {
    t.skip(
      `runner stalled ${run.maxGapMs}ms mid-loop, which is too close to the ` +
        `${STALL_MS / 2}ms signal to read: total ${run.totalMs}ms, reset at ` +
        `${run.resetAtMs}ms`,
    );
    return;
  }

  assert.ok(
    run.totalMs >= run.resetAtMs + STALL_MS - run.maxGapMs,
    `a healthy read did not restart the stall window: the loop ended ` +
      `${run.totalMs}ms in, having reported progress at ${run.resetAtMs}ms, so ` +
      `it should have run to at least ${run.resetAtMs + STALL_MS}ms. A run of ` +
      "unreadable polls is inheriting the elapsed time of the run before it, so " +
      "a slow download that keeps reporting progress can still be abandoned.",
  );
});

test("the load is announced again once the POST has committed", async () => {
  const { seen, stop } = record();
  let announcedBeforeStart = 0;

  await withBackgroundLoadNotice(
    "image",
    "unsloth/flux",
    async () => {
      announcedBeforeStart = seen.length;
      return null;
    },
    async () => "ready",
    TIMING,
  );

  assert.equal(announcedBeforeStart, 1, "announced optimistically first");
  assert.equal(seen.length, 2, "and again once the backend has taken the GPU");
  assert.deepEqual(
    seen.map((s) => s.loading),
    [true, true],
  );
  stop();
});
