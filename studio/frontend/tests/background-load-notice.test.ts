import assert from "node:assert/strict";
import test from "node:test";

// The images and video loads only START the work: the POST returns as soon as
// the background thread is running. These cover that the "loading" notice
// outlives the POST and settles from load-progress instead, which is what keeps
// the indicator row and the page toast on screen for the same span.
//
// model-lifecycle-events dispatches on `window`, so stand one up as an
// EventTarget and import the module after it exists.
class FakeWindow extends EventTarget {}

const originalWindow = (globalThis as { window?: unknown }).window;
(globalThis as { window?: unknown }).window = new FakeWindow();

const { subscribeModelLifecycle, withBackgroundLoadNotice } = await import(
  "../src/lib/model-lifecycle-events.ts"
);

/** The real cadences are 2s / 10s; these drive the same loop without waiting. */
const TIMING = { pollMs: 1, readTimeoutMs: 25, deadlineMs: 5000 };

type Seen = { runtime: string; loading: boolean; model: string | null };

/**
 * Record every announcement, and expose a promise that resolves on the settle.
 * Waiting on the event rather than on a sleep keeps these deterministic however
 * slowly the runner schedules the poll.
 */
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
      // Every non-terminal read must leave the row loading: the whole point is
      // that the notice spans the background load, not just the POST.
      if (phase !== "ready") assert.equal(seen.length, 1);
      return phase;
    },
    TIMING,
  );

  // The POST has resolved, and the row must still say loading.
  assert.equal(result, "started");
  assert.deepEqual(seen, [
    { runtime: "image", loading: true, model: "unsloth/flux" },
  ]);

  await settled;
  assert.equal(read, 3);
  assert.deepEqual(seen, [
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
  // Exactly one settle, and no poll: the two paths must not both fire.
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
      // A failed read is not proof the load ended, so the row is still up.
      assert.equal(seen.length, 1);
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

test("a null phase is not terminal, so the row survives the first poll", async () => {
  const { seen, settled, stop } = record();
  const phases: (null | "ready")[] = [null, null, "ready"];
  let read = 0;

  await withBackgroundLoadNotice(
    "video",
    "unsloth/wan",
    async () => null,
    async () => {
      const phase = phases[Math.min(read++, phases.length - 1)];
      if (phase === null) assert.equal(seen.length, 1);
      return phase;
    },
    TIMING,
  );

  await settled;
  assert.equal(read, 3);
  assert.equal(seen.length, 2);
  assert.equal(seen[1].loading, false);
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
    // Accepts the connection and never answers, which is what parks the loop
    // and defeats the deadline unless each read is bounded on its own.
    (signal) =>
      new Promise<never>((_resolve, reject) => {
        read += 1;
        signal.addEventListener("abort", () => {
          aborts += 1;
          reject(new Error("aborted"));
        });
      }),
    { pollMs: 1, readTimeoutMs: 10, deadlineMs: 60 },
  );

  await settled;
  // Several reads were started and every one was cut loose, and the notice
  // settled at the deadline rather than never.
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
  // The per-read timer is cleared on the way out, so a healthy read leaves no
  // abort behind for a later turn of the loop to trip over.
  await new Promise((resolve) => setTimeout(resolve, 60));
  assert.equal(aborted, false);
  stop();
});
