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

/** The real cadence is 2s; these drive the same loop without the waiting. */
const POLL_MS = 1;

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
    POLL_MS,
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
    POLL_MS,
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
      POLL_MS,
    ),
    /unsupported model kind/,
  );

  assert.deepEqual(seen, [
    { runtime: "image", loading: true, model: "unsloth/flux" },
    { runtime: "image", loading: false, model: "unsloth/flux" },
  ]);
  // Exactly one settle, and no poll: the two paths must not both fire.
  await new Promise((resolve) => setTimeout(resolve, POLL_MS * 20));
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
    POLL_MS,
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
    POLL_MS,
  );

  await settled;
  assert.equal(read, 3);
  assert.equal(seen.length, 2);
  assert.equal(seen[1].loading, false);
  stop();
});
