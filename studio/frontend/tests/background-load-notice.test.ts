import assert from "node:assert/strict";
import test from "node:test";

// The images and video loads only START the work: the POST returns as soon as
// the background thread is running. These cover that the "loading" notice
// outlives the POST and settles from load-progress instead, which is what keeps
// the indicator row and the page toast on screen for the same span.
//
// Two `loading: true` announcements per load, deliberately: one before the POST
// so the row appears with the toast, and one after it returns, which is the
// instant the GPU arbiter has committed its eviction. Listeners that re-read
// another runtime need that second edge, so these assert both.
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
const TIMING = { pollMs: 1, readTimeoutMs: 25, stallMs: 5000 };

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
      if (phase !== "ready") assert.equal(seen.length, 2);
      return phase;
    },
    TIMING,
  );

  // The POST has resolved, and the row must still say loading: the second
  // announcement is the post-commit one, not a settle.
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

  // An eject or an eviction cancels the background worker, and load-progress
  // then reports null for good: nothing loading and nothing loaded. Treating it
  // as non-terminal left a "Loading" row with no eject on it for an hour.
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
    { pollMs: 1, readTimeoutMs: 10, stallMs: 60 },
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

test("a long but healthy download is never abandoned", async () => {
  const { seen, settled, stop } = record();
  let read = 0;

  await withBackgroundLoadNotice(
    "video",
    "unsloth/wan",
    async () => null,
    async () => {
      read += 1;
      // Far more polls than the stall window would allow if it were timed from
      // the start of the load: a 100 GB checkpoint on a slow link is hours.
      return read < 12 ? "downloading" : "ready";
    },
    // A stall window shorter than the run of healthy polls it must survive.
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

/**
 * Poll until the notice settles, with the read at `healthyAt` reporting progress
 * and every other read unreadable. `healthyAt: 0` means none of them do.
 * Returns how many reads the loop survived.
 */
async function readsBeforeSettling(healthyAt: number): Promise<number> {
  const { settled, stop } = record();
  let read = 0;
  await withBackgroundLoadNotice(
    "image",
    "unsloth/flux",
    async () => null,
    async () => {
      read += 1;
      if (read === healthyAt) return "downloading";
      throw new Error("backend restarting");
    },
    // readTimeoutMs is generous on purpose: the read answers immediately, and a
    // busy runner must not turn a healthy read into an unreadable one.
    { pollMs: 1, readTimeoutMs: 5_000, stallMs: STALL_MS },
  );
  await settled;
  stop();
  return read;
}

test("a healthy read resets the stall window", async () => {
  // The source reads `Date.now()` and arms `setTimeout` itself, so this runs on
  // the wall clock. That rules out asserting a read count against a millisecond
  // budget: one poll costs about 1 ms here and about 15 ms on a Windows runner,
  // whose default timer granularity is coarser than any margin small enough to
  // keep the test quick. The previous version asserted `reads > 4` against a
  // 30 ms window and was both flaky on Windows (2 reads consumed it before the
  // healthy one arrived) and vacuous on Linux, where 30 ms of 1 ms polls clears
  // 4 reads whether or not the window is ever reset.
  //
  // So measure the reset against the same loop without one. The comparison
  // divides the platform out: whatever a poll costs, a window restarted halfway
  // through must carry the loop about half as far again.
  const withoutReset = await readsBeforeSettling(0);
  assert.ok(
    withoutReset > 8,
    `the baseline is too short to compare against: ${withoutReset} reads. ` +
      "Raise STALL_MS.",
  );

  const withReset = await readsBeforeSettling(Math.floor(withoutReset / 2));
  assert.ok(
    withReset >= withoutReset * 1.25,
    "a healthy read did not restart the stall window: " +
      `${withReset} reads with one, ${withoutReset} without. A run of ` +
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
      // The arbiter has not run yet, so a listener re-reading another runtime
      // here would still see the model this load is about to evict.
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
