// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Which attempt a retained generation failure belongs to.
 *
 * The reason outlives its run because a caller whose POST was lost has nothing else to
 * read, which is why it cannot be taken at face value: the POST may never have arrived,
 * or the run that failed may be an earlier one or a concurrent client's. Reporting either
 * is a failure that did not happen here, and it skips the gallery probe that says what did.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  generationFailureForAttempt,
  generationFailureWasLogged,
  retainedFailureWasLogged,
  newGenerationAttemptId,
} from "../src/features/images/lib/generation-failure.ts";

const REASON = "Image generation failed. The GPU ran out of memory.";
const MINE = "attempt-mine";

test("a failure carrying this attempt's own id is this attempt's", () => {
  assert.equal(
    generationFailureForAttempt(
      { error: REASON, generation_attempt: MINE },
      MINE,
    ),
    REASON,
  );
});

test("a failure from any other run is not, however recent", () => {
  // The two cases a monotonic counter cannot separate: a previous run of this client, and a
  // run a concurrent client started AFTER this post and failed before this waiter polled.
  // Both are "later than my baseline"; neither is this attempt.
  for (const other of ["attempt-earlier", "attempt-other-tab"]) {
    assert.equal(
      generationFailureForAttempt(
        { error: REASON, generation_attempt: other },
        MINE,
      ),
      null,
      other,
    );
  }
});

test("a reason that cannot be identified is not used", () => {
  // An older backend, or a run started by a request that carried no id. The gallery probe
  // still settles those, as it did before this field existed, rather than a guess here.
  for (const carried of [undefined, null, ""]) {
    assert.equal(
      generationFailureForAttempt(
        {
          error: REASON,
          generation_attempt: carried as string | null | undefined,
        },
        MINE,
      ),
      null,
      String(carried),
    );
  }
  // And an attempt with no id of its own cannot claim a reason either.
  assert.equal(
    generationFailureForAttempt(
      { error: REASON, generation_attempt: MINE },
      null,
    ),
    null,
  );
});

test("no failure is no failure, whatever the id says", () => {
  for (const error of [undefined, null, ""]) {
    assert.equal(
      generationFailureForAttempt(
        { error: error as string | null | undefined, generation_attempt: MINE },
        MINE,
      ),
      null,
      String(error),
    );
  }
});

test("a minted id is unique, and something the backend will accept", () => {
  // Bounded and patterned on the backend, because it comes off a request and goes back out
  // on a response: attempt_id is max_length 64 with ^[A-Za-z0-9_-]+$.
  const ids = new Set<string>();
  for (let i = 0; i < 64; i++) {
    const id = newGenerationAttemptId();
    assert.match(id, /^[A-Za-z0-9_-]+$/, id);
    assert.ok(id.length > 0 && id.length <= 64, `length ${id.length}`);
    ids.add(id);
  }
  assert.equal(
    ids.size,
    64,
    "minted ids collided, so two attempts could share a failure",
  );
});

test("an id is minted per post and sent with it", () => {
  // Source-shape, because images-page.tsx cannot be loaded on its own, and scoped to the
  // generate loop's body: an id minted anywhere else would not describe one post. Asserted
  // as: the mint, the payload field and the settle call all sit inside handleGenerate,
  // between its start and the next top-level callback.
  const src = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  const start = src.indexOf("const handleGenerate = useCallback(async () => {");
  assert.ok(start > 0, "handleGenerate was renamed");
  const end = src.indexOf("const handleGenerateWithRecall", start);
  assert.ok(end > start, "handleGenerateWithRecall was renamed");
  const body = src.slice(start, end);
  assert.ok(
    body.includes("const attemptId = newGenerationAttemptId();"),
    "the attempt id is not minted in the submit path",
  );
  assert.ok(
    body.includes("attempt_id: attemptId,"),
    "the minted id is not sent with the generate request",
  );
  assert.ok(
    body.indexOf("const attemptId") < body.indexOf("attempt_id: attemptId,"),
    "the id is sent before it is minted",
  );
  assert.ok(
    body.includes("attemptId,"),
    "the settling waiter is not given this attempt's id",
  );
});

test("only a failure the server logged offers the logs action", () => {
  // The 500 paths log and answer with a classified reason, which always opens with this
  // prefix. Nothing else is in the log the action would open.
  assert.equal(generationFailureWasLogged("Image generation failed."), true);
  assert.equal(
    generationFailureWasLogged(
      "Image generation failed. The GPU ran out of memory.",
    ),
    true,
  );
  // Never left the browser.
  assert.equal(
    generationFailureWasLogged(
      "The image generation request did not reach the server.",
    ),
    false,
    "offered the logs of an unrelated run for a request that never arrived",
  );
  // Answered at validation, without logging.
  assert.equal(generationFailureWasLogged("prompt must not be empty"), false);
  assert.equal(
    generationFailureWasLogged(
      "Timed out waiting for the image generation to finish.",
    ),
    false,
  );
  assert.equal(
    generationFailureWasLogged("Lost connection to the image server."),
    false,
  );

  // And the page gates the action on it rather than attaching it to every error.
  const page = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  // The prefix judgement is still the fallback for a failure the POST reported directly,
  // where there is no retained record to ask.
  assert.match(
    page,
    /: generationFailureWasLogged\(msg\)\s*\)\s*\?\s*viewLogsAction\("server"\)\s*:\s*undefined/,
    "the generate error toast still offers logs for a failure that was never logged",
  );
});

test("a retained failure says whether it was logged, since its text cannot", () => {
  // The reason a settling caller reads has already been classified, so a client-input
  // failure the route answered WITHOUT logging carries the same prefix as an internal one.
  assert.equal(
    retainedFailureWasLogged({
      error: "Image generation failed.",
      error_logged: false,
    }),
    false,
    "a failure the server never logged was offered as one the log explains",
  );
  assert.equal(
    retainedFailureWasLogged({
      error: "Image generation failed. The GPU ran out of memory.",
      error_logged: true,
    }),
    true,
  );
  // An older backend sends nothing, and the reasons it retains are the logged ones.
  assert.equal(retainedFailureWasLogged({ error: "Image generation failed." }), true);
  assert.equal(
    retainedFailureWasLogged({ error: "Image generation failed.", error_logged: null }),
    true,
  );

  // The page carries the answer out of the settling loop with the error it throws, and
  // prefers it over the prefix guess when it is there.
  const page = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(page, /errorLogged: reportedWasLogged/);
  assert.match(page, /reportedWasLogged = retainedFailureWasLogged\(p\)/);
  assert.match(
    page,
    /typeof \(err as \{ errorLogged\?: boolean \}\)\.errorLogged === "boolean"/,
  );
});

test("a video failure the backend did not log offers no logs action", () => {
  const page = readFileSync(
    new URL("../src/features/video/video-page.tsx", import.meta.url),
    "utf8",
  );
  // Both sites: the live poll and the mount-time resume, which shows the same terminal
  // phase after a reload.
  const gates = page.match(/error_logged === false \? undefined : viewLogsAction\("server"\)/g);
  assert.equal(
    gates?.length,
    2,
    "a video failure the server never logged still offers to open its log",
  );
});

test("a load that never reached the server offers no logs action", () => {
  const runtime = readFileSync(
    new URL("../src/features/chat/hooks/use-chat-model-runtime.ts", import.meta.url),
    "utf8",
  );
  // The flag is set immediately before the request goes out, on both the main load and
  // the rollback, and nowhere else.
  const sets = runtime.match(/loadRequestIssued = true;/g);
  assert.equal(sets?.length, 2, "the request-issued flag is not set where the load is sent");
  assert.match(runtime, /let loadRequestIssued = false;/);
  assert.match(
    runtime,
    /runnerLogPath \|\| loadRequestIssued\s*\?\s*viewLogsAction\(/,
    "a failure before the request was sent still offers the server log",
  );
});

test("a run that failed across a reload is reported, not taken for finished", () => {
  const page = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  // Both resume paths: the poll that finds the run already idle, and the mount probe that
  // never saw it active at all. A reload leaves no POST to reject and no settling loop, so
  // the retained reason is the only channel left.
  const calls = page.match(/reportResumedGenerateFailure\(/g);
  assert.equal(
    calls?.length,
    3,
    "a failure spanning a reload is still silently read as a finished run",
  );
  // And it is gated like every other site, on what the server logged.
  assert.match(
    page,
    /action: retainedFailureWasLogged\(progress\)\s*\?\s*viewLogsAction\("server"\)/,
  );
  // A cancellation is still not an error.
  assert.match(page, /shouldReportGenerateError\(\{ message: reason, stopRequested: false \}\)/);
});

test("a synchronous video refusal the backend logged offers its log", () => {
  const page = readFileSync(
    new URL("../src/features/video/video-page.tsx", import.meta.url),
    "utf8",
  );
  // Polling never starts for a rejected POST, so neither progress branch can attach the
  // action: this is the only place it can be offered. Classified refusals carry the
  // fallback prefix and are logged first; a 400 carries the raw validation text.
  assert.match(
    page,
    /refusal\.startsWith\(VIDEO_FAILURE_LOGGED_PREFIX\)\s*\?\s*viewLogsAction\("server"\)\s*:\s*undefined/,
    "a logged video refusal still has no way back to its log",
  );
  assert.match(page, /VIDEO_FAILURE_LOGGED_PREFIX = "Video generation failed\."/);
});

test("the load-issued flag is set at the send boundary, not before it", () => {
  const runtime = readFileSync(
    new URL("../src/features/chat/hooks/use-chat-model-runtime.ts", import.meta.url),
    "utf8",
  );
  // loadModel does its own token preparation and abort check before sending, so a flag set
  // before the call is still set when that inner prompt is declined and the backend
  // received nothing. onRequestStart fires at the actual send.
  const viaCallback = runtime.match(
    /onRequestStart: \(\) => \{\s*loadRequestIssued = true;\s*\},/g,
  );
  assert.equal(viaCallback?.length, 2, "the flag no longer rides the send boundary");
  assert.ok(
    !/loadRequestIssued = true;\n\s*const (loadResponse|rollbackResponse)/.test(runtime),
    "the flag is still set before the call rather than at the send",
  );
});
