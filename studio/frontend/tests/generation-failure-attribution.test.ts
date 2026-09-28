// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Which attempt a retained generation failure belongs to. */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  GENERATE_FAILURE_LOGGED_MESSAGES,
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
  // An older backend, or a run started by a request that carried no id.
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
  // The 500 paths log and answer with a classified reason, which always opens with this prefix.
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
  assert.match(
    page,
    /: generationFailureWasLogged\(msg\)\s*\)\s*\?\s*viewLogsAction\("server"\)\s*:\s*undefined/,
    "the generate error toast still offers logs for a failure that was never logged",
  );
});

test("a retained failure says whether it was logged, since its text cannot", () => {
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

  const page = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(page, /errorLogged: reportedWasLogged/);
  assert.match(page, /reportedWasLogged = retainedFailureWasLogged\(p\)/);
  assert.match(
    page,
    /const logged = \(err as \{ errorLogged\?: boolean \}\)\.errorLogged;[\s\S]{0,120}typeof logged === "boolean"/,
  );
});

test("a video failure the backend did not log offers no logs action", () => {
  const page = readFileSync(
    new URL("../src/features/video/video-page.tsx", import.meta.url),
    "utf8",
  );
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
  const sets = runtime.match(/loadRequestIssued = true;/g);
  assert.equal(sets?.length, 1, "the request-issued flag is not set where the load is sent");
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

test("a failure the user has already seen is not replayed on the next mount", () => {
  const page = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(
    page,
    /^let surfacedGenerateFailure: string \| null = null;$/m,
    "the already-surfaced failure is not remembered across a remount",
  );
  assert.match(
    page,
    /if \(key === surfacedGenerateFailure\) return;/,
    "a retained failure is replayed once per navigation back to Images",
  );
  assert.match(
    page,
    /return attemptId \? `attempt:\$\{attemptId\}` : `reason:\$\{reason\}`;/,
    "an unattributed failure has no key, so it is replayed forever",
  );
  assert.match(
    page,
    /markGenerateFailureSurfaced\(generateFailureKey\(postedAttemptId, msg\)\);/,
    "the run that reported its own failure lets the next mount report it again",
  );
  const marks = page.match(/markGenerateFailureSurfaced\(/g);
  assert.equal(
    marks?.length,
    3,
    "the surfaced-failure slot is written from somewhere unaccounted for",
  );
});

test("a synchronous video refusal the backend logged offers its log", () => {
  const page = readFileSync(
    new URL("../src/features/video/video-page.tsx", import.meta.url),
    "utf8",
  );
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
  const viaCallback = runtime.match(
    /onRequestStart: \(\) => \{\s*loadRequestIssued = true;\s*\},/g,
  );
  assert.equal(viaCallback?.length, 1, "the flag no longer rides the send boundary");
  assert.ok(
    !/loadRequestIssued = true;\n\s*const (loadResponse|rollbackResponse)/.test(runtime),
    "the flag is still set before the call rather than at the send",
  );
});

test("a logged persistence failure also gets its log", () => {
  assert.equal(
    generationFailureWasLogged("Failed to save the generated image."),
    true,
    "a logged persistence failure has no way back to the log that explains it",
  );
  assert.ok(
    GENERATE_FAILURE_LOGGED_MESSAGES.includes("Failed to save the generated image."),
  );
  // Still not a blanket yes for anything unclassified.
  assert.equal(generationFailureWasLogged("prompt must not be empty"), false);
  assert.equal(
    generationFailureWasLogged("The image generation request did not reach the server."),
    false,
  );
});

test("a rollback request does not mark the failed load as sent", () => {
  const runtime = readFileSync(
    new URL("../src/features/chat/hooks/use-chat-model-runtime.ts", import.meta.url),
    "utf8",
  );
  // One setter, on the TARGET load only.
  const setters = runtime.match(
    /onRequestStart: \(\) => \{\s*loadRequestIssued = true;\s*\},/g,
  );
  assert.equal(
    setters?.length,
    1,
    "the rollback still marks the target request as having been sent",
  );
  const rollbackAt = runtime.indexOf("const rollbackResponse = await loadModel({");
  assert.ok(rollbackAt > 0, "the rollback load moved");
  assert.ok(
    !runtime.slice(rollbackAt, rollbackAt + 4000).includes("loadRequestIssued"),
    "the rollback call still touches the flag",
  );
});
