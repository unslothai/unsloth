// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A cached load that is really a download must say so. #9094: the toast read "Loading cached
// model into memory." for fourteen minutes while bytes arrived from Hugging Face.
//
// Asserted on the decision, not on the source text: a regex over the hook would pass with the
// watch wired to nothing.

import assert from "node:assert/strict";
import test from "node:test";

import {
  CACHE_MISS_DOWNLOAD_DESCRIPTION,
  EMPTY_CACHE_MISS_WATCH,
  watchCacheMissDownload,
} from "../src/features/chat/lib/cache-miss-download.ts";
import { readText } from "./helpers/kit.ts";

const partial = (bytes: number) => ({
  downloaded_bytes: bytes,
  expected_bytes: 1_000,
  progress: bytes / 1_000,
  cache_measured: true,
});

test("bytes that grow between two readings are a download", () => {
  const first = watchCacheMissDownload(EMPTY_CACHE_MISS_WATCH, partial(100));
  assert.equal(first.started, false, "one reading cannot show movement");

  const second = watchCacheMissDownload(first.watch, partial(400));
  assert.equal(second.started, true);
  assert.equal(second.percent, 40);
});

test("an incomplete cache that is not moving is a load, not a download", () => {
  // The state #9094's reporter was in before the transfer began, and the state of every repo
  // whose last download was cancelled. Relabelling here would cry download on every load.
  let watch = EMPTY_CACHE_MISS_WATCH;
  for (let poll = 0; poll < 5; poll += 1) {
    const verdict = watchCacheMissDownload(watch, partial(250));
    assert.equal(verdict.started, false);
    watch = verdict.watch;
  }
});

test("a complete cache resets the watch instead of arming it", () => {
  const complete = {
    downloaded_bytes: 1_000,
    expected_bytes: 1_000,
    progress: 1,
    cache_measured: true,
  };
  const armed = watchCacheMissDownload(EMPTY_CACHE_MISS_WATCH, partial(100));
  const reset = watchCacheMissDownload(armed.watch, complete);
  assert.equal(reset.started, false);
  assert.equal(reset.watch.bytes, null);
  // And the next partial reading is a first reading again, so it cannot claim movement.
  assert.equal(watchCacheMissDownload(reset.watch, partial(900)).started, false);
});

test("an unmeasurable cache leaves the comparison alone", () => {
  const armed = watchCacheMissDownload(EMPTY_CACHE_MISS_WATCH, partial(100));
  const unreadable = watchCacheMissDownload(armed.watch, {
    downloaded_bytes: 0,
    expected_bytes: 0,
    progress: 0,
    cache_measured: false,
  });
  assert.equal(unreadable.started, false);
  assert.equal(unreadable.watch.bytes, 100, "the reading was not evidence, so it is not kept");
  assert.equal(watchCacheMissDownload(unreadable.watch, partial(300)).started, true);
});

test("a missing or malformed reading is never a download", () => {
  for (const reading of [null, undefined, {}, { downloaded_bytes: Number.NaN, progress: 0 }]) {
    assert.equal(
      watchCacheMissDownload({ bytes: 1 }, reading as never).started,
      false,
    );
  }
});

test("the phase has words of its own, naming the reason", () => {
  assert.match(CACHE_MISS_DOWNLOAD_DESCRIPTION, /downloading from Hugging Face/);
  assert.doesNotMatch(CACHE_MISS_DOWNLOAD_DESCRIPTION, /cached model into memory/);
});

test("a total the backend could not establish reports bytes rather than a wrong percent", () => {
  const first = watchCacheMissDownload(EMPTY_CACHE_MISS_WATCH, {
    downloaded_bytes: 10,
    expected_bytes: 0,
    progress: 0,
    cache_measured: true,
  });
  const second = watchCacheMissDownload(first.watch, {
    downloaded_bytes: 20,
    expected_bytes: 0,
    progress: 0,
    cache_measured: true,
  });
  assert.equal(second.started, true);
  assert.equal(second.percent, null);
});

test("the load hook actually consults the watch, and on the cached branch", () => {
  // The decision above is only worth anything if the poll loop asks it. A cached load polls the
  // mmap phase and nothing else on main, which is how #9094 stayed invisible for fourteen
  // minutes, so assert the call sits on that branch and that finding a download reopens the
  // download poller rather than only changing the words.
  const hook = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
  assert.match(hook, /watchCacheMissDownload\(/);
  assert.match(hook, /watchForCacheMiss && !cacheMissDownload && \(await cacheMissDownloadStarted\(\)\)/);
  assert.match(hook, /downloadComplete = false;\n\s+activeLoadingDescription = cacheMissDescription;/);
  // Local paths, Ollama manifests and cached LoRAs never reach the Hub, so they are not polled.
  assert.match(
    hook,
    /const watchForCacheMiss =\n\s+isDownloaded && !isLocal && nativePathToken == null && !isOllamaModelId\(modelId\);/,
  );
});

test("an unknown total still updates the toast the user is looking at", () => {
  // The state write in this branch is read only by the inline status shown AFTER the toast
  // is dismissed. With the toast still up, and nothing else in the branch touching it, the
  // words the user sees stay on whatever the previous phase set, which on exactly this path
  // is "Loading cached model into memory" for the length of the download. That is #9094,
  // reappearing on the branch where the backend cannot size the repo.
  const hook = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
  const start = hook.indexOf("prog.expected_bytes === 0 &&");
  assert.ok(start > 0, "the unknown-total branch moved");
  const end = hook.indexOf("} else if (prog.progress >= 1 && hasShownProgress)", start);
  assert.ok(end > start, "the branch after the unknown-total one moved");
  const branch = hook.slice(start, end);

  // Dismissed: the inline status still gets the state write, and only then, so the chat page
  // is not re-rendered on every poll while the toast is the thing being read.
  assert.match(branch, /if \(loadToastDismissedRef\.current\) \{[\s\S]*setLoadProgress\(/);
  // Visible: the toast is written directly, the same way both neighbouring branches do it.
  assert.match(branch, /\} else \{[\s\S]*toast\(null, \{[\s\S]*renderLoadDescription\(/);
  assert.match(branch, /"Downloading model…"/);
  // No percentage: there is no total to compute one from, and inventing one is the failure
  // the byte-count label exists to avoid.
  assert.match(branch, /renderLoadDescription\(\s*"Downloading model…",[\s\S]*?null,/);
});

test("detecting the download counts as having shown progress", () => {
  // The completion branch in pollDownload is gated on hasShownProgress, and the only other
  // places that set it are the two branches that report bytes. A tail that finishes between
  // the watcher request and the pollDownload immediately behind it therefore arrives at a
  // poller that ignores `progress >= 1`: downloadComplete stays false, every later tick
  // re-enters pollDownload rather than pollLoad, and the UI is stuck in the downloading
  // phase with the mmap and startup progress suppressed for the rest of the load.
  const hook = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
  const start = hook.indexOf("const cacheMissDownloadStarted");
  assert.ok(start > 0, "the watcher moved");
  const end = hook.indexOf("const pollProgress", start);
  assert.ok(end > start);
  const branch = hook.slice(start, end);
  assert.match(branch, /if \(!verdict\.started\) return false;/);
  assert.match(branch, /cacheMissDownload = true;[\s\S]*hasShownProgress = true;/);
  // And it is set only once the watch has actually reported movement, so a first reading
  // cannot arm it.
  const armed = branch.indexOf("hasShownProgress = true;");
  assert.ok(armed > branch.indexOf("if (!verdict.started) return false;"));
});

test("the cache-miss watcher re-reads the load after its own await", () => {
  // The request is in flight for as long as the backend takes to answer, and a load that
  // completes or is cancelled in that window runs `finally`: the interval is cleared and
  // resetLoadingUi() puts the UI back. A callback that then writes the watch and
  // setLoadProgress left the dismissed-toast inline status reading "Downloading the rest of
  // the model" for a load that had already ended. pollDownload and pollLoad both re-read
  // after their awaits for this reason; this one did not.
  const hook = readText("../src/features/chat/hooks/use-chat-model-runtime.ts");
  const start = hook.indexOf("const cacheMissDownloadStarted");
  assert.ok(start > 0, "the watcher moved");
  const end = hook.indexOf("const pollProgress", start);
  assert.ok(end > start);
  const branch = hook.slice(start, end);

  const awaited = branch.indexOf("await getDownloadProgress(");
  assert.ok(awaited > 0, "the watcher no longer reads download progress");
  const recheck = branch.indexOf(
    "if (abortCtrl.signal.aborted || !loadingModelRef.current) return false;",
  );
  assert.ok(recheck > awaited, "the load is not re-read after the await");
  // Before anything is mutated: the watch, the flags and the progress write all follow it.
  for (const mutation of [
    "cacheMissWatch = verdict.watch;",
    "cacheMissDownload = true;",
    "hasShownProgress = true;",
    "setLoadProgress({",
  ]) {
    const at = branch.indexOf(mutation);
    assert.ok(at > recheck, `${mutation} runs before the re-read`);
  }
});
