// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pages pass onEject only when loaded, so first loads need a toast Cancel and a keyboard button.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const PAGES = [
  ["Images", "../src/features/images/images-page.tsx"],
  ["Video", "../src/features/video/video-page.tsx"],
] as const;

const DOWNLOAD_PANEL = readText(
  "../src/features/hub/download-manager/download-manager-panel.tsx",
);

for (const [page, path] of PAGES) {
  const SOURCE = readText(path);

  test(`the ${page} load toast offers a Cancel action`, () => {
    assert.match(
      SOURCE,
      /cancel: \{ label: "Cancel", onClick: onCancel \}/,
      "the load toast must carry a cancel action, not just a close button",
    );
    // Every toast site, including progress ticks, must pass the action or the button vanishes.
    // Call sites only: the declaration's argument list opens with a newline.
    const sites = SOURCE.match(/loadToastArgs\((?!\n)[^)\n]*\)/g) ?? [];
    assert.equal(sites.length, 4, "expected the four load-toast call sites");
    for (const site of sites) {
      assert.match(
        site,
        /cancelLoadFromToast/,
        `a load toast built without the cancel action: ${site}`,
      );
    }
  });

  test(`the ${page} page shows a cancel control while a load is in flight`, () => {
    assert.match(
      SOURCE,
      /busy === "loading" && \(\s*<Tooltip>/,
      "the cancel control must be gated on the load being in flight",
    );
    assert.match(SOURCE, /aria-label="Cancel load"/);
    assert.match(SOURCE, /onClick=\{\(\) => void handleCancelLoad\(\)\}/);
  });

  test(`the ${page} cancel control does not wait for a resident model`, () => {
    const control = SOURCE.slice(
      SOURCE.indexOf('aria-label="Cancel load"') - 600,
      SOURCE.indexOf('aria-label="Cancel load"') + 200,
    );
    assert.ok(control.length > 0, "expected the cancel control");
    assert.doesNotMatch(
      control,
      /status\?\.loaded/,
      "gating the cancel on a resident model reintroduces the bug",
    );
  });

  test(`the ${page} cancel routes through the backend unload`, () => {
    const handler = SOURCE.slice(
      SOURCE.indexOf("const handleCancelLoad = useCallback("),
      SOURCE.indexOf("useEffect(() => {\n    cancelLoadRef.current"),
    );
    assert.ok(handler.length > 0, "expected handleCancelLoad");
    assert.match(
      handler,
      /await handleUnload\(\)/,
      "unload is what aborts the load: it sets the cancel event and bumps the load token",
    );
    assert.match(handler, /if \(await handleUnload\(\)\)/);
    assert.match(handler, /Stopped loading the model/);
  });

  test(`the ${page} unload reports whether it succeeded`, () => {
    const unload = SOURCE.slice(
      SOURCE.indexOf("const handleUnload = useCallback("),
      SOURCE.indexOf("const handleCancelLoad = useCallback("),
    );
    assert.match(unload, /Promise<boolean>/);
    // handleLoad can restore tracking while this waits, so success means the load is gone.
    assert.match(unload, /return !loadTrackingRestored[.]current;/);
    assert.match(unload, /return false;/);
  });

  test(`the ${page} selector eject is fenced on the pending start too`, () => {
    // The fence lives in handleUnload so the selector's eject gets it too.
    const unload = SOURCE.slice(
      SOURCE.indexOf("const handleUnload = useCallback("),
      SOURCE.indexOf("const handleCancelLoad = useCallback("),
    );
    assert.match(unload, /const pending = pendingStart[.]current;/);
    assert.match(unload, /setBusy[(]"unloading"[)];[\s\S]*await pending;/);
    assert.match(
      unload,
      /await pending;[\s\S]*setBusy[(][(]prev[)] => [(]prev === "unloading" [?] null : prev[)][)];/,
    );
  });

  test(`the ${page} unload keeps a restored load visible instead of clearing it`, () => {
    // A failed compensating unload restores loading state, so finally must not clear busy.
    const unload = SOURCE.slice(
      SOURCE.indexOf("const handleUnload = useCallback("),
      SOURCE.indexOf("const handleCancelLoad = useCallback("),
    );
    assert.match(unload, /loadTrackingRestored[.]current = false;/);
    assert.match(unload, /return !loadTrackingRestored[.]current;/);
    assert.match(
      unload,
      /setBusy[(][(]prev[)] => [(]prev === "unloading" [?] null : prev[)][)];/,
    );
    assert.doesNotMatch(unload, /setBusy[(]null[)];/);
    // Restoring twice raises a duplicate toast and a second poll.
    const restore = SOURCE.slice(
      SOURCE.indexOf("const restoreLoadTracking = useCallback("),
      SOURCE.indexOf("}, [pollLoadProgress, cancelLoadFromToast]);"),
    );
    assert.match(restore, /loadTrackingRestored[.]current = true;/);
    const handler = SOURCE.slice(
      SOURCE.indexOf("const handleCancelLoad = useCallback("),
      SOURCE.indexOf("useEffect(() => {\n    cancelLoadRef.current"),
    );
    assert.match(handler, /if [(]!wasLoading [|][|] loadTrackingRestored[.]current[)] return;/);
  });

  test(`the ${page} eject is still offered only for a resident model`, () => {
    assert.match(
      SOURCE,
      /onEject=\{status\?\.loaded \? handleUnload : undefined\}/,
    );
  });

  test(`the ${page} cancel fences the pending start request`, () => {
    // Cancel can reach the backend before begin_load registers, so handleLoad must unload again.
    const load = SOURCE.slice(
      SOURCE.indexOf("const handleLoad = useCallback("),
      SOURCE.indexOf("// Set (or clear) the Transform"),
    );
    const body = load.length > 0 ? load : SOURCE.slice(SOURCE.indexOf("const handleLoad = useCallback("));
    assert.match(
      body,
      /const startSeq = cancelSeq\.current;/,
      "the cancel counter must be sampled BEFORE the start request goes out",
    );
    assert.match(body, /if \(startSeq !== cancelSeq\.current\) \{/);
    assert.match(
      body,
      /await unload(Diffusion|Video)Model\(\)/,
      "a cancel that raced the start must unload again once the load exists",
    );
    const raced = body.slice(body.indexOf("if (startSeq !== cancelSeq.current)"));
    assert.doesNotMatch(
      raced.slice(0, raced.indexOf("return settle(false);")),
      /void pollLoadProgress\(\)/,
    );
    // The compensating unload names no load, so it must not fire once a newer load owns the page.
    assert.match(body, /const startLoad = \+\+loadSeq\.current;/);
    assert.match(raced, /if \(startLoad === loadSeq\.current\) \{/);
  });

  test(`the ${page} cancelled poll leaves the status read to the unload`, () => {
    // Use the unload's response; /status can report resident during teardown.
    const poll = SOURCE.slice(
      SOURCE.indexOf("const pollLoadProgress = useCallback("),
      SOURCE.indexOf("}, [dismissLoadToast, refreshStatus, cancelLoadFromToast]);"),
    );
    const cancelled = poll.slice(poll.lastIndexOf("if (seq !== cancelSeq.current) {"));
    assert.ok(cancelled.length > 0, "expected the cancelled-status branch");
    assert.doesNotMatch(
      cancelled.slice(0, cancelled.indexOf("return;")),
      /refreshStatus\(\)/,
      "the cancelled branch must not allocate a ticket newer than the unload's",
    );
  });

  test(`the ${page} progress poll is invalidated by a cancel`, () => {
    // clearTimeout does not stop an in-flight tick, so it must recheck after awaiting.
    const poll = SOURCE.slice(
      SOURCE.indexOf("const pollLoadProgress = useCallback("),
      SOURCE.indexOf("}, [dismissLoadToast, refreshStatus, cancelLoadFromToast]);"),
    );
    assert.match(poll, /const seq = cancelSeq\.current;/);
    assert.match(poll, /if \(seq !== cancelSeq\.current\) return;/);
    assert.match(poll, /const loaded = await get(Diffusion|Video)Status\(\);\s*\n\s*if \(seq !== cancelSeq\.current\) \{/);
  });

  test(`the ${page} cancel counter is bumped by every teardown`, () => {
    const drop = SOURCE.slice(
      SOURCE.indexOf("const dropResidentState = useCallback("),
      SOURCE.indexOf(
        "}, [dismissLoadToast,",
        SOURCE.indexOf("const dropResidentState = useCallback("),
      ),
    );
    assert.match(
      drop,
      /cancelSeq\.current \+= 1;/,
      "an eject from the loaded-models card cancels a load too, so it must fence as well",
    );
  });

  test(`the ${page} restores load tracking when the unload fails`, () => {
    // dropResidentState already killed the poll, so a failed unload must restore tracking.
    const handler = SOURCE.slice(
      SOURCE.indexOf("const handleCancelLoad = useCallback("),
      SOURCE.indexOf("useEffect(() => {\n    cancelLoadRef.current"),
    );
    assert.match(handler, /const wasLoading = busy === "loading";/);
    assert.match(handler, /restoreLoadTracking\(\);/);
    const restore = SOURCE.slice(
      SOURCE.indexOf("const restoreLoadTracking = useCallback("),
      SOURCE.indexOf("}, [pollLoadProgress, cancelLoadFromToast]);"),
    );
    assert.match(restore, /setBusy\("loading"\);/);
    assert.match(restore, /loadToastId\.current = toast\(/);
    assert.match(restore, /void pollLoadProgress\(\);/);
  });

  test(`the ${page} cancel holds the page until a pending start settles`, () => {
    // begin_load refuses a second load while one is registered, so stay busy until it settles.
    const handler = SOURCE.slice(
      SOURCE.indexOf("const handleUnload = useCallback("),
      SOURCE.indexOf("useEffect(() => {\n    cancelLoadRef.current"),
    );
    assert.match(handler, /const pending = pendingStart\.current;/);
    assert.match(handler, /setBusy\("unloading"\);/);
    assert.match(handler, /await pending;/);
    assert.match(handler, /if \(await handleUnload\(\)\) \{/);
    const load = SOURCE.slice(
      SOURCE.indexOf("const handleLoad = useCallback("),
      SOURCE.indexOf("const handleLoad = useCallback(") + 3000,
    );
    assert.match(load, /pendingStart\.current = inFlight;/);
    assert.match(load, /const settle = \(started: boolean\): boolean => \{/);
    assert.doesNotMatch(load, /pendingStart\.current = startRequest;/);
  });

  test(`the ${page} recovers when the compensating unload fails`, () => {
    // This request is the last chance to stop the missed load, so its failure must surface.
    const at = SOURCE.indexOf("if (startSeq !== cancelSeq.current)");
    const raced = SOURCE.slice(at, at + 1400);
    assert.match(raced, /restoreLoadTracking\(\);/);
  });

  test(`the ${page} external eject is fenced by the pending start too`, () => {
    // The loaded-models card ejects without handleCancelLoad, so it needs the same fence.
    const listener = SOURCE.slice(
      SOURCE.indexOf("subscribeModelEjected("),
      SOURCE.indexOf("subscribeModelEjected(") + 1600,
    );
    assert.match(listener, /const pending = pendingStart\.current;/);
    assert.match(listener, /setBusy\(\(prev\) => \(prev === "loading" \? "unloading" : prev\)\);/);
  });

  test(`the ${page} cancel names the load, not the download`, () => {
    const control = SOURCE.slice(
      SOURCE.indexOf('aria-label="Cancel load"'),
      SOURCE.indexOf('aria-label="Cancel load"') + 400,
    );
    assert.doesNotMatch(control, /Cancel download/);
    assert.match(control, /Stop loading this model/);
  });
}

test("the download manager keeps its own, differently named cancel", () => {
  assert.match(DOWNLOAD_PANEL, /"Cancel download"/);
});

test("cancelling a deploy does not leave the adapter queued", () => {
  // pendingDeploy applies to the next resident model, so cancels must clear it.
  const SOURCE = readText("../src/features/images/images-page.tsx");
  const drop = SOURCE.slice(
    SOURCE.indexOf("const dropResidentState = useCallback("),
    SOURCE.indexOf(
      "}, [dismissLoadToast,",
      SOURCE.indexOf("const dropResidentState = useCallback("),
    ),
  );
  assert.match(drop, /pendingDeploy\.current = null;/);
});
