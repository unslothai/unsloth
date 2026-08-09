// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A first load on the Images or Video page had no stop button. The selector's
// eject is the control that cancels an in-flight load -- the backend's unload
// sets the running load's cancel event and bumps its token -- but the pages
// only pass `onEject` when `status.loaded` is true, and nothing is resident
// until the load commits. So the one control that would have stopped a
// multi-gigabyte pull was hidden for exactly its duration.
//
// Two exposures fix it, and both are asserted here because either alone can be
// missed: a Cancel action on the persistent load toast (chat's pattern), and a
// real "Cancel load" button beside the selector, which survives the toast being
// dismissed and is reachable by keyboard (the trigger's eject hit area is an
// aria-hidden span, mouse-only by design).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const PAGES = [
  ["Images", "../src/features/images/images-page.tsx"],
  ["Video", "../src/features/video/video-page.tsx"],
] as const;

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
}

const DOWNLOAD_PANEL = read(
  "../src/features/hub/download-manager/download-manager-panel.tsx",
);

for (const [page, path] of PAGES) {
  const SOURCE = read(path);

  test(`the ${page} load toast offers a Cancel action`, () => {
    assert.match(
      SOURCE,
      /cancel: \{ label: "Cancel", onClick: onCancel \}/,
      "the load toast must carry a cancel action, not just a close button",
    );
    // Every toast site has to pass it: the toast is created by handleLoad, by
    // the mount-time resume of a load started elsewhere, re-created in place by
    // each progress tick, and rebuilt by handleCancelLoad when the unload fails
    // and the load it could not stop has to stay visible. A tick that dropped
    // the action would make the button vanish one second into the load.
    // Call sites only: the declaration's own argument list opens with a newline.
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
    // The whole bug: `status?.loaded` is false for the span of a first load, so
    // anything gated on it is invisible exactly when the user wants it.
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
      SOURCE.indexOf("const handleCancelLoad = useCallback(") + 400,
    );
    assert.ok(handler.length > 0, "expected handleCancelLoad");
    assert.match(
      handler,
      /await handleUnload\(\)/,
      "unload is what aborts the load: it sets the cancel event and bumps the load token",
    );
    // Only report a stop the backend actually accepted, or a failed unload
    // would claim the load had stopped while it kept downloading.
    assert.match(handler, /if \(await handleUnload\(\)\)/);
    assert.match(handler, /Stopped loading the model/);
  });

  test(`the ${page} unload reports whether it succeeded`, () => {
    const unload = SOURCE.slice(
      SOURCE.indexOf("const handleUnload = useCallback("),
      SOURCE.indexOf("const handleUnload = useCallback(") + 700,
    );
    assert.match(unload, /Promise<boolean>/);
    assert.match(unload, /return true;/);
    assert.match(unload, /return false;/);
  });

  test(`the ${page} eject is still offered only for a resident model`, () => {
    // Unchanged wiring: eject remains the resident-model control. The fix adds
    // a second control rather than widening this one, because "Eject" is the
    // wrong word for stopping a load that never produced anything.
    assert.match(
      SOURCE,
      /onEject=\{status\?\.loaded \? handleUnload : undefined\}/,
    );
  });


  test(`the ${page} cancel fences the pending start request`, () => {
    // Cancel is reachable the instant `busy` turns "loading", which is before the start request
    // has even been sent. Its unload can therefore reach the backend BEFORE begin_load registers
    // the load, find nothing to stop, and return success -- after which the load runs on with no
    // toast and no Cancel button. handleLoad has to notice that and unload again.
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
    // ...and must NOT then start polling a load it just cancelled.
    const raced = body.slice(body.indexOf("if (startSeq !== cancelSeq.current)"));
    assert.doesNotMatch(
      raced.slice(0, raced.indexOf("return false;")),
      /void pollLoadProgress\(\)/,
    );
  });

  test(`the ${page} progress poll is invalidated by a cancel`, () => {
    // clearTimeout stops the NEXT tick. A tick already awaiting its response still lands, and its
    // ready branch would announce "Model loaded" and issue a status ticket newer than the
    // unload's, so the unloaded answer is dropped as stale and the controls keep advertising a
    // model that is gone.
    const poll = SOURCE.slice(
      SOURCE.indexOf("const pollLoadProgress = useCallback("),
      SOURCE.indexOf("}, [dismissLoadToast, refreshStatus, cancelLoadFromToast]);"),
    );
    assert.match(poll, /const seq = cancelSeq\.current;/);
    assert.match(poll, /if \(seq !== cancelSeq\.current\) return;/);
    // The status read is the one that has to be checked on BOTH sides of its await.
    assert.match(poll, /const loaded = await get(Diffusion|Video)Status\(\);\s*\n\s*if \(seq !== cancelSeq\.current\) \{/);
  });

  test(`the ${page} cancel counter is bumped by every teardown`, () => {
    const drop = SOURCE.slice(
      SOURCE.indexOf("const dropResidentState = useCallback("),
      SOURCE.indexOf("}, [dismissLoadToast, pickGuard]);"),
    );
    assert.match(
      drop,
      /cancelSeq\.current \+= 1;/,
      "an eject from the loaded-models card cancels a load too, so it must fence as well",
    );
  });

  test(`the ${page} restores load tracking when the unload fails`, () => {
    // dropResidentState has already killed the poll and the toast by the time the unload's
    // failure is known, and refreshStatus cannot bring them back: a first load has nothing
    // resident to report. Without a restore the load keeps running, invisibly and uncancellable.
    const handler = SOURCE.slice(
      SOURCE.indexOf("const handleCancelLoad = useCallback("),
      SOURCE.indexOf("useEffect(() => {\n    cancelLoadRef.current"),
    );
    assert.match(handler, /const wasLoading = busy === "loading";/);
    assert.match(handler, /setBusy\("loading"\);/);
    assert.match(handler, /loadToastId\.current = toast\(/);
    assert.match(handler, /void pollLoadProgress\(\);/);
  });

  test(`the ${page} cancel names the load, not the download`, () => {
    // A user mid-load can have a staged download in the manager panel too, and
    // the two stop different things: this one abandons the load, that one stops
    // a queued pull. Same word in both corners would be its own bug.
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
  // handleDeployAdapter parks the trained adapter in pendingDeploy and loads its base. That ref
  // is applied to whatever LoRA-capable model becomes resident NEXT, so a cancelled deploy would
  // silently mix a discarded adapter into an unrelated model's generations. Clearing it belongs
  // in dropResidentState, which every cancel and every eject already runs.
  const SOURCE = read("../src/features/images/images-page.tsx");
  const drop = SOURCE.slice(
    SOURCE.indexOf("const dropResidentState = useCallback("),
    SOURCE.indexOf("}, [dismissLoadToast, pickGuard]);"),
  );
  assert.match(drop, /pendingDeploy\.current = null;/);
});
