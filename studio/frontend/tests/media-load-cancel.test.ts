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
    // the mount-time resume of a load started elsewhere, and re-created in place
    // by each progress tick. A tick that dropped the action would make the
    // button vanish one second into the load.
    // Call sites only: the declaration's own argument list opens with a newline.
    const sites = SOURCE.match(/loadToastArgs\((?!\n)[^)\n]*\)/g) ?? [];
    assert.equal(sites.length, 3, "expected the three load-toast call sites");
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
