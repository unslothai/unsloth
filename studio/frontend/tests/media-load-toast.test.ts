// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { register } from "node:module";

register("./helpers/toast-resolver.mjs", import.meta.url);

const { calls } = await import("./helpers/toast-stub.mjs");
const { createMediaLoadToast } = await import("../src/lib/media-load-toast.ts");

type Options = {
  id?: string | number;
  description?: unknown;
  cancel?: { label: string; onClick: () => void };
  onDismiss?: (t: { id: string | number }) => void;
};
type Call = { kind: string; id?: string | number; options?: Options };

function make() {
  calls.length = 0;
  const closed: unknown[] = [];
  const loadToast = createMediaLoadToast<string>({
    toastOptions: (progress, cancel) => ({
      description: progress,
      cancel: { label: "Cancel", onClick: cancel },
    }),
    onClosedChange: (progress) => closed.push(progress),
  });
  return { loadToast, closed };
}

const raised = () => (calls as Call[]).filter((c) => c.kind === "default");

/** What Sonner does when the user clicks the toast's X: it calls the raise's onDismiss. */
function userClose(raise: Call) {
  const id = raise.options?.id;
  assert.ok(id != null, "every load toast in these tests is raised on a known id");
  raise.options?.onDismiss?.({ id });
}

test("progress updates raise the same toast until the user closes it", () => {
  const { loadToast, closed } = make();
  loadToast.open("starting", "pick-1");
  loadToast.update("downloading 10%");
  assert.deepEqual(
    raised().map((c) => [c.options?.id, c.options?.description]),
    [
      ["pick-1", "starting"],
      ["pick-1", "downloading 10%"],
    ],
  );
  assert.deepEqual(closed, []);
});

test("after a user close, progress goes to the inline status and the toast is not raised again", () => {
  const { loadToast, closed } = make();
  loadToast.open("starting", "load-1");
  loadToast.update("downloading 10%");
  userClose(raised()[0]);
  const before = raised().length;
  loadToast.update("downloading 40%");
  loadToast.update("loading to GPU");
  assert.equal(raised().length, before, "a closed toast must not be raised again");
  assert.deepEqual(closed, ["downloading 10%", "downloading 40%", "loading to GPU"]);
});

test("closing right after open shows the progress the toast opened with", () => {
  // A resumed load opens on its current progress, so an early close must not show an idle "Starting".
  const { loadToast, closed } = make();
  loadToast.open("downloading 75%", "load-1");
  userClose(raised()[0]);
  assert.deepEqual(closed, ["downloading 75%"]);
});

test("the load's own dismiss clears the inline status, and its late onDismiss is ignored", () => {
  const { loadToast, closed } = make();
  loadToast.open("downloading 10%", "load-1");
  userClose(raised()[0]);
  loadToast.dismiss();
  assert.deepEqual(closed, ["downloading 10%", null]);
  // A dismissed toast's onDismiss fires after the fact; it must not reopen the inline status.
  loadToast.open("starting", "load-2");
  loadToast.dismiss();
  userClose(raised().at(-1) as Call);
  loadToast.update("downloading 10%");
  assert.deepEqual(closed, ["downloading 10%", null]);
  assert.deepEqual(
    (calls as Call[]).filter((c) => c.kind === "dismiss").map((c) => c.id),
    ["load-2"],
  );
});

test("an earlier toast's close does not close the next load's toast", () => {
  const { loadToast, closed } = make();
  loadToast.open("starting", "load-1");
  const first = raised()[0];
  loadToast.open("starting", "load-2");
  userClose(first);
  loadToast.update("downloading 10%");
  assert.deepEqual(raised().at(-1)?.options?.id, "load-2");
  assert.deepEqual(closed, []);
});

test("every raise carries a Cancel that reaches the handler set after render", () => {
  const { loadToast } = make();
  loadToast.open("starting", "load-1");
  let cancelled = 0;
  loadToast.setCancel(() => cancelled++);
  loadToast.update("downloading 10%");
  for (const raise of raised()) {
    assert.equal(raise.options?.cancel?.label, "Cancel");
    raise.options?.cancel?.onClick();
  }
  assert.equal(cancelled, 2);
});
