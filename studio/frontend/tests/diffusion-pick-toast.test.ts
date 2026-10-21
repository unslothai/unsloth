// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { register } from "node:module";

register("./helpers/toast-resolver.mjs", import.meta.url);

const { calls } = await import("./helpers/toast-stub.mjs");
const { createPickToast } = await import("../src/lib/diffusion-pick-toast.ts");
const { readSrc } = await import("./helpers/kit.ts");

type Raise = {
  kind: string;
  id?: string;
  options?: { id?: string; description?: unknown; onDismiss?: () => void };
};

function make() {
  calls.length = 0;
  return createPickToast({
    describe: (phase, progress) =>
      progress ? `${phase} ${progress.downloadedBytes}/${progress.totalBytes}` : phase,
  });
}

function shown(): Raise[] {
  return (calls as Raise[]).slice();
}

test("a pick raises its toast at once, before any plan or download", () => {
  const pick = make();
  const id = pick.show();
  assert.deepEqual(
    shown().map((c) => [c.kind, c.options?.id, c.options?.description]),
    [["default", id, "preparing"]],
  );
});

test("a newer pick replaces the older toast, and the stale pick cannot touch it", () => {
  const pick = make();
  const older = pick.show();
  const newer = pick.show();
  assert.notEqual(older, newer);
  assert.deepEqual(shown()[1], { kind: "dismiss", id: older });

  calls.length = 0;
  // The superseded pick's plan resolves late and cleans up after itself.
  pick.dismiss(older);
  pick.setPhase(older, "downloading", 1);
  assert.deepEqual(shown(), []);
  assert.equal(pick.take(older), undefined);
});

test("progress shows only while downloading, and each phase starts from its own plan", () => {
  const pick = make();
  const id = pick.show();
  calls.length = 0;
  pick.progress({ downloadedBytes: 5, totalBytes: 10, plan: 1 });
  assert.deepEqual(shown(), [], "a queued or preparing pick showed another plan's bytes");

  pick.setPhase(id, "downloading", 2);
  pick.progress({ downloadedBytes: 1, totalBytes: 10, plan: 2 });
  pick.progress({ downloadedBytes: 1, totalBytes: 10, plan: 2 });
  pick.progress({ downloadedBytes: 4, totalBytes: 10, plan: 2 });
  assert.deepEqual(
    shown().map((c) => c.options?.description),
    ["downloading", "downloading 1/10", "downloading 4/10"],
  );
});

test("a download-only plan staged after the pick's own never shows in its toast", () => {
  // A queued download-only plan uses the same progress hook after the pick finishes.
  const pick = make();
  const id = pick.show();
  pick.setPhase(id, "downloading", 3);
  pick.progress({ downloadedBytes: 10, totalBytes: 10, plan: 3 });
  calls.length = 0;

  pick.progress({ downloadedBytes: 2, totalBytes: 50, plan: 4 });
  assert.deepEqual(shown(), [], "another plan's bytes reached the pick's toast");

  pick.setPhase(id, "waiting");
  pick.progress({ downloadedBytes: 20, totalBytes: 50, plan: 4 });
  assert.deepEqual(shown().map((c) => c.options?.description), ["waiting"]);
});

test("the Images and Video toasts never share an id", () => {
  // Both pages stay mounted and share Sonner's store.
  const images = make();
  const video = createPickToast({ describe: (phase) => phase });
  const imagesId = images.show();
  const videoId = video.show();
  assert.notEqual(imagesId, videoId);

  calls.length = 0;
  images.dismissAll();
  assert.deepEqual(shown(), [{ kind: "dismiss", id: imagesId }]);
  video.setPhase(videoId, "downloading", 1);
  assert.equal(shown().at(-1)?.options?.id, videoId, "the Video toast went with the Images one");
});

test("the load takes the toast over in place and later ticks leave it alone", () => {
  const pick = make();
  const id = pick.show();
  pick.setPhase(id, "downloading", 1);
  assert.equal(pick.take(id), id);

  calls.length = 0;
  pick.progress({ downloadedBytes: 9, totalBytes: 10, plan: 1 });
  pick.dismissAll();
  assert.deepEqual(shown(), [], "the pick kept writing to the load's toast");
  assert.equal(pick.take(id), undefined, "one toast was handed over twice");
});

test("a toast the user closed is not raised again by the next tick", () => {
  const pick = make();
  const id = pick.show();
  pick.setPhase(id, "downloading", 1);
  const closed = shown().at(-1)?.options?.onDismiss;
  assert.ok(closed);
  closed();

  calls.length = 0;
  pick.progress({ downloadedBytes: 3, totalBytes: 10, plan: 1 });
  assert.deepEqual(shown(), []);
  // The load then raises its own toast rather than reusing the closed id.
  assert.equal(pick.take(id), undefined);
});

test("a cancelled or retired pick drops its toast", () => {
  const pick = make();
  const id = pick.show();
  calls.length = 0;
  pick.dismiss(id);
  assert.deepEqual(shown(), [{ kind: "dismiss", id }]);

  const next = pick.show();
  calls.length = 0;
  pick.dismissAll();
  assert.deepEqual(shown(), [{ kind: "dismiss", id: next }]);
  pick.dismiss(undefined);
  assert.equal(shown().length, 1, "an absent id dropped something");
});

test("every cancelled pick drops its toast on both pages", () => {
  // A cancelled pick can no longer load, so a toast left up would promise a load that never comes.
  for (const page of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
    const lines = readSrc(page).split("\n");
    const cancels = lines.flatMap((line, i) => (line.includes("pickGuard.cancel();") ? [i] : []));
    assert.ok(cancels.length > 0, page);
    for (const i of cancels) {
      const after = lines
        .slice(i + 1, i + 4)
        .filter((line) => !line.trim().startsWith("//"))
        .join("\n");
      assert.match(after, /pickToast\.dismissAll\(\);/, `${page}:${i + 1}`);
    }
  }
});

