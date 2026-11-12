// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A reload during a Mercury diffusing run hands the reply to the recovery follower. It must apply
// the same replace semantics as the live stream: the checkpoint may already end with a snapshot,
// and the closing delta.content is the whole text again, so appending would duplicate the answer.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { createDiffusionReplace, recoveredDiffusionStart } = await import(
  "../src/features/chat/utils/diffusion-replace.ts"
);

type Frame = { replace?: string; content?: string; toolEnd?: boolean };

// The follower's text path, as runtime-provider.tsx wires it.
function follow(
  checkpoint: string,
  frames: Frame[],
  carriedAts: number[] = [],
  continued?: string,
): string {
  let raw = checkpoint;
  const diffusion = createDiffusionReplace();
  let inCheckpoint = true;
  for (const frame of frames) {
    if (frame.toolEnd) {
      diffusion.endTurn();
      inCheckpoint = false;
      continue;
    }
    const text = frame.replace ?? frame.content ?? "";
    if (!text) continue;
    const firstStart =
      !diffusion.active && inCheckpoint
        ? recoveredDiffusionStart(raw, carriedAts, continued)
        : undefined;
    inCheckpoint = false;
    const cut = diffusion.cutAt(raw.length, text, firstStart);
    if (cut !== null) raw = raw.slice(0, cut) + text;
  }
  return raw;
}

const FINAL = "The capital of France is Paris.";

test("replace frames then the final content yield the final text once", () => {
  const frames = [
    { replace: "The cxpital of Frxnce is Pxris." },
    { replace: "The capital of France is Pxris." },
    { replace: FINAL },
    { content: FINAL },
  ];
  assert.equal(follow("", frames), FINAL);
  // Reloaded after a checkpoint that already saved a snapshot.
  assert.equal(follow("The cxpital of Frxnce is", frames.slice(1)), FINAL);
  // Reloaded after the last frame: only the closing content is left to replay.
  assert.equal(follow("The capital of France is Pxris.", [{ content: FINAL }]), FINAL);
});

test("the checkpoint's reasoning, cards and continued partial survive the swap", () => {
  assert.equal(
    follow("<think>plan</think>Hxllo", [{ content: "Hello" }]),
    "<think>plan</think>Hello",
  );
  assert.equal(follow("Before card.Hxllo", [{ content: "Hello" }], [12]), "Before card.Hello");
  assert.equal(
    follow("Once upon a time Hxllo", [{ content: "Hello" }], [], "Once upon a time "),
    "Once upon a time Hello",
  );
});

test("a tool round ends the turn, so the next turn's snapshots append after it", () => {
  assert.equal(
    follow("", [
      { replace: "Lxt me check." },
      { content: "Let me check." },
      { toolEnd: true },
      { replace: "It is sxnny." },
      { content: "It is sunny." },
    ]),
    "Let me check.It is sunny.",
  );
});

test("the recovery follower uses the shared replace helper", () => {
  const src = readSrc("features/chat/runtime-provider.tsx");
  assert.match(src, /const diffusion = createDiffusionReplace\(\);/);
  assert.match(
    src,
    /typeof diffusionReplace === "string"\s*\? diffusionReplace\s*: deltaRecord\?\.content/,
  );
  assert.match(
    src,
    /typeof diffusionReplace === "string" \|\|\s*diffusion\.active \|\|\s*update\.run\.requestPayload\.diffusing === true/,
  );
  assert.match(
    src,
    /const cut = diffusion\.cutAt\(raw\.length, delta, firstStart\);\s*if \(cut !== null\) raw = raw\.slice\(0, cut\) \+ delta;/,
  );
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /endProviderTurn\(\);\s*\/\/[^\n]*\n\s*diffusion\.endTurn\(\);/);
});
