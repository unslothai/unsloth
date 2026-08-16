// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Where a tab learns that the RUNNING model was loaded with its vision projector off.
//
// The load paths set the switch for the tab that performed the load. A tab that
// reloads the page, opens fresh, or comes back after Studio restarts never saw that
// load, so without a seed from /api/inference/status the Advanced Settings Vision
// switch renders ON over a model running with the projector off -- and the next
// Reload model silently puts the projector back, undoing both the setting and the
// VRAM it freed. Observed in Chromium, Firefox and WebKit against a real Studio.
//
// tensorParallel next door in the same applier already does this; the toggle was
// written to follow that pattern and this is the hop it was missing. Checked at the
// source, like the chat-template and llama-extra-args seed tests next door: the
// applier is one large object literal with no seam to call.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (relative: string) =>
  readFileSync(path.join(HERE, "..", relative), "utf8");

const APPLIER = read("src/features/chat/lib/apply-inference-status-to-store.ts");
const API_TYPES = read("src/features/chat/types/api.ts");
const THREAD = read("src/components/assistant-ui/thread.tsx");
const RUNTIME_PROVIDER = read("src/features/chat/runtime-provider.tsx");

test("both response types carry the raw disable_vision echo", () => {
  // vision_disabled_by_user is additionally gated on the model HAVING a projector,
  // so it cannot round-trip the switch on a text-only GGUF. The seed needs the
  // request the load actually ran with, and it has to exist on the load response
  // and the status poll alike.
  assert.equal(API_TYPES.match(/^ {2}disable_vision\?: boolean;$/gm)?.length, 2);
});

test("the applier seeds the switch from the status echo", () => {
  assert.match(APPLIER, /status\.disable_vision !== undefined/);
  assert.match(APPLIER, /disableVision: status\.disable_vision,/);
});

test("the seed uses the same unseeded guard as tensorParallel", () => {
  // Unguarded it would fight the user: every 10s poll would stamp the running
  // model's value over a switch that was flipped but not yet applied.
  assert.match(
    APPLIER,
    /status\.disable_vision !== undefined &&\s*\n\s*\(prevState\.loadedVisionDisabledByUser === null \|\| hydratingExistingModel\) && \{\s*\n\s*disableVision: status\.disable_vision,/,
  );
});

test("the composer's own mirror of the flag stays unguarded", () => {
  // loadedVisionDisabledByUser is what the image gate reads. It mirrors the live
  // load rather than a user setting, so every poll must land on it -- adding the
  // seed guard here would freeze the refusal string at the first read.
  assert.match(
    APPLIER,
    /status\.vision_disabled_by_user !== undefined && \{\s*\n\s*loadedVisionDisabledByUser: status\.vision_disabled_by_user,/,
  );
});

test("an older backend that omits the field changes nothing", () => {
  assert.doesNotMatch(APPLIER, /disableVision: status\.disable_vision \?\? false/);
});

test("a rejection the adapter already toasted does not draw a second message", () => {
  // Pasting an image with Vision off answered with the correct reason AND a flat
  // contradiction of it -- "the clipboard item is unsupported, unreadable, or
  // exceeds its size limit" -- because the adapter's deliberate reject reached
  // pasteClipboardFiles' generic onError. Verified in Chromium and WebKit.
  assert.match(RUNTIME_PROVIDER, /throw attachmentRejectionAlreadyToasted\(unavailableReason\)/);
  assert.match(THREAD, /isAttachmentRejectionAlreadyToasted\(result\.reason\)/);
  // A rejection nobody explained must still reach the generic toast.
  assert.match(THREAD, /if \(unexplained\.length > 0\) \{/);
});
