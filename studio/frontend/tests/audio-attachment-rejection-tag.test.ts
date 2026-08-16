// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The attach menu's new catch draws "Could not attach that file." for any rejection
// the adapter did NOT already explain. AudioAttachmentAdapter.add toasts its refusal
// ("... cannot accept audio", the size cap, "Only one audio file ...") and then
// rejects, so an untagged throw there stacks a second toast on top of the first for
// every audio refusal picked from that menu. Composite routing sends any audio file
// to that adapter -- VisionImageAdapter's accept is image/jpeg,png,webp,gif -- so the
// picker path reaches it directly.
//
// Source-level like its vision siblings: the adapter reaches the chat barrel and the
// "@/" alias, neither of which --experimental-strip-types can resolve.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import {
  attachmentRejectionAlreadyToasted,
  isAttachmentRejectionAlreadyToasted,
} from "../src/features/chat/utils/attachment-rejection.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (relative: string) =>
  readFileSync(path.join(HERE, "..", relative), "utf8");

const AUDIO_ADAPTER = read("src/features/chat/audio-attachment-adapter.ts");
const THREAD = read("src/components/assistant-ui/thread.tsx");

const TOAST_THEN_THROW = /toast\.error\([^;]*\);\s*\n\s*throw ([^;]+);/g;

test("every audio refusal the adapter toasts rejects with the tag", () => {
  const throws = [...AUDIO_ADAPTER.matchAll(TOAST_THEN_THROW)].map((m) => m[1]);
  assert.ok(throws.length >= 3, "the audio refusal branches vanished");
  for (const thrown of throws) {
    assert.match(
      thrown,
      /attachmentRejectionAlreadyToasted\(/,
      `audio adapter toasts then throws \`${thrown}\`: the attach menu's catch adds a second "Could not attach that file." toast on top of it`,
    );
  }
});

test("the attach menu suppresses exactly the tagged rejections", () => {
  assert.match(THREAD, /if \(isAttachmentRejectionAlreadyToasted\(error\)\) return;/);
  assert.match(THREAD, /toast\.error\("Could not attach that file\."/);
});

test("the tag survives the catch predicate", () => {
  assert.equal(
    isAttachmentRejectionAlreadyToasted(
      attachmentRejectionAlreadyToasted("Only one audio file can be attached per message."),
    ),
    true,
  );
  assert.equal(
    isAttachmentRejectionAlreadyToasted(new Error("Image size exceeds 20MB limit")),
    false,
  );
});
