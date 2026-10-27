// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("Edit numbers its steps in text, not circled glyphs", () => {
  const policy = readSrc("features/audio/edit-policy.ts");
  assert.match(policy, /transcriptLabel: "1\. Check the transcript"/);
  assert.match(policy, /changesLabel: "2\. Make your changes"/);
  assert.match(policy, /deliveryLabel: "2\. Change the delivery"/);
  for (const line of policy.split("\n")) {
    if (/[①②]/.test(line)) assert.match(line.trim(), /^(\/\/|\/\*|\*)/);
  }
});

test("Transcribe buttons use the Transcribe workflow's icon, sized from the label", () => {
  assert.match(
    readSrc("features/audio/workflows.ts"),
    /id: "transcribe",[\s\S]{0,80}icon: ClosedCaptionIcon/,
  );
  for (const page of ["edit", "clone", "convert"]) {
    const src = readSrc(`features/audio/pages/${page}-page.tsx`);
    assert.match(
      src,
      /<HugeiconsIcon icon=\{ClosedCaptionIcon\} className="size-\[1em\]" \/>\s*\{transcriber\.transcribing \? "Transcribing…" : "Transcribe"\}/,
    );
    assert.doesNotMatch(src, /SpeechToTextIcon/);
  }
});
