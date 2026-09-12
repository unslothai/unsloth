// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const mentionsSource = readFileSync(
  new URL("../src/components/assistant-ui/skill-mentions.tsx", import.meta.url),
  "utf8",
);
const composerSource = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);

test("the compare composer exposes its skill suggestions as a linked combobox", () => {
  assert.ok(composerSource.includes("{...skillMentions.inputProps}"));
  for (const semantic of [
    'role: "combobox"',
    '"aria-controls": open ? listboxId : undefined',
    '"aria-activedescendant": activeOptionId',
    'role="listbox"',
    'role="option"',
    "aria-selected={index === highlighted}",
  ]) {
    assert.ok(mentionsSource.includes(semantic), semantic);
  }
});
