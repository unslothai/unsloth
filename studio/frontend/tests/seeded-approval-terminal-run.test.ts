// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A tool call parked on its approval before the tab closed must get Approve/Deny back during a faithful
// replay. #10910 moved that handling out of the old arm/disarm lifecycle (createGenerationToolRecovery's
// armSeededApprovals / disarmAll) and into the replay engine: createRecoveryReplay re-raises a parked card as
// it folds the run's frames, through the toolConfirmations callbacks the recovery wires to the store. This
// pins that wiring in the recovery -- deleting it leaves the other tests green but silently drops the cards.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SOURCE = fileURLToPath(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
);

test("the recovery drives faithful replay through the #10910 engine", () => {
  const text = readFileSync(SOURCE, "utf8");
  assert.match(
    text,
    /createRecoveryReplay\(/,
    "scheduleGenerationRecovery must build the follower's reply via createRecoveryReplay; that is where " +
      "parked-card re-raising and think/reasoning splitting now live",
  );
});

test("parked cards are re-raised by wiring register to setToolConfirmation", () => {
  const text = readFileSync(SOURCE, "utf8");
  assert.match(
    text,
    /register:\s*\(id,\s*approvalId\)[\s\S]{0,160}?setToolConfirmation\(/,
    "toolConfirmations.register must re-raise the parked card via setToolConfirmation, or a call parked " +
      "before the tab closed never gets its Approve/Deny back on replay",
  );
});

test("parked cards are cleared by wiring resolve to clearToolConfirmation", () => {
  const text = readFileSync(SOURCE, "utf8");
  assert.match(
    text,
    /resolve:\s*\(id\)[\s\S]{0,120}?clearToolConfirmation\(/,
    "toolConfirmations.resolve must clear the card via clearToolConfirmation, so a settled call does not " +
      "leave a stale entry that skews soleRequest's Enter/Escape chords",
  );
});
