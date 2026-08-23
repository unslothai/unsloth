// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { busyUpdateStartFeedback } from "../src/features/hub/catalog/update-start-feedback.ts";

const updateStartCall = /requestDownloadStart\(variant, expectedBytes\)/;
const startingSlot =
  /!selected\?\.downloaded \|\|\s*downloadToCurrentCache \|\|\s*downloadingThisVariant \|\|\s*cancelling \|\|\s*downloadAction\.starting/;
const hiddenUpdateAction = /updateAvailable &&\s*!downloadAction\.starting/;

test("Hub update confirmations report an occupied download slot", () => {
  assert.deepEqual(busyUpdateStartFeedback("busy"), {
    title: "A download for this model is already in progress",
    description: "Try updating again once it finishes.",
  });
  for (const outcome of ["started", "conflict", "error"] as const) {
    assert.equal(busyUpdateStartFeedback(outcome), null);
  }
});

test("GGUF updates retain the action slot while the start request is pending", () => {
  const card = readFileSync(
    new URL(
      "../src/features/hub/catalog/gguf-download-card.tsx",
      import.meta.url,
    ),
    "utf8",
  );

  assert.match(card, updateStartCall);
  assert.match(card, startingSlot);
  assert.match(card, hiddenUpdateAction);
});
