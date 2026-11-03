// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { replaceRef } from "../src/features/recipe-studio/utils/refs.ts";

test("renaming a column rewrites it wherever it is a variable root, and nowhere else", () => {
  const before =
    "{{ ticket }} / {{ ticket.issue_title }} {% if ticket.priority == 'P1' %}urgent{% endif %} " +
    "{{ ticket_id }} {{ other.ticket }} {% if x == 'the ticket' %}{% endif %} ticket";
  assert.equal(
    replaceRef(before, "ticket", "support_ticket"),
    "{{ support_ticket }} / {{ support_ticket.issue_title }} {% if support_ticket.priority == 'P1' %}urgent{% endif %} " +
      "{{ ticket_id }} {{ other.ticket }} {% if x == 'the ticket' %}{% endif %} ticket",
  );
});
