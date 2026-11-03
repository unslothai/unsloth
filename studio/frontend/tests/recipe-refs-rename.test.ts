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

test("a filter or test with the column's name is left alone", () => {
  assert.equal(
    replaceRef(
      "{{ length }} {{ response | length }} {% if x is length %}{% endif %} {{ length + 1 }}",
      "length",
      "word_count",
    ),
    "{{ word_count }} {{ response | length }} {% if x is length %}{% endif %} {{ word_count + 1 }}",
  );
});

test("a keyword argument with the column's name is left alone", () => {
  assert.equal(
    replaceRef(
      "{{ x | truncate(length=10) }} {{ f(length = 2) }} {% if length == 3 %}{% endif %}",
      "length",
      "word_count",
    ),
    "{{ x | truncate(length=10) }} {{ f(length = 2) }} {% if word_count == 3 %}{% endif %}",
  );
});
