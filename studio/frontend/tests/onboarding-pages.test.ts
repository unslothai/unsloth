// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  ONBOARDING_PAGES,
  START_TOURS,
  decideOnboarding,
  greetingName,
} from "../src/features/onboarding/lib/pages.ts";

const record = (
  over: Partial<Parameters<typeof decideOnboarding>[0]> = {},
) => ({
  saved: false,
  pages: [] as string[],
  completedAt: null,
  hasChatHistory: false,
  ...over,
});

test("a fresh account sees every page", () => {
  assert.deepEqual(decideOnboarding(record()), {
    kind: "show",
    pages: [...ONBOARDING_PAGES],
  });
});

test("an account that already chatted is adopted, not shown", () => {
  assert.deepEqual(decideOnboarding(record({ hasChatHistory: true })), {
    kind: "adopt",
  });
});

test("a run closed midway resumes at the first unseen page", () => {
  const decision = decideOnboarding(
    record({ saved: true, pages: ["welcome", "profile"] }),
  );
  assert.deepEqual(decision, {
    kind: "show",
    pages: ["look", "device", "start", "done"],
  });
});

test("a finished account sees nothing, and only pages added since", () => {
  const all = record({
    saved: true,
    pages: [...ONBOARDING_PAGES],
    completedAt: 1,
  });
  assert.deepEqual(decideOnboarding(all), { kind: "none" });
  const older = record({
    saved: true,
    pages: ["welcome", "profile", "look", "start", "done"],
    completedAt: 1,
  });
  assert.deepEqual(decideOnboarding(older), {
    kind: "show",
    pages: ["device", "done"],
  });
});

test("bookends alone are never shown", () => {
  const decision = decideOnboarding(
    record({ saved: true, pages: ["profile", "look", "device", "start"] }),
  );
  assert.deepEqual(decision, { kind: "none" });
});

test("greetings use the first word of the name", () => {
  assert.equal(greetingName("  Ada   Lovelace "), "Ada");
  assert.equal(greetingName("   "), "");
});

test("every start tour id is a registered tour route", () => {
  const routes = readFileSync(
    new URL("../src/features/tour/lib/tour-routes.ts", import.meta.url),
    "utf8",
  );
  for (const id of Object.values(START_TOURS)) {
    assert.match(routes, new RegExp(`id: "${id}"`));
  }
});

test("page ids fit the backend's pattern", () => {
  for (const id of ONBOARDING_PAGES) assert.match(id, /^[a-z0-9-]{1,40}$/);
});
