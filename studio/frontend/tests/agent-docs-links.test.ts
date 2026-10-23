// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  SUPPORTED_AGENTS,
  UNSLOTH_START_DOCS_URL,
} from "../src/features/settings/components/coding-agent-list.ts";
import { readSrc } from "./helpers/kit.ts";

test("Pi and Mistral Vibe docs open the unsloth start page", () => {
  for (const id of ["pi", "vibe"]) {
    const agent = SUPPORTED_AGENTS.find((a) => a.id === id);
    assert.equal(agent?.docsUrl, UNSLOTH_START_DOCS_URL, id);
  }
});

test("the Setup docs row has no Pi or Mistral Vibe chip", () => {
  const src = readSrc("features/settings/components/usage-examples.tsx");
  const row = /const DOC_LINKS = \[[\s\S]*?\];/.exec(src)?.[0];
  assert.ok(row, "DOC_LINKS not found");
  assert.doesNotMatch(row, /pi\.dev|mistral/i);
});
