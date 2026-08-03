// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { providerSupportsStudioTools } from "../src/features/chat/provider-capabilities.ts";

test("Studio-managed external tools are enabled only for Ollama", () => {
  assert.equal(providerSupportsStudioTools("ollama"), true);
  assert.equal(providerSupportsStudioTools("OLLAMA"), true);
  assert.equal(providerSupportsStudioTools("openai"), false);
  assert.equal(providerSupportsStudioTools("custom"), false);
  assert.equal(providerSupportsStudioTools(null), false);
});
