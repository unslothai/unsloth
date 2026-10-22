// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { createServer } from "vite";
import { installLocalStorageFake } from "./helpers/kit.ts";

installLocalStorageFake();
Object.assign(window.location, { href: "http://localhost/" });
const vite = await createServer({ server: { middlewareMode: true } });
after(() => vite.close());

test("a full endpoint pasted as the base URL is trimmed back to the base", async () => {
  const { stripEndpointPath } = await vite.ssrLoadModule(
    "/src/features/chat/provider-base-url.ts",
  );
  for (const [input, expected] of [
    ["/v1/chat/completions", "/v1"],
    ["/v1/chat/completions/", "/v1"],
    ["/v1/completions", "/v1"],
    ["/v1/responses", "/v1"],
    ["/v1/models", "/v1"],
    ["/v1", "/v1"],
    ["/v1/", "/v1"],
    ["/openai/v1", "/openai/v1"],
    ["/", ""],
  ]) {
    assert.equal(stripEndpointPath(input), expected, input);
  }
});
