// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { buildMcpOAuthFormPayload } from "../src/features/chat/utils/mcp-oauth-form.ts";

const CHAT_ROOT = new URL("../src/features/chat/", import.meta.url);

test("the MCP form collects pre-registered OAuth credentials", async () => {
  const source = await readFile(
    new URL("chat-mcp-servers-dialog.tsx", CHAT_ROOT),
    "utf8",
  );
  assert.match(source, /id="mcp-oauth-client-id"/);
  assert.match(source, /id="mcp-oauth-client-secret"/);
  assert.match(source, /type="password"/);
  assert.match(source, /Leave blank to keep the stored secret/);
});

test("the MCP API sends credentials but never models a returned secret", async () => {
  const source = await readFile(
    new URL("api/mcp-servers-api.ts", CHAT_ROOT),
    "utf8",
  );
  assert.match(source, /oauth_client_id/);
  assert.match(source, /oauth_client_secret/);
  assert.match(source, /has_oauth_client_secret/);
  assert.doesNotMatch(source, /oauth_client_secret:\s*string/);
});

test("OAuth form payloads omit credentials when OAuth is disabled", () => {
  assert.deepEqual(
    buildMcpOAuthFormPayload(
      false,
      "configured-client-id",
      "configured-client-secret",
    ),
    { useOauth: false },
  );
});

test("blank edit secrets preserve the stored secret without resending it", () => {
  assert.deepEqual(
    buildMcpOAuthFormPayload(true, " configured-client-id ", ""),
    {
      useOauth: true,
      oauthClientId: "configured-client-id",
    },
  );
});

test("new and rotated OAuth credentials are sent together", () => {
  assert.deepEqual(
    buildMcpOAuthFormPayload(
      true,
      "replacement-client-id",
      "replacement-secret",
    ),
    {
      useOauth: true,
      oauthClientId: "replacement-client-id",
      oauthClientSecret: "replacement-secret",
    },
  );
});

test("a blank OAuth client ID is represented as an explicit clear", () => {
  assert.deepEqual(buildMcpOAuthFormPayload(true, " ", ""), {
    useOauth: true,
    oauthClientId: null,
  });
});
