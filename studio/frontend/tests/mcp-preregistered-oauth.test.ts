// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  MCP_OAUTH_SECRET_PLACEHOLDER_CLEARED,
  MCP_OAUTH_SECRET_PLACEHOLDER_KEPT,
  MCP_OAUTH_SECRET_PLACEHOLDER_NEW,
  buildMcpOAuthFormPayload,
  mcpOAuthSecretPlaceholder,
} from "../src/features/chat/utils/mcp-oauth-form.ts";

const CHAT_ROOT = new URL("../src/features/chat/", import.meta.url);
// The hint must be derived from the stored secret's owner AND from the two
// fields whose edits clear it, never from a bare "a secret exists" flag that
// survives them. Asserting the arguments, not just the call, is what makes a
// mis-wired hint fail here: this suite has no DOM, so it cannot render the
// dialog and read the attribute back.
const DERIVES_SECRET_HINT =
  /placeholder=\{mcpOAuthSecretPlaceholder\(\s*form\.storedSecretOwner,\s*form\.url,\s*form\.oauthClientId,?\s*\)\}/;

test("the MCP form collects pre-registered OAuth credentials", async () => {
  const source = await readFile(
    new URL("chat-mcp-servers-dialog.tsx", CHAT_ROOT),
    "utf8",
  );
  assert.match(source, /id="mcp-oauth-client-id"/);
  assert.match(source, /id="mcp-oauth-client-secret"/);
  assert.match(source, /type="password"/);
  assert.match(source, DERIVES_SECRET_HINT);
  assert.doesNotMatch(source, /hasOauthClientSecret/);
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

test("an untouched OAuth server promises to keep its stored secret", () => {
  assert.equal(
    mcpOAuthSecretPlaceholder(
      { url: "https://example.com/mcp", clientId: "configured-client-id" },
      "https://example.com/mcp",
      "configured-client-id",
    ),
    MCP_OAUTH_SECRET_PLACEHOLDER_KEPT,
  );
});

test("a server with no stored secret asks for an optional one", () => {
  assert.equal(
    mcpOAuthSecretPlaceholder(null, "https://example.com/mcp", "any-client-id"),
    MCP_OAUTH_SECRET_PLACEHOLDER_NEW,
  );
});

test("editing the client ID warns that the stored secret will be dropped", () => {
  assert.equal(
    mcpOAuthSecretPlaceholder(
      { url: "https://example.com/mcp", clientId: "configured-client-id" },
      "https://example.com/mcp",
      "replacement-client-id",
    ),
    MCP_OAUTH_SECRET_PLACEHOLDER_CLEARED,
  );
});

test("editing the address warns that the stored secret will be dropped", () => {
  assert.equal(
    mcpOAuthSecretPlaceholder(
      { url: "https://example.com/mcp", clientId: "configured-client-id" },
      "https://other.example.com/mcp",
      "configured-client-id",
    ),
    MCP_OAUTH_SECRET_PLACEHOLDER_CLEARED,
  );
});

test("clearing the client ID warns that the stored secret will be dropped", () => {
  assert.equal(
    mcpOAuthSecretPlaceholder(
      { url: "https://example.com/mcp", clientId: "configured-client-id" },
      "https://example.com/mcp",
      "   ",
    ),
    MCP_OAUTH_SECRET_PLACEHOLDER_CLEARED,
  );
});

test("surrounding whitespace alone does not threaten the stored secret", () => {
  assert.equal(
    mcpOAuthSecretPlaceholder(
      { url: " https://example.com/mcp ", clientId: " configured-client-id " },
      "  https://example.com/mcp  ",
      "  configured-client-id  ",
    ),
    MCP_OAUTH_SECRET_PLACEHOLDER_KEPT,
  );
});
