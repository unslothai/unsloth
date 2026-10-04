// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type McpOAuthFormPayload = {
  readonly useOauth: boolean;
  readonly oauthClientId?: string | null;
  readonly oauthClientSecret?: string;
};

export function buildMcpOAuthFormPayload(
  useOauth: boolean,
  clientId: string,
  clientSecret: string,
): McpOAuthFormPayload {
  if (!useOauth) {
    return { useOauth: false };
  }
  return {
    useOauth: true,
    oauthClientId: clientId.trim() || null,
    ...(clientSecret ? { oauthClientSecret: clientSecret } : {}),
  };
}

/** Which registered client, at which address, the stored secret belongs to. */
export type McpOAuthSecretOwner = {
  readonly url: string;
  readonly clientId: string;
};

export const MCP_OAUTH_SECRET_PLACEHOLDER_NEW = "Optional client secret";
export const MCP_OAUTH_SECRET_PLACEHOLDER_KEPT =
  "Leave blank to keep the stored secret";
export const MCP_OAUTH_SECRET_PLACEHOLDER_CLEARED =
  "Re-enter the secret: changing the client ID or the address clears the stored one";

export function mcpOAuthSecretPlaceholder(
  storedSecretOwner: McpOAuthSecretOwner | null,
  url: string,
  clientId: string,
): string {
  if (!storedSecretOwner) {
    return MCP_OAUTH_SECRET_PLACEHOLDER_NEW;
  }
  // The update route drops the secret when the client ID or URL changes without a replacement.
  const keepsStoredSecret =
    url.trim() === storedSecretOwner.url.trim() &&
    clientId.trim() === storedSecretOwner.clientId.trim();
  return keepsStoredSecret
    ? MCP_OAUTH_SECRET_PLACEHOLDER_KEPT
    : MCP_OAUTH_SECRET_PLACEHOLDER_CLEARED;
}
