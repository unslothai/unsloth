// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type NetworkRequest = {
  id: string;
  sessionId: string;
  /** "website": may it be reached at all; "upload": may more than the budget be sent to it. */
  kind: "website" | "upload";
  host: string;
  port: number;
  tool: "python" | "terminal" | "web";
  sent: number;
  canAlways: boolean;
};

/** The question a `network_request` event carries, null once it is answered, undefined if malformed. */
export function parseNetworkRequest(raw: unknown): NetworkRequest | null | undefined {
  if (typeof raw !== "object" || raw === null) return undefined;
  const event = raw as Record<string, unknown>;
  if (typeof event.id !== "string" || !event.id) return undefined;
  if (event.done === true) return null;
  if (typeof event.host !== "string" || !event.host) return undefined;
  return {
    id: event.id,
    sessionId: typeof event.session_id === "string" ? event.session_id : "",
    kind: event.kind === "upload" ? "upload" : "website",
    host: event.host,
    port: typeof event.port === "number" ? event.port : 443,
    tool: event.tool === "terminal" || event.tool === "web" ? event.tool : "python",
    sent: typeof event.sent === "number" ? event.sent : 0,
    canAlways: event.can_always === true,
  };
}

export function userNamedHost(host: string, text: string): boolean {
  const bare = host.toLowerCase().replace(/^www\./, "");
  const escaped = bare.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // A whole name: not e.com inside example.com, and not example.com.evil.test.
  return new RegExp(`(^|[^a-z0-9.-])(www\\.)?${escaped}($|[^a-z0-9.-]|\\.(?![a-z0-9-]))`).test(
    text.toLowerCase(),
  );
}
