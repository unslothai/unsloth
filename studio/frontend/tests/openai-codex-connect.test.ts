// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer, type ViteDevServer } from "vite";

interface CodexOAuthFlow {
  flow_id: string;
  method: "browser" | "device";
  status: "pending" | "connected" | "error" | "cancelled";
  expires_at: number;
  authorization_url?: string | null;
  verification_url?: string | null;
  user_code?: string | null;
  message?: string | null;
}

interface ConnectProps {
  providerId: string | null;
  authStatus?: "disconnected" | "connected" | "reauthorization_required";
  onChanged: () => void;
  ensureProvider?: () => Promise<string>;
  initialFlow?: CodexOAuthFlow | null;
}

type ConnectComponent = (props: ConnectProps) => ReturnType<typeof createElement>;

let vite: ViteDevServer;
let OpenAICodexConnect: ConnectComponent;

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  const loaded = await vite.ssrLoadModule(
    "/src/features/chat/openai-codex-connect.tsx",
  );
  OpenAICodexConnect = loaded.OpenAICodexConnect as ConnectComponent;
});

after(async () => {
  await vite.close();
});

function render(props: Partial<ConnectProps> = {}): string {
  return renderToStaticMarkup(createElement(OpenAICodexConnect, {
    providerId: "provider-1",
    onChanged: () => undefined,
    ...props,
  }));
}

function flow(overrides: Partial<CodexOAuthFlow>): CodexOAuthFlow {
  return {
    flow_id: "flow-1",
    method: "browser",
    status: "pending",
    expires_at: 4_000_000_000,
    ...overrides,
  };
}

test("renders login options before the provider has been saved", () => {
  const markup = render({ providerId: null, ensureProvider: async () => "provider-1" });
  assert.match(markup, /Connect in browser/);
  assert.match(markup, /Use device code/);
});


test("renders browser callback completion and cancellation controls", () => {
  const markup = render({ initialFlow: flow({ method: "browser" }) });
  assert.match(markup, /paste the complete localhost callback URL/i);
  assert.match(markup, /http:\/\/localhost:1455\/auth\/callback/);
  assert.match(markup, />Complete</);
  assert.match(markup, />Cancel</);
});

test("renders device-code instructions without exposing OAuth internals", () => {
  const markup = render({
    initialFlow: flow({ method: "device", user_code: "ABCD-EFGH" }),
  });
  assert.match(markup, /Enter this code in ChatGPT/);
  assert.match(markup, /ABCD-EFGH/);
  assert.match(markup, />Copy code</);
  assert.doesNotMatch(markup, /code_verifier|device_auth_id|secret-state/);
});


test("renders an expired authorization error as an alert", () => {
  const markup = render({
    initialFlow: flow({
      status: "error",
      message: "Authorization expired. Start a new connection.",
    }),
  });
  assert.match(markup, /role="alert"/);
  assert.match(markup, /Authorization expired/);
  assert.match(markup, /Connect in browser/);
});

test("renders cancelled, reconnect-required, and connected states", () => {
  assert.match(
    render({ initialFlow: flow({ status: "cancelled" }) }),
    /Authorization cancelled/,
  );
  assert.match(
    render({ authStatus: "reauthorization_required" }),
    /Reconnect in browser/,
  );
  const connected = render({ authStatus: "connected" });
  assert.match(connected, /Connected securely/);
  assert.match(connected, /Disconnect locally/);
  assert.doesNotMatch(connected, /Connect in browser/);
});
