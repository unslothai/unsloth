// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A saved Hugging Face token the Hub refuses (an expired OAuth token, a revoked key) gets 401 on
 * every read, public ones included, and the browser used to report that as "Couldn't reach
 * Hugging Face". A token the Hub accepts gets 404 for what it cannot see, so a tokened 401 that
 * succeeds anonymously is the token being refused: fetchHub answers with the anonymous result
 * and records the refusal once per token.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);

const { resetHfEndpoints, setHfEndpoints, setHubSessionRefresh } = await import(
  "../src/lib/hf-endpoint.ts"
);
const { fetchHub, hubRejectionScope } = await import("../src/lib/hub-fetch.ts");
const {
  HF_TOKEN_REJECTION_RECHECK_MS,
  clearHfTokenRejected,
  hasRejectedHfToken,
  hfTokenRejectionVersion,
  isHfTokenRejected,
  noteHfTokenRejected,
  subscribeHfTokenRejected,
} = await import("../src/lib/hf-token-rejection.ts");
const { isCompleteHfTokenShape } = await import("../src/lib/hf-token-shape.ts");

const OAUTH = `hf_oauth_${"A1b2C3d4E5".repeat(4)}`;
const CLASSIC = `hf_${"a".repeat(34)}`;
const API = "https://huggingface.co/api/models?search=qwen";

type Sent = { url: string; authorization: string | null; hfAuthorization: string | null };

/** A Hub that refuses *rejectedToken* with the headers the real one sends, and answers
 * anonymous reads with *anonymousStatus*. */
function stubHub(opts: {
  rejectedToken?: string;
  anonymousStatus?: number;
  tokenStatus?: number;
  relay?: string;
}) {
  const sent: Sent[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | Request, init?: RequestInit) => {
    // As fetch does: init.headers, when given, replace a Request's own.
    const isRequest = input instanceof Request;
    const headers = new Headers(init?.headers ?? (isRequest ? input.headers : undefined));
    const url = isRequest ? input.url : String(input);
    const authorization = headers.get("authorization");
    const hfAuthorization = headers.get("x-hf-authorization");
    sent.push({ url, authorization, hfAuthorization });
    const viaRelay = opts.relay !== undefined && url.startsWith(opts.relay);
    const hubToken = viaRelay ? hfAuthorization : authorization;
    const upstream: Record<string, string> = viaRelay ? { "X-Hub-Upstream": "1" } : {};
    if (hubToken) {
      if (opts.tokenStatus !== undefined) {
        return new Response("{}", { status: opts.tokenStatus, headers: upstream });
      }
      if (opts.rejectedToken && hubToken === `Bearer ${opts.rejectedToken}`) {
        return new Response('{"error":"Invalid credentials in Authorization header"}', {
          status: 401,
          headers: {
            ...upstream,
            "X-Error-Message": "OAuth token verification failed: Invalid Compact JWS",
            "WWW-Authenticate": 'Bearer realm="Authentication required"',
          },
        });
      }
      return new Response('[{"id":"private/model"}]', { status: 200, headers: upstream });
    }
    const status = opts.anonymousStatus ?? 200;
    return new Response(status === 200 ? '[{"id":"unsloth/Qwen3-0.6B-GGUF"}]' : "{}", {
      status,
      headers: upstream,
    });
  }) as typeof fetch;
  return {
    sent,
    restore: () => {
      globalThis.fetch = realFetch;
    },
  };
}

function withToken(token: string): RequestInit {
  return { headers: { Authorization: `Bearer ${token}` } };
}

test.beforeEach(() => {
  resetHfEndpoints();
  clearHfTokenRejected();
});

test("OAuth tokens are validated like classic tokens instead of never being checked", () => {
  assert.equal(isCompleteHfTokenShape(OAUTH), true);
  assert.equal(isCompleteHfTokenShape(CLASSIC), true);
  // Still not while typing: a prefix or a partial body is not worth a Hub request.
  assert.equal(isCompleteHfTokenShape("hf_oauth_abc"), false);
  assert.equal(isCompleteHfTokenShape("hf_abc"), false);
  assert.equal(isCompleteHfTokenShape(`hf_${"a".repeat(33)}`), false);
  const hook = readFileSync(
    new URL("../src/hooks/use-hf-token-validation.ts", import.meta.url),
    "utf8",
  );
  assert.match(hook, /isCompleteHfTokenShape\(debouncedToken\)/);
  assert.match(hook, /isCompleteHfTokenShape\(normalizedToken\)/);
});

test("a refused token is retried once without it and the public answer is returned", async () => {
  const hub = stubHub({ rejectedToken: OAUTH });
  let notified = 0;
  const unsubscribe = subscribeHfTokenRejected(() => {
    notified += 1;
  });
  try {
    const response = await fetchHub(API, withToken(OAUTH));
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), [{ id: "unsloth/Qwen3-0.6B-GGUF" }]);
    assert.deepEqual(
      hub.sent.map((s) => s.authorization),
      [`Bearer ${OAUTH}`, null],
    );
    assert.equal(isHfTokenRejected(OAUTH), true);
    // Later reads skip the refused token instead of paying a 401 round trip each.
    const again = await fetchHub(API, withToken(OAUTH));
    assert.equal(again.status, 200);
    assert.deepEqual(
      hub.sent.map((s) => s.authorization),
      [`Bearer ${OAUTH}`, null, null],
    );
    assert.equal(notified, 1, "one notice per token, not per request");
  } finally {
    unsubscribe();
    hub.restore();
  }
});

test("when anonymous access fails too, the original 401 is returned and nothing is recorded", async () => {
  const hub = stubHub({ rejectedToken: OAUTH, anonymousStatus: 401 });
  try {
    const response = await fetchHub(API, withToken(OAUTH));
    assert.equal(response.status, 401);
    assert.equal(
      response.headers.get("X-Error-Message"),
      "OAuth token verification failed: Invalid Compact JWS",
    );
    assert.equal(hub.sent.length, 2);
    assert.equal(isHfTokenRejected(OAUTH), false);
  } finally {
    hub.restore();
  }
});

test("403, 404, 429 and 5xx answer for the resource, not the token, and are never retried", async () => {
  for (const status of [403, 404, 429, 500, 503]) {
    const hub = stubHub({ tokenStatus: status });
    try {
      const response = await fetchHub(API, withToken(CLASSIC));
      assert.equal(response.status, status);
      assert.equal(hub.sent.length, 1, `status ${status}`);
      assert.equal(isHfTokenRejected(CLASSIC), false);
    } finally {
      hub.restore();
    }
  }
});

test("a network failure propagates without a retry", async () => {
  const realFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = (async () => {
    calls += 1;
    throw new TypeError("Failed to fetch");
  }) as typeof fetch;
  try {
    await assert.rejects(fetchHub(API, withToken(OAUTH)), TypeError);
    assert.equal(calls, 1);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("an anonymous 401 is not retried and flags nothing", async () => {
  const hub = stubHub({ anonymousStatus: 401 });
  try {
    const response = await fetchHub(API, {});
    assert.equal(response.status, 401);
    assert.equal(hub.sent.length, 1);
  } finally {
    hub.restore();
  }
});

test("a write is never replayed without the token", async () => {
  const hub = stubHub({ rejectedToken: OAUTH });
  try {
    const response = await fetchHub("https://huggingface.co/api/repos/create", {
      ...withToken(OAUTH),
      method: "POST",
      body: "{}",
    });
    assert.equal(response.status, 401);
    assert.equal(hub.sent.length, 1);
    assert.equal(isHfTokenRejected(OAUTH), false);
  } finally {
    hub.restore();
  }
});

test("through the Studio relay, only the endpoint's own 401 retries, and the session stays", async () => {
  const relay = "http://127.0.0.1:8888/api/hub/proxy";
  const hub = stubHub({ rejectedToken: OAUTH, relay });
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: { getItem: () => "session" },
  });
  setHubSessionRefresh(async () => false);
  try {
    setHfEndpoints(relay, null, "huggingface", { endpoint: true });
    const response = await fetchHub(`${relay}/api/models?search=qwen`, withToken(OAUTH));
    assert.equal(response.status, 200);
    assert.deepEqual(
      hub.sent.map((s) => [s.authorization, s.hfAuthorization]),
      [
        ["Bearer session", `Bearer ${OAUTH}`],
        ["Bearer session", null],
      ],
    );
    assert.equal(isHfTokenRejected(OAUTH), true);
  } finally {
    hub.restore();
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});

test("a relay 401 without the upstream marker is the session, not the token", async () => {
  const relay = "http://127.0.0.1:8888/api/hub/proxy";
  const realFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = (async () => {
    calls += 1;
    return new Response("{}", { status: 401 });
  }) as typeof fetch;
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: { getItem: () => "session" },
  });
  setHubSessionRefresh(async () => false);
  try {
    setHfEndpoints(relay, null, "huggingface", { endpoint: true });
    const response = await fetchHub(`${relay}/api/models`, withToken(OAUTH));
    assert.equal(response.status, 401);
    assert.equal(calls, 1);
    assert.equal(isHfTokenRejected(OAUTH), false);
  } finally {
    globalThis.fetch = realFetch;
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});

test("a different token starts clean", () => {
  const before = hfTokenRejectionVersion();
  assert.equal(noteHfTokenRejected(OAUTH), true);
  assert.equal(noteHfTokenRejected(OAUTH), false);
  assert.equal(isHfTokenRejected(OAUTH), true);
  assert.equal(isHfTokenRejected(CLASSIC), false);
  assert.equal(isHfTokenRejected(null), false);
  assert.ok(hfTokenRejectionVersion() > before);
  clearHfTokenRejected();
  assert.equal(isHfTokenRejected(OAUTH), false);
});

test("OAuth tokens are redacted from notifications and diagnostics", async () => {
  for (const file of ["../src/lib/native-notifications.ts", "../src/lib/tauri-diagnostics.ts"]) {
    const source = readFileSync(new URL(file, import.meta.url), "utf8");
    const patterns = [...source.matchAll(/\/(\\bhf_[^/]+)\/g?/g)].map((m) => new RegExp(m[1]));
    assert.ok(patterns.length > 0, file);
    for (const pattern of patterns) {
      const match = `token ${OAUTH} end`.match(pattern);
      assert.equal(match?.[0], OAUTH, `${file}: ${pattern} must cover the whole OAuth token`);
      assert.equal(`token ${CLASSIC} end`.match(pattern)?.[0], CLASSIC);
    }
  }
});

test("after a refusal, a read anonymous access cannot answer still tries the token once", async () => {
  noteHfTokenRejected(OAUTH, hubRejectionScope());
  // The verifier recovered: the token works again and the private repo answers with it.
  const hub = stubHub({ anonymousStatus: 404 });
  try {
    const response = await fetchHub("https://huggingface.co/api/models/me/private", withToken(OAUTH));
    assert.equal(response.status, 200);
    assert.deepEqual(
      hub.sent.map((s) => s.authorization),
      [null, `Bearer ${OAUTH}`],
    );
    assert.equal(isHfTokenRejected(OAUTH), false);
  } finally {
    hub.restore();
  }
});

test("after a refusal, a token that is still refused leaves the anonymous answer and the flag", async () => {
  noteHfTokenRejected(OAUTH, hubRejectionScope());
  const hub = stubHub({ rejectedToken: OAUTH, anonymousStatus: 404 });
  try {
    const response = await fetchHub("https://huggingface.co/api/models/me/private", withToken(OAUTH));
    assert.equal(response.status, 404);
    assert.equal(hub.sent.length, 2);
    assert.equal(isHfTokenRejected(OAUTH), true);
  } finally {
    hub.restore();
  }
});


test("a refusal recorded against one Hub endpoint does not skip the token on another", async () => {
  noteHfTokenRejected(OAUTH, "huggingface|https://mirror.example");
  const hub = stubHub({});
  try {
    const response = await fetchHub("https://huggingface.co/api/models/me/private", withToken(OAUTH));
    assert.equal(response.status, 200);
    assert.deepEqual(
      hub.sent.map((s) => s.authorization),
      [`Bearer ${OAUTH}`],
    );
  } finally {
    hub.restore();
    clearHfTokenRejected();
  }
});

test("a refusal by the datasets server does not skip the token on the model Hub", async () => {
  const datasetsScope = hubRejectionScope("https://datasets-server.huggingface.co/size?dataset=x");
  assert.notEqual(datasetsScope, hubRejectionScope("https://huggingface.co/api/models"));
  assert.equal(hubRejectionScope("https://huggingface.co/api/models"), hubRejectionScope());
  noteHfTokenRejected(OAUTH, datasetsScope);
  const hub = stubHub({});
  try {
    const response = await fetchHub("https://huggingface.co/api/models/me/private", withToken(OAUTH));
    assert.equal(response.status, 200);
    assert.deepEqual(
      hub.sent.map((s) => s.authorization),
      [`Bearer ${OAUTH}`],
    );
  } finally {
    hub.restore();
    clearHfTokenRejected();
  }
});

test("a token carried by a Request input is retried and recorded like an init header", async () => {
  const hub = stubHub({ rejectedToken: OAUTH });
  try {
    const request = new Request(API, { headers: { Authorization: `Bearer ${OAUTH}`, "X-Probe": "1" } });
    const response = await fetchHub(request);
    assert.equal(response.status, 200);
    assert.deepEqual(
      hub.sent.map((s) => s.authorization),
      [`Bearer ${OAUTH}`, null],
    );
    assert.equal(isHfTokenRejected(OAUTH, hubRejectionScope(API)), true);
  } finally {
    hub.restore();
    clearHfTokenRejected();
  }
});

test("through the relay, a Request's own token is dropped from the anonymous retry", async () => {
  const relay = "http://127.0.0.1:8888/api/hub/proxy";
  const hub = stubHub({ rejectedToken: OAUTH, relay });
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: { getItem: () => "session" },
  });
  setHubSessionRefresh(async () => false);
  try {
    setHfEndpoints(relay, null, "huggingface", { endpoint: true });
    const request = new Request(`${relay}/api/models?search=qwen`, {
      headers: { Authorization: `Bearer ${OAUTH}` },
    });
    const response = await fetchHub(request);
    assert.equal(response.status, 200);
    assert.deepEqual(
      hub.sent.map((s) => [s.authorization, s.hfAuthorization]),
      [
        ["Bearer session", `Bearer ${OAUTH}`],
        ["Bearer session", null],
      ],
    );
    assert.equal(isHfTokenRejected(OAUTH, hubRejectionScope(request.url)), true);
  } finally {
    hub.restore();
    resetHfEndpoints();
    clearHfTokenRejected();
    Reflect.deleteProperty(globalThis, "localStorage");
  }
});

test("a cached valid verdict does not clear a refusal a read just saw", () => {
  const source = readFileSync(new URL("../src/features/hf-auth/api.ts", import.meta.url), "utf8");
  assert.equal(source.includes("clearHfTokenRejected("), false);
});

test("refusals by two Hubs are both kept", () => {
  const models = hubRejectionScope("https://huggingface.co/api/models");
  const datasets = hubRejectionScope("https://datasets-server.huggingface.co/size?dataset=x");
  try {
    noteHfTokenRejected(OAUTH, models);
    noteHfTokenRejected(OAUTH, datasets);
    assert.equal(isHfTokenRejected(OAUTH, models), true);
    assert.equal(isHfTokenRejected(OAUTH, datasets), true);
    clearHfTokenRejected(datasets);
    assert.equal(isHfTokenRejected(OAUTH, models), true);
    assert.equal(isHfTokenRejected(OAUTH, datasets), false);
  } finally {
    clearHfTokenRejected();
  }
});

test("a datasets server under the model endpoint gets its own scope", () => {
  setHfEndpoints("https://mirror.example", "https://mirror.example/datasets-server", "huggingface");
  try {
    assert.notEqual(
      hubRejectionScope("https://mirror.example/datasets-server/size?dataset=x"),
      hubRejectionScope("https://mirror.example/api/models"),
    );
    assert.equal(
      hubRejectionScope("https://mirror.example/datasets-server/size?dataset=x"),
      "huggingface|https://mirror.example/datasets-server",
    );
  } finally {
    resetHfEndpoints();
  }
});

test("after the recheck window the token is tried again, and an answer clears the refusal", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: 1_000_000 });
  const scope = hubRejectionScope(API);
  noteHfTokenRejected(OAUTH, scope);
  const hub = stubHub({});
  try {
    await fetchHub(API, withToken(OAUTH));
    assert.equal(hub.sent[0].authorization, null, "skipped while the refusal is fresh");
    t.mock.timers.tick(HF_TOKEN_REJECTION_RECHECK_MS + 1);
    await fetchHub(API, withToken(OAUTH));
    assert.equal(hub.sent[1].authorization, `Bearer ${OAUTH}`);
    assert.equal(hasRejectedHfToken(), false);
  } finally {
    hub.restore();
    clearHfTokenRejected();
  }
});

test("a token still refused after the recheck is recorded again without a new notification", async (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: 1_000_000 });
  const scope = hubRejectionScope(API);
  noteHfTokenRejected(OAUTH, scope);
  const before = hfTokenRejectionVersion();
  const hub = stubHub({ rejectedToken: OAUTH });
  try {
    t.mock.timers.tick(HF_TOKEN_REJECTION_RECHECK_MS + 1);
    const response = await fetchHub(API, withToken(OAUTH));
    assert.equal(response.status, 200);
    assert.equal(isHfTokenRejected(OAUTH, scope), true);
    assert.equal(hfTokenRejectionVersion(), before);
  } finally {
    hub.restore();
    clearHfTokenRejected();
  }
});

test("clearing a refusal is not reported as a new one", () => {
  noteHfTokenRejected(OAUTH, hubRejectionScope());
  assert.equal(hasRejectedHfToken(), true);
  clearHfTokenRejected();
  assert.equal(hasRejectedHfToken(), false);
});

test("a failed anonymous probe keeps the Hub's 401 instead of throwing", async () => {
  const realFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = (async () => {
    calls += 1;
    if (calls === 1) {
      return new Response("{}", {
        status: 401,
        headers: { "X-Error-Message": "OAuth token verification failed: Invalid Compact JWS" },
      });
    }
    throw new TypeError("Failed to fetch");
  }) as typeof fetch;
  try {
    const response = await fetchHub("https://huggingface.co/api/models", withToken(OAUTH));
    assert.equal(response.status, 401);
    assert.equal(isHfTokenRejected(OAUTH), false);
  } finally {
    globalThis.fetch = realFetch;
  }
});
