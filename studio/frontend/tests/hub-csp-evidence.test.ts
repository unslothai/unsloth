// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

// The violation fires on the document, not the rejected promise, so network.ts
// installs a listener and correlates by origin. Stand one up before importing:
// installCspViolationListener bails out when there is no document.
const listeners: Record<string, (event: unknown) => void> = {};
(globalThis as { document?: unknown }).document = {
  addEventListener: (type: string, fn: (event: unknown) => void) => {
    listeners[type] = fn;
  },
  removeEventListener: (type: string) => {
    delete listeners[type];
  },
};

const { classifyFetchFailure, installCspViolationListener } = await import(
  "../src/features/hub/lib/network.ts"
);

const HF_ORIGIN = "https://huggingface.co";

function violate(blockedURI: string, directive = "connect-src") {
  listeners["securitypolicyviolation"]?.({
    effectiveDirective: directive,
    violatedDirective: directive,
    blockedURI,
  });
}

function opaque(origin = HF_ORIGIN) {
  return classifyFetchFailure(new TypeError("Failed to fetch"), origin, {
    startedAt: 0,
  });
}

test("a CSP violation names the cause instead of the opaque TypeError", async () => {
  installCspViolationListener();
  violate(`${HF_ORIGIN}/api/models?search=gemma`);
  const failure = opaque();
  assert.equal(failure.kind, "csp-blocked");
  assert.match(failure.effectiveDirective ?? "", /connect-src/);
  // The blocked URI carried the search terms; only the host survives.
  assert.ok(!failure.message.includes("gemma"));
});

test("concurrent failures under one policy are both named", async () => {
  installCspViolationListener();
  violate(`${HF_ORIGIN}/api/models?search=gemma`);
  // The picker starts the model and dataset listings together, so two fetches
  // fail under the same policy. Consuming the record left the second one
  // classified network-opaque, and it then overwrote the better diagnosis.
  assert.equal(opaque().kind, "csp-blocked");
  assert.equal(opaque().kind, "csp-blocked", "the evidence outlives one read");
});

test("only connect-src can explain a failed fetch", async () => {
  installCspViolationListener();
  // Its own origin: a record is kept for its TTL now, so the connect-src hit
  // above would still be live and would mask what this asserts.
  const other = "https://assets.example";
  violate(`${other}/logo.png`, "img-src");
  assert.equal(opaque(other).kind, "network-opaque");
});
