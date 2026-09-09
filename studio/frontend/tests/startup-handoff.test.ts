// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const source = readSrc("app/provider.tsx");

test("desktop splash waits for scoped route readiness, not global events or backend auth", () => {
  assert.match(source, /<AppReadinessBoundary onReady=\{setAppShellReady\} revealed=\{showApp\}>/);
  assert.doesNotMatch(source, /addEventListener\("unsloth:app-shell-ready"/);
  assert.match(source, /const showApp = canMountApp && appShellReady/);
  assert.match(source, /\{canMountApp && \(/);
  assert.match(source, /inert=\{!showApp\}/);
  assert.match(source, /\{!showApp && \(/);
  assert.match(source, /<StartupScreen/);
});

test("missing readiness has a bounded escape and backend restarts reset the handoff", () => {
  assert.match(source, /if \(!canMountApp\) \{\s*setAppShellReady\(false\)/);
  assert.match(source, /window\.setTimeout\(\(\) => setAppShellReady\(true\), 15_000\)/);
  assert.match(source, /window\.clearTimeout\(timeout\)/);
});

test("handoff keeps app geometry and identity while deferring native intents", () => {
  assert.match(source, /style=\{\{ visibility: showApp \? "visible" : "hidden" \}\}/);
  assert.match(source, /\{showApp && <NativeIntentDrain \/>\}/);
  // The startup layer stays below the draggable/titlebar controls (z-50/z-70).
  assert.match(source, /fixed inset-0 z-40 bg-background/);
  const root = readSrc("app/routes/__root.tsx");
  assert.match(root, /<CredentialBootstrapGate active=\{!isAuthFlowRoute\}>/);
  assert.match(root, /\{active && !ready \? <RouteFallback \/> : children\}/);
});
