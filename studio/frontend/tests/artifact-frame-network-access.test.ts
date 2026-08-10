// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { openingTag } from "./helpers/tsx-ast.ts";

// No DOM renderer here and the frame pulls in React plus the runtime store, so
// assert the wiring in the source the way artifact-source-key.test.ts does.
const sourceFile = (relative: string): ts.SourceFile => {
  const path = fileURLToPath(new URL(relative, import.meta.url));
  return ts.createSourceFile(
    path,
    readFileSync(path, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
    ts.ScriptKind.TSX,
  );
};

const SURFACE = "../src/features/chat/artifacts/artifact-surface.tsx";
const FRAME = "../src/features/chat/artifacts/html-frame.tsx";

/** Every `<ArtifactHtmlFrame>` opening tag in the artifact surface. */
function readFrameOpeningTags(): string[] {
  const source = sourceFile(SURFACE);
  const tags: string[] = [];
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (opening?.tagName.getText() === "ArtifactHtmlFrame") {
      tags.push(opening.getText());
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return tags;
}

// The bug: a fenced html block is source "fence", so gating on "tool" left every
// fenced canvas on the strict CSP and a CDN import (three.js) silently died.
// Every call site is checked, so a source-gated one cannot hide behind another.
test("no canvas preview is gated on the artifact source", () => {
  const tags = readFrameOpeningTags();
  assert.ok(
    tags.length > 0,
    "<ArtifactHtmlFrame> not found in the artifact surface",
  );
  for (const tag of tags) {
    assert.doesNotMatch(
      tag,
      /\bsource\b/,
      "the preview frame must not discriminate on artifact.source",
    );
  }
});

/** Every condition guarding an `allow_network` query flag in the frame. */
function readAllowNetworkGuards(): string[] {
  const source = sourceFile(FRAME);
  const conditions: string[] = [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isIfStatement(node) &&
      node.thenStatement.getText().includes("allow_network")
    ) {
      conditions.push(node.expression.getText());
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return conditions;
}

// These two operands are the whole gate: the persistent setting, or a grant the
// user clicked for this canvas. Asserting the condition exactly is the point,
// since a third operand is how the gate gets defeated.
test("the permissive CSP is gated on the setting or a per-canvas grant", () => {
  const conditions = readAllowNetworkGuards();
  assert.equal(conditions.length, 1, "expected exactly one allow_network guard");
  assert.equal(conditions[0], "networkAllowed");
});

test("the gate is exactly the setting or the per-canvas grant", () => {
  assert.equal(
    readConst("networkAllowed"),
    "networkAccessEnabled || grantedForCanvas",
  );
});

/** Initializer of a `const` declared in the frame, by name. */
function readConst(name: string): string {
  const source = sourceFile(FRAME);
  let text: string | undefined;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText() === name &&
      node.initializer
    ) {
      text = node.initializer.getText();
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(text, `${name} not found in the frame`);
  return text;
}

// The banner's only call to action is the grant, so one that survives it prompts
// for something already on. The blocked list stays in the condition too, or
// every offline canvas gets a banner.
test("the blocked banner is hidden once network access is on", () => {
  const condition = readConst("showBlockedBanner");
  assert.match(condition, /!networkAllowed/);
  assert.match(condition, /blockedForCanvas\.uris\.length > 0/);
});

// The button grants network to the CURRENT code, so a banner left over from a
// swapped-out canvas is a grant for one that reported nothing. Derived during
// render: the [src] effect that used to clear it ran a render too late, and
// that stale render is the one carrying the button.
test("the blocked banner is tied to the canvas that reported it", () => {
  assert.equal(
    readConst("blockedForCanvas"),
    "blocked.code === code ? blocked : NOTHING_BLOCKED",
  );
});

// The old clear was a wholesale `setBlocked({...})` in the [src] effect, a
// render too late. Every write goes through an updater that carries the code
// forward, so no reset can reintroduce the stale-banner window.
test("no write resets the blocked list wholesale", () => {
  const source = sourceFile(FRAME);
  const args: ts.Node[] = [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isCallExpression(node) &&
      node.expression.getText() === "setBlocked" &&
      node.arguments[0]
    ) {
      args.push(node.arguments[0]);
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(args.length > 0, "setBlocked is never called");
  for (const argument of args) {
    assert.ok(
      ts.isArrowFunction(argument) || ts.isFunctionExpression(argument),
      "setBlocked must take an updater, not a replacement object",
    );
  }
});

const GRANT_SETTER = /\bsetGranted\w*\(/;

/** Arguments of every `setGrantedCode(...)` call, with the enclosing JSX handler. */
function readGrantCalls(): { argument: string; handler: string | null }[] {
  const source = sourceFile(FRAME);
  const calls: { argument: string; handler: string | null }[] = [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isCallExpression(node) &&
      node.expression.getText() === "setGrantedCode"
    ) {
      let handler: string | null = null;
      for (let at: ts.Node = node; at.parent; at = at.parent) {
        if (ts.isJsxAttribute(at.parent)) {
          handler = at.parent.name.getText();
          break;
        }
      }
      calls.push({ argument: node.arguments[0]?.getText() ?? "", handler });
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  return calls;
}

// The canvas is what reports being blocked, so the grant must never be reachable
// from that path or a page could post its way onto the network. Only a JSX click
// handler may grant, and it stores the code it was clicked for.
test("only a click can grant the per-canvas exception", () => {
  const calls = readGrantCalls();
  assert.ok(calls.length > 0, "setGrantedCode is never called");
  for (const { argument, handler } of calls) {
    assert.equal(argument, "code", "the grant stores the current code");
    assert.equal(handler, "onClick", "the grant must come from a click handler");
  }
});

// A new canvas is new untrusted code, so the grant must not carry over. The tie
// has to be compared during render: an effect resetting it on [code] runs after
// React updated the DOM, so canvas B's first render still had A's grant.
test("the per-canvas grant is tied to the code it was granted for", () => {
  assert.equal(readConst("grantedForCanvas"), "grantedCode === code");
});

test("no effect resets the grant, which would leave a stale render", () => {
  const source = sourceFile(FRAME);
  let resetInEffect = false;
  const visit = (node: ts.Node): void => {
    if (
      ts.isCallExpression(node) &&
      node.expression.getText() === "useEffect" &&
      GRANT_SETTER.test(node.arguments[0]?.getText() ?? "")
    ) {
      resetInEffect = true;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(!resetInEffect, "the grant must not be reset from an effect");
});

const DIRECTIVE_FILTER =
  /GRANT_CANNOT_FIX\.has\(event\.data\.effectiveDirective\)/;
const SCHEME_FILTER =
  /GRANT_CANNOT_FIX_SCHEME\[event\.data\.effectiveDirective\] === uri/;
const URI_LENGTH_GUARD = /uri\.length > BLOCKED_URI_MAX_CHARS/;
const URI_CAP = /\buris\.length >= BLOCKED_URIS_TRACKED/;
const URI_DUPLICATE = /\buris\.includes\(uri\)/;
const BAILS_OUT = /\bcurrent\b/;

/** Body of a function declared in the frame, by name. */
function readFunctionBody(name: string): string {
  const source = sourceFile(FRAME);
  let text: string | undefined;
  const visit = (node: ts.Node): void => {
    if (
      ts.isFunctionDeclaration(node) &&
      node.name?.getText() === name &&
      node.body
    ) {
      text = node.body.getText();
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  if (!text) throw new Error(`${name} not found in the frame`);
  return text;
}

// event.source survives the swap navigation and the handler closes over the NEW
// code, so an in-flight report from the outgoing canvas would be stored as the
// incoming one's. The frame stamps each report with the load it came from.
test("blocked reports from a stale frame load are rejected", () => {
  const source = sourceFile(FRAME);
  let guarded = false;
  const visit = (node: ts.Node): void => {
    if (
      ts.isIfStatement(node) &&
      /event\.data\.v !== codeVersion/.test(node.expression.getText()) &&
      node.thenStatement.getText().includes("return")
    ) {
      guarded = true;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(guarded, "reports are not matched against the current load");
});

// The entry cap bounds how many reports are kept, not how big each one is, and
// the canvas can post these directly rather than going through the CSP. Without
// this a handful parks megabytes of parent state; it bounds the host too, which
// is derived from the URI.
test("oversized blocked URIs are dropped before anything is stored", () => {
  const source = sourceFile(FRAME);
  let bounded = false;
  const visit = (node: ts.Node): void => {
    if (
      ts.isIfStatement(node) &&
      URI_LENGTH_GUARD.test(node.expression.getText()) &&
      node.thenStatement.getText().includes("return")
    ) {
      bounded = true;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(bounded, "an oversized blockedURI is not rejected");
});

// A non-HTTP(S) violation reports a bare token, not a URL: eval() reports
// "eval" and a blob Worker reports "blob" (verified in Chromium). new URL() has
// no host for either, so they were dropped and a canvas broken only by those
// stayed blank, even though the permissive CSP widens both.
test("a hostless blocked URI still reaches the banner", () => {
  assert.match(readFunctionBody("blockedHost"), /BLOCKED_KEYWORD\.test\(uri\)/);
  assert.equal(readConst("BLOCKED_KEYWORD"), "/^[a-z-]+$/");
});

// ...but only where the grant widens that scheme for that directive. The
// permissive worker-src is `http: https: blob:`, so a data: Worker reports under
// both policies (verified in Chromium) and the grant is a dead end: it widens
// the policy for nothing, then hides the banner because networkAllowed is true.
test("the grant is not offered for a scheme it cannot widen", () => {
  assert.equal(
    readConst("GRANT_CANNOT_FIX_SCHEME"),
    '{ "worker-src": "data" }',
  );
  const source = sourceFile(FRAME);
  let filtered = false;
  const visit = (node: ts.Node): void => {
    if (
      ts.isIfStatement(node) &&
      SCHEME_FILTER.test(node.expression.getText()) &&
      node.thenStatement.getText().includes("return")
    ) {
      filtered = true;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(filtered, "reports are not filtered by blocked scheme");
});

// These three stay at 'none' in the permissive policy, so the grant cannot fix
// them and prompting widens the policy for nothing. The backend counterpart
// asserts they are exactly what the two policies agree on, so the set cannot
// drift.
test("the grant is not offered for directives it cannot fix", () => {
  assert.equal(
    readConst("GRANT_CANNOT_FIX"),
    'new Set(["object-src", "base-uri", "form-action"])',
  );
  const source = sourceFile(FRAME);
  let filtered = false;
  const visit = (node: ts.Node): void => {
    if (
      ts.isIfStatement(node) &&
      DIRECTIVE_FILTER.test(node.expression.getText()) &&
      node.thenStatement.getText().includes("return")
    ) {
      filtered = true;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(filtered, "reports are not filtered by effective directive");
});

// The canvas picks the blocked URIs, so an uncapped list is memory growth and a
// parent re-render per message, both driven from inside the sandbox. The cap is
// checked BEFORE the duplicate scan so past it the work is O(1) too.
test("blocked-resource state is capped against the untrusted canvas", () => {
  const updater = readFunctionBody("appendBlocked");
  assert.match(updater, URI_CAP);
  assert.match(updater, URI_DUPLICATE);
  assert.ok(
    updater.search(URI_CAP) < updater.search(URI_DUPLICATE),
    "the cap must be checked before the duplicate scan",
  );
  assert.match(updater, BAILS_OUT);
});
