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
// Every call site is checked, so a source-gated one cannot hide behind an
// ungated one that happens to be visited later.
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

// The permissive CSP is opt-in, and these two operands are the whole gate:
// the persistent setting, or a grant the user clicked for this canvas. Asserting
// the condition exactly is the point, since a third operand is how the gate gets
// defeated (a source check smuggled back in, or anything the canvas controls).
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

// Enabling the setting is the banner's only call to action, so a banner that
// survives it prompts for something already on. The blocked list also has to
// stay in the condition, or every offline canvas gets the banner.
test("the blocked banner is hidden once network access is on", () => {
  const condition = readConst("showBlockedBanner");
  assert.match(condition, /!networkAllowed/);
  assert.match(condition, /blockedForCanvas\.uris\.length > 0/);
});

// Reports from the canvas that was swapped out must not prompt for the one now
// on screen: the banner's button grants network to the CURRENT code, so a stale
// banner is a grant for a canvas that never reported anything blocked. Derived
// during render, because the [src] effect that used to clear this ran a render
// too late and the stale render is the one carrying the button.
test("the blocked banner is tied to the canvas that reported it", () => {
  assert.equal(
    readConst("blockedForCanvas"),
    "blocked.code === code ? blocked : NOTHING_BLOCKED",
  );
});

// The old clear was `setBlocked({ uris: [], hosts: [] })` in the [src] effect,
// a render too late. Every write must go through an updater that carries the
// code forward, so no wholesale reset can reintroduce the stale-banner window.
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
// from that path: a page could otherwise post its way onto the network. Granting
// is allowed only from a JSX click handler, and it stores the code it was clicked
// for, so nothing the canvas controls can widen it.
test("only a click can grant the per-canvas exception", () => {
  const calls = readGrantCalls();
  assert.ok(calls.length > 0, "setGrantedCode is never called");
  for (const { argument, handler } of calls) {
    assert.equal(argument, "code", "the grant stores the current code");
    assert.equal(handler, "onClick", "the grant must come from a click handler");
  }
});

// A new canvas is new untrusted code, so a grant must not carry over to it. The
// tie has to be a comparison made during render: an effect that reset the grant
// on [code] would run only after React had already updated the DOM, so the first
// render carrying canvas B still built src with canvas A's allow_network=1.
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

const URI_CAP = /\buris\.length >= BLOCKED_URIS_TRACKED/;
const URI_DUPLICATE = /\buris\.includes\(uri\)/;
const BAILS_OUT = /\bcurrent\b/;

/** Body of the `appendBlocked` reducer the blocked-state updater delegates to. */
function readBlockedAppendUpdater(): string {
  const source = sourceFile(FRAME);
  let text: string | undefined;
  const visit = (node: ts.Node): void => {
    if (
      ts.isFunctionDeclaration(node) &&
      node.name?.getText() === "appendBlocked" &&
      node.body
    ) {
      text = node.body.getText();
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  if (!text) throw new Error("appendBlocked not found in the frame");
  return text;
}

// The canvas picks the blocked URIs, so an uncapped list is memory growth and a
// parent re-render per message, both driven from inside the sandbox. The cap is
// checked BEFORE the duplicate scan so that past the cap the work is O(1) too,
// not a rescan of every stored string per message.
test("blocked-resource state is capped against the untrusted canvas", () => {
  const updater = readBlockedAppendUpdater();
  assert.match(updater, URI_CAP);
  assert.match(updater, URI_DUPLICATE);
  assert.ok(
    updater.search(URI_CAP) < updater.search(URI_DUPLICATE),
    "the cap must be checked before the duplicate scan",
  );
  assert.match(updater, BAILS_OUT);
});
