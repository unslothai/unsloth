// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { openingTag } from "./helpers/tsx-ast.ts";

// No DOM renderer here and the frame pulls in React, so assert the wiring in source.
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

const SURFACE = "../src/features/browser/file-view.tsx";
const FRAME = "../src/features/chat/artifacts/html-frame.tsx";
const ALERT = "../src/components/ui/alert.tsx";

/** Every `<ArtifactHtmlFrame>` opening tag in the browser's file view, where chat HTML opens. */
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

// Fenced html blocks have source "fence", so gating on "tool" broke CDN imports in canvases.
test("no canvas preview is gated on the artifact source", () => {
  const tags = readFrameOpeningTags();
  assert.ok(
    tags.length > 0,
    "<ArtifactHtmlFrame> not found in the browser file view",
  );
  for (const tag of tags) {
    assert.doesNotMatch(
      tag,
      /\bsource\b/,
      "the preview frame must not discriminate on artifact.source",
    );
  }
});

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

// The gate is exactly these two operands; a third operand is how it gets defeated.
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

test("the blocked banner is hidden once network access is on", () => {
  const condition = readConst("showBlockedBanner");
  assert.match(condition, /!networkAllowed/);
  assert.match(condition, /blockedForCanvas\.uris\.length > 0/);
});

// Derive the banner during render: a [src] effect clears it one render too late.
test("the blocked banner is tied to the canvas that reported it", () => {
  assert.equal(
    readConst("blockedForCanvas"),
    "blocked.code === code ? blocked : NOTHING_BLOCKED",
  );
});

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

function readSetterCalls(
  setter: string,
): { argument: string; handler: string | null }[] {
  const source = sourceFile(FRAME);
  const calls: { argument: string; handler: string | null }[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node) && node.expression.getText() === setter) {
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

const readGrantCalls = () => readSetterCalls("setGrantedCode");

// The canvas reports blocks itself, so only a JSX click handler may grant network access.
test("only a click can grant the per-canvas exception", () => {
  const calls = readGrantCalls();
  assert.ok(calls.length > 0, "setGrantedCode is never called");
  for (const { argument, handler } of calls) {
    assert.equal(argument, "code", "the grant stores the current code");
    assert.equal(handler, "onClick", "the grant must come from a click handler");
  }
});

// A grant must be compared during render; an effect reset on [code] runs too late.
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

// event.source survives navigation, so reports are stamped with the load they came from.
test("blocked reports from a stale frame load are rejected", () => {
  const source = sourceFile(FRAME);
  let guarded = false;
  const visit = (node: ts.Node): void => {
    if (
      ts.isIfStatement(node) &&
      /event\.data\.v !== loadVersion/.test(node.expression.getText()) &&
      node.thenStatement.getText().includes("return")
    ) {
      guarded = true;
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(guarded, "reports are not matched against the current load");
});

// The canvas can post reports directly, so URI size must be bounded, not just entry count.
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

// Non-HTTP violations report bare tokens like "eval" or "blob" with no host.
test("a hostless blocked URI still reaches the banner", () => {
  assert.match(readFunctionBody("blockedHost"), /BLOCKED_KEYWORD\.test\(uri\)/);
  assert.equal(readConst("BLOCKED_KEYWORD"), "/^[a-z-]+$/");
});

// The permissive worker-src is `http: https: blob:`, so a data: Worker cannot be fixed by the grant.
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

// These stay 'none' in the permissive policy too; a backend test pins that set.
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

// The cap is checked before the duplicate scan so work past it is O(1).
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

test("only a click can dismiss the banner, and only for the code on screen", () => {
  const calls = readSetterCalls("setDismissedCode");
  assert.ok(calls.length > 0, "setDismissedCode is never called");
  for (const { argument, handler } of calls) {
    assert.equal(argument, "code", "the dismissal stores the current code");
    assert.equal(
      handler,
      "onClick",
      "the dismissal must come from a click handler",
    );
  }
});

test("the dismissal is tied to the code it was dismissed for", () => {
  assert.equal(readConst("dismissedForCanvas"), "dismissedCode === code");
  assert.match(readConst("showBlockedBanner"), /!dismissedForCanvas/);
});

test("the banner deep-links to the network access setting", () => {
  const source = sourceFile(FRAME);
  const deepLinks: { tab: string; options: string; handler: string | null }[] =
    [];
  const visit = (node: ts.Node): void => {
    if (
      ts.isCallExpression(node) &&
      node.expression.getText().endsWith(".openDialog")
    ) {
      let handler: string | null = null;
      for (let at: ts.Node = node; at.parent; at = at.parent) {
        if (ts.isJsxAttribute(at.parent)) {
          handler = at.parent.name.getText();
          break;
        }
      }
      deepLinks.push({
        tab: node.arguments[0]?.getText() ?? "",
        options: node.arguments[1]?.getText() ?? "",
        handler,
      });
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.equal(
    deepLinks.length,
    1,
    "the banner opens the settings dialog once",
  );
  assert.equal(deepLinks[0].tab, '"browser"');
  assert.match(deepLinks[0].options, /scrollTarget:\s*"browser-html-network"/);
  assert.equal(deepLinks[0].handler, "onClick");
});

function readTagWithRef(name: string): string {
  const source = sourceFile(FRAME);
  let text: string | undefined;
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (opening?.getText().includes(`ref={${name}}`)) {
      text = opening.getText();
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(text, `no element carries ref={${name}}`);
  return text;
}

function readHandlerCalling(needle: string): string {
  const source = sourceFile(FRAME);
  let text: string | undefined;
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node) && node.expression.getText() === needle) {
      for (let at: ts.Node = node; at.parent; at = at.parent) {
        if (ts.isJsxAttribute(at.parent)) {
          text = at.parent.getText();
          break;
        }
      }
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(text, `no JSX handler calls ${needle}`);
  return text;
}

test("dismissing the banner leaves focus on the canvas", () => {
  const handler = readHandlerCalling("setDismissedCode");
  const focusAt = handler.indexOf("focusAfterAction");
  const dismissAt = handler.indexOf("setDismissedCode");
  assert.ok(focusAt >= 0, "the dismissal must hand focus to the canvas");
  assert.ok(focusAt < dismissAt, "focus must move before the button unmounts");
});

test("granting network access leaves focus on the canvas", () => {
  const handler = readHandlerCalling("setGrantedCode");
  const focusAt = handler.indexOf("focusAfterAction");
  const grantAt = handler.indexOf("setGrantedCode");
  assert.ok(focusAt >= 0, "the grant must hand focus to the canvas");
  assert.ok(focusAt < grantAt, "focus must move before the button unmounts");
});

test("the settings deep link keeps the invoking button as its opener", () => {
  const handler = readHandlerCalling(
    "useSettingsDialogStore.getState().openDialog",
  );
  assert.match(
    handler,
    /focusFallback:\s*actionFocusTargetRef\?\.current \?\? iframeRef\.current/,
  );
  assert.doesNotMatch(handler, /iframeRef\.current\?\.focus/);
});

test("page actions return focus to the frame unless a caller names a target", () => {
  const tags = readFrameOpeningTags();
  assert.equal(tags.length, 1);
  assert.doesNotMatch(tags[0], /actionFocusTargetRef/);
  assert.match(
    readConst("focusAfterAction"),
    /\(actionFocusTargetRef\?\.current \?\? iframeRef\.current\)\?\.focus/,
  );
});

test("the named iframe is the visible focus fallback", () => {
  const tag = readTagWithRef("iframeRef");
  assert.match(tag, /title=\{title\}/);
  assert.match(
    tag,
    /focus-visible:outline/,
    "the restored focus target needs a visible focus treatment",
  );
});

test("the shared alert uses logical alignment and action spacing", () => {
  const source = readFileSync(
    fileURLToPath(new URL(ALERT, import.meta.url)),
    "utf8",
  );
  assert.match(source, /text-start/);
  assert.match(source, /has-data-\[slot=alert-action\]:pe-18/);
  assert.match(source, /absolute top-2\.5 end-3/);
  assert.doesNotMatch(source, /\btext-left\b|\bpr-18\b|\bright-3\b/);
});

test("the blocked alert scopes direction to its locale", () => {
  const source = sourceFile(FRAME);
  let alert: string | undefined;
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (opening?.tagName.getText() === "Alert") alert = opening.getText();
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(alert, "blocked alert not found");
  assert.match(alert, /dir=\{locale === "ar" \|\| locale === "he" \? "rtl" : "ltr"\}/);
});

test("the climbing blocked count stays outside the assertive live region", () => {
  const source = sourceFile(FRAME);
  let alert: string | undefined;
  let alertTitle: string | undefined;
  let paragraph: string | undefined;
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (opening?.tagName.getText() === "Alert") {
      alert = opening.getText();
    }
    if (opening?.tagName.getText() === "AlertTitle") {
      alertTitle = opening.getText();
    }
    if (
      ts.isJsxElement(node) &&
      node.openingElement.tagName.getText() === "p" &&
      node.getText().includes("blockedBanner")
    ) {
      paragraph = node.getText();
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(alert, "the blocked alert is missing");
  assert.ok(alertTitle, "the blocked alert title is missing");
  assert.ok(paragraph, "the paragraph carrying the blocked count is missing");
  assert.match(alert, /role="group"/);
  assert.match(alertTitle, /role="alert"/);
  assert.doesNotMatch(paragraph, /aria-live/);
});

function readButtonCalling(needle: string): string {
  const source = sourceFile(FRAME);
  let text: string | undefined;
  const visit = (node: ts.Node): void => {
    const opening = openingTag(node);
    if (
      opening?.tagName.getText() === "Button" &&
      node.getText().includes(needle)
    ) {
      text = opening.getText();
    }
    node.forEachChild(visit);
  };
  source.forEachChild(visit);
  assert.ok(text, `no <Button> calls ${needle}`);
  return text;
}

// Like Studio's tool controls, the narrow per-canvas grant is the primary action.
test("the emphasized action is the per-canvas grant, not the global setting", () => {
  const grant = readButtonCalling("setGrantedCode");
  const settings = readButtonCalling("openDialog");
  assert.doesNotMatch(
    grant,
    /variant=/,
    "the per-canvas grant must stay the default (primary) button",
  );
  assert.match(
    settings,
    /variant="outline"/,
    "the global setting must be the quieter outline button",
  );
});
