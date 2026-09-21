// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { loadWithStubs } from "./helpers/module-stubs.ts";
import { openingTag } from "./helpers/tsx-ast.ts";

// Both modules come from the shipped source with only their edges cut, so the
// constant under test is the one the app ships rather than one spelled here.
const { OWNER_USERNAME } = loadWithStubs<{ OWNER_USERNAME: string }>(
  new URL("../src/features/auth/account-session.ts", import.meta.url),
  {
    react: { useEffect: () => {}, useSyncExternalStore: () => undefined },
    "./login-client": {
      ensureLoginMode: () => {},
      getFullAccessAllowed: () => true,
      getLoginMode: () => "single",
      subscribeLoginMode: () => () => {},
    },
    "./session": {
      AUTH_SESSION_CLEARED_EVENT: "cleared",
      AUTH_SESSION_STORED_EVENT: "stored",
      getAuthToken: () => null,
    },
  },
);

const { loginDisplayName } = loadWithStubs<{
  loginDisplayName: (sub: string | null) => string;
}>(new URL("../src/features/profile/hooks/use-effective-profile.ts", import.meta.url), {
  "@/features/auth": { getAuthToken: () => null, OWNER_USERNAME },
  "../utils/jwt-subject": { decodeJwtSubject: () => null },
  "../stores/user-profile-store": { useUserProfileStore: () => "" },
});

// The owner's login id is reserved rather than chosen, and it is the product name
// in lower case. Every surface that falls back to that id owes the user the same
// spelling: a sidebar reading "Unsloth" over a settings panel reading "unsloth"
// describes one account with two names.

test("the reserved owner id is the only subject whose spelling is mapped", () => {
  // Read from the shipped export, not a copy: a test pinned to its own literal
  // cannot notice the constant it is about drifting.
  assert.equal(loginDisplayName(OWNER_USERNAME), "Unsloth");
  for (const chosen of ["alice", "bob", "unsloth2", "Unsloth", "UNSLOTH", " unsloth "]) {
    assert.equal(loginDisplayName(chosen), chosen);
  }
  assert.equal(loginDisplayName(null), "");
  assert.equal(loginDisplayName(""), "");
});

const panelSource = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/profile/components/profile-personalization-panel.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);

test("the personalization panel spells its login fallback the way the sidebar does", () => {
  const file = ts.createSourceFile(
    "profile-personalization-panel.tsx",
    panelSource,
    ts.ScriptTarget.ES2022,
    true,
    ts.ScriptKind.TSX,
  );

  // Both fallbacks route through the shared mapper rather than the raw subject.
  const previewName = findInitializer(file, "previewName");
  assert.ok(previewName, "previewName is no longer declared; update this test");
  assert.match(previewName, /loginName/);
  assert.doesNotMatch(
    previewName,
    /sessionSub/,
    "the preview avatar fell back to the raw subject, so it spells the brand lower case",
  );

  const placeholders = jsxAttributeTexts(file, "placeholder");
  assert.ok(placeholders.length > 0, "no placeholder attribute found; update this test");
  for (const placeholder of placeholders) {
    assert.doesNotMatch(
      placeholder,
      /sessionSub/,
      "the display-name placeholder shows the raw subject, so it spells the brand lower case",
    );
  }

  // The draft the user is typing still outranks any fallback.
  assert.match(previewName, /draftName\.trim\(\)\s*\|\|/);
});

/** The initializer text of `const <name> = ...`, or null when it is gone. */
function findInitializer(file: ts.SourceFile, name: string): string | null {
  let found: string | null = null;
  const visit = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === name &&
      node.initializer
    ) {
      found = node.initializer.getText(file);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return found;
}

/** Every `<x <name>={...}>` expression in the file, as source text. */
function jsxAttributeTexts(file: ts.SourceFile, name: string): string[] {
  const texts: string[] = [];
  const visit = (node: ts.Node): void => {
    const tag = openingTag(node);
    if (tag) {
      for (const attribute of tag.attributes.properties) {
        if (
          ts.isJsxAttribute(attribute) &&
          attribute.name.getText(file) === name &&
          attribute.initializer
        ) {
          texts.push(attribute.initializer.getText(file));
        }
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return texts;
}
