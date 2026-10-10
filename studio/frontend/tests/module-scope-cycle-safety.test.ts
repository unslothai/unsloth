// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Keys live in an import-free module, so module-scope reads cannot hit the TDZ in a cycle. */

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

const SRC = fileURLToPath(new URL("../src", import.meta.url));
const KEYS = path.join(SRC, "features/chat/stores/sidebar-organization-keys.ts");
const GENERAL_TAB = path.join(SRC, "features/settings/tabs/general-tab.tsx");
const CHAT_RUNTIME = path.join(
  SRC,
  "features/chat/stores/chat-runtime-store.ts",
);
const PRESET_LOAD_CONFIG = path.join(
  SRC,
  "features/chat/presets/preset-load-config.ts",
);
const APPLY_PER_MODEL_CONFIG = path.join(
  SRC,
  "features/model-picker/model-config/apply-per-model-config.ts",
);
const TOOL_GROUP = path.join(SRC, "components/assistant-ui/tool-group.tsx");

const parse = (file: string, text: string) =>
  ts.createSourceFile(
    file,
    text,
    ts.ScriptTarget.ESNext,
    true,
    file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );

const staticSpecifiers = (file: string, text: string): string[] => {
  const specifiers: string[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
      const specifier = node.moduleSpecifier;
      if (specifier && ts.isStringLiteral(specifier)) {
        specifiers.push(specifier.text);
      }
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(parse(file, text), visit);
  return specifiers;
};

test("the sidebar organization keys module imports nothing", async () => {
  // An import here puts it back in the cycle and the white screen comes back.
  const text = await readFile(KEYS, "utf8");
  assert.deepEqual(
    staticSpecifiers(KEYS, text),
    [],
    "sidebar-organization-keys.ts must not import anything",
  );
  assert.match(text, /export const SIDEBAR_ORGANIZATION_STORAGE_KEY/);
});

test("general-tab reads the key from the keys module, not the store or the barrel", async () => {
  const text = await readFile(GENERAL_TAB, "utf8");
  const source = parse(GENERAL_TAB, text);

  let specifier: string | null = null;
  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) && node.importClause?.namedBindings) {
      const bindings = node.importClause.namedBindings;
      if (
        ts.isNamedImports(bindings) &&
        bindings.elements.some(
          (element) => element.name.text === "SIDEBAR_ORGANIZATION_STORAGE_KEY",
        ) &&
        ts.isStringLiteral(node.moduleSpecifier)
      ) {
        specifier = node.moduleSpecifier.text;
      }
    }
    ts.forEachChild(node, visit);
  };
  ts.forEachChild(source, visit);

  assert.equal(
    specifier,
    "@/features/chat/stores/sidebar-organization-keys",
    "general-tab must not reach the key through the store or the chat barrel; " +
      "both are in an import cycle with this file",
  );
});

test("the key is still read at module scope, so the guard above is load-bearing", async () => {
  const text = await readFile(GENERAL_TAB, "utf8");
  const source = parse(GENERAL_TAB, text);

  let readAtModuleScope = false;
  for (const statement of source.statements) {
    if (!ts.isVariableStatement(statement)) {
      continue;
    }
    const visit = (node: ts.Node): void => {
      if (
        ts.isIdentifier(node) &&
        node.text === "SIDEBAR_ORGANIZATION_STORAGE_KEY"
      ) {
        readAtModuleScope = true;
      }
      ts.forEachChild(node, visit);
    };
    ts.forEachChild(statement, visit);
  }

  assert.ok(
    readAtModuleScope,
    "general-tab no longer reads the key at module scope; drop this file's guards",
  );
});

/**
 * Dev's unbundled ESM evaluated use-model-memory before the chat store (an import cycle),
 * so module-scope key reads threw; production bundling hides it.
 */
const MODEL_MEMORY = path.join(SRC, "hooks/use-model-memory.ts");
const CHAT_RUNTIME_KEYS = path.join(
  SRC,
  "features/chat/stores/chat-runtime-keys.ts",
);

test("the chat runtime keys module imports nothing", async () => {
  const text = await readFile(CHAT_RUNTIME_KEYS, "utf8");
  assert.deepEqual(
    staticSpecifiers(CHAT_RUNTIME_KEYS, text),
    [],
    "chat-runtime-keys.ts must not import anything",
  );
});

test("the model memory hook reads runtime keys from the leaf module", async () => {
  const text = await readFile(MODEL_MEMORY, "utf8");
  assert.match(
    text,
    /CHAT_GPU_MEMORY_MODE_KEY,[\s\S]*CHAT_SPECULATIVE_TYPE_KEY,[\s\S]*from "@\/features\/chat\/stores\/chat-runtime-keys"/,
  );
});

test("the model memory hook imports no feature barrel", async () => {
  const text = await readFile(MODEL_MEMORY, "utf8");
  const barrels = staticSpecifiers(MODEL_MEMORY, text).filter((specifier) =>
    /^@\/features\/[^/]+$/.test(specifier),
  );

  assert.deepEqual(
    barrels,
    [],
    "a bare @/features/<name> import here closes the cycle and the dev server white screens again",
  );
});

test("the chat runtime imports the Hub token store directly", async () => {
  const text = await readFile(CHAT_RUNTIME, "utf8");
  const specifiers = staticSpecifiers(CHAT_RUNTIME, text);
  assert.ok(
    specifiers.includes("@/features/hub/stores/hf-token-store"),
    "chat-runtime-store.ts must import the token store directly",
  );
  assert.ok(
    !specifiers.includes("@/features/hub"),
    "the Hub barrel closes a module-scope cycle through the model picker",
  );
});

test("the startup cycle imports no feature barrel", async () => {
  for (const file of [
    PRESET_LOAD_CONFIG,
    APPLY_PER_MODEL_CONFIG,
    TOOL_GROUP,
  ]) {
    const text = await readFile(file, "utf8");
    const barrels = staticSpecifiers(file, text).filter((specifier) =>
      /^@\/features\/[^/]+$/.test(specifier),
    );
    assert.deepEqual(
      barrels,
      [],
      `${path.relative(SRC, file)} closes the chat-store and model-picker cycle`,
    );
  }
});
