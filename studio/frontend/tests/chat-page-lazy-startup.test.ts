// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// ChatPage loads on first use so other screens start without it; a start on Chat preloads it.
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

import { chatPreloadScript } from "../vite-plugin-chat-preload.ts";
import { readSrc } from "./helpers/kit.ts";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sources(path);
    return /\.(ts|tsx)$/.test(name) && !/\.(test|check)\.tsx?$/.test(name)
      ? [path]
      : [];
  });
}

test("nothing imports the chat page statically", () => {
  const staticImport =
    /^\s*(import|export)\s+(?!type\b)[^;]*?\bfrom\s+["'](?:\.\/chat-page|@\/features\/chat\/chat-page)["']/m;
  const importers = sources(SRC)
    .filter((path) => staticImport.test(readFileSync(path, "utf8")))
    .map((path) => path.slice(SRC.length));
  assert.deepEqual(importers, []);
});

test("the chat route validates its search without loading the chat page", () => {
  assert.match(
    readSrc("features/chat/index.ts"),
    /validateChatSearch[^;]*from "\.\/chat-search"/,
  );
  assert.match(
    readSrc("features/chat/load-chat-page.ts"),
    /import\("\.\/chat-page"\)/,
  );
});

test("the vite build registers the chat preload", () => {
  const config = readFileSync(
    new URL("../vite.config.ts", import.meta.url),
    "utf8",
  );
  assert.match(config, /chatPagePreload\(\)/);
});

function preloadsFor(
  path: string,
  storage: Record<string, string>,
  desktop = false,
): string[] {
  const appended: string[] = [];
  const window: Record<string, unknown> = desktop
    ? { __TAURI_INTERNALS__: {} }
    : {};
  const context = {
    window,
    location: { pathname: path, protocol: desktop ? "tauri:" : "http:" },
    localStorage: { getItem: (key: string) => storage[key] ?? null },
    document: {
      createElement: () =>
        ({}) as { rel?: string; href?: string; crossOrigin?: string },
      head: {
        appendChild: (link: {
          rel?: string;
          href?: string;
          crossOrigin?: string;
        }) => {
          assert.equal(link.rel, "modulepreload");
          assert.equal(link.crossOrigin, "");
          appended.push(link.href as string);
        },
      },
    },
  };
  vm.runInNewContext(
    chatPreloadScript(["/assets/a.js", "/assets/b.js"]),
    context,
  );
  return appended;
}

test("the preload runs only where the app lands on Chat", () => {
  const files = ["/assets/a.js", "/assets/b.js"];
  const signedIn = { unsloth_auth_token: "t" };
  assert.deepEqual(preloadsFor("/chat", {}), files);
  assert.deepEqual(preloadsFor("/", signedIn), files);
  // "/" without a session goes to login, and a pending password change goes there first.
  assert.deepEqual(preloadsFor("/", {}), []);
  assert.deepEqual(
    preloadsFor("/", { ...signedIn, unsloth_auth_must_change_password: "1" }),
    [],
  );
  for (const path of ["/login", "/hub", "/studio", "/chatty", "/chat/x"]) {
    assert.deepEqual(preloadsFor(path, signedIn), [], path);
  }
});

test("the preload follows main.tsx for the single-user Desktop app", () => {
  const files = ["/assets/a.js", "/assets/b.js"];
  const pending = {
    unsloth_auth_token: "t",
    unsloth_auth_must_change_password: "1",
  };
  // Single-user Desktop lands on Chat even with a password change pending; multi-user does not.
  assert.deepEqual(preloadsFor("/", pending, true), files);
  assert.deepEqual(
    preloadsFor(
      "/",
      { ...pending, "unsloth.auth-login-mode.v1": "multi" },
      true,
    ),
    [],
  );
  assert.deepEqual(preloadsFor("/", { unsloth_auth_token: "t" }, true), files);
  assert.deepEqual(preloadsFor("/", {}, true), []);
  assert.deepEqual(
    preloadsFor("/login", { unsloth_auth_token: "t" }, true),
    [],
  );
});

test("a failed chat page load stays inside the chat area and retries", () => {
  const root = readSrc("app/routes/__root.tsx");
  // Each attempt gets its own lazy(), since lazy() keeps a rejection for good.
  assert.match(root, /function lazyChatPage\(\) \{\s*return lazy\(/);
  assert.match(root, /<LazyImportBoundary\s+key=\{page\.attempt\}/);
  assert.match(
    root,
    /<Suspense fallback=\{<RouteFallback \/>\}>\s*<Page \{\.\.\.props\} \/>/,
  );
  const loader = readSrc("features/chat/load-chat-page.ts");
  // A rejected import is dropped so the next caller fetches again.
  assert.match(loader, /chatPageModule = null;\s*throw error;/);
});
