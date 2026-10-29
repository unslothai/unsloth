// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// ChatPage loads on first use so other screens start without it; a start on Chat preloads it.
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
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
    return /\.(ts|tsx)$/.test(name) && !/\.(test|check)\.tsx?$/.test(name) ? [path] : [];
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
  assert.match(readSrc("features/chat/index.ts"), /validateChatSearch[^;]*from "\.\/chat-search"/);
  assert.match(readSrc("features/chat/load-chat-page.ts"), /import\("\.\/chat-page"\)/);
});

test("the vite build registers the chat preload", () => {
  const config = readFileSync(new URL("../vite.config.ts", import.meta.url), "utf8");
  assert.match(config, /chatPagePreload\(\)/);
});

function preloadsFor(path: string, storage: Record<string, string>): string[] {
  const appended: string[] = [];
  const context = {
    location: { pathname: path },
    localStorage: { getItem: (key: string) => storage[key] ?? null },
    document: {
      createElement: () => ({}) as { rel?: string; href?: string; crossOrigin?: string },
      head: {
        appendChild: (link: { rel?: string; href?: string; crossOrigin?: string }) => {
          assert.equal(link.rel, "modulepreload");
          assert.equal(link.crossOrigin, "");
          appended.push(link.href as string);
        },
      },
    },
  };
  vm.runInNewContext(chatPreloadScript(["/assets/a.js", "/assets/b.js"]), context);
  return appended;
}

test("the preload runs only where the app lands on Chat", () => {
  const files = ["/assets/a.js", "/assets/b.js"];
  const signedIn = { unsloth_auth_token: "t" };
  assert.deepEqual(preloadsFor("/chat", {}), files);
  assert.deepEqual(preloadsFor("/", signedIn), files);
  // "/" without a session goes to login, and a pending password change goes there first.
  assert.deepEqual(preloadsFor("/", {}), []);
  assert.deepEqual(preloadsFor("/", { ...signedIn, unsloth_auth_must_change_password: "1" }), []);
  for (const path of ["/login", "/hub", "/studio", "/chatty", "/chat/x"]) {
    assert.deepEqual(preloadsFor(path, signedIn), [], path);
  }
});
