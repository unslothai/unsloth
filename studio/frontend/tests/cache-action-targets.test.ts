// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

for (const [name, module, method] of [
  ["revealCachedModel", "chat/api/chat-api.ts", "POST"],
  ["getCachedModelPath", "chat/api/chat-api.ts", "GET"],
  ["fetchDeleteImpact", "hub/inventory/api.ts", "POST"],
] as const) {
  test(`${name} carries the selected cache copy in its HTTP request`, async () => {
    const source = readFileSync(
      new URL(`../src/features/${module}`, import.meta.url),
      "utf8",
    );
    const start = source.indexOf(`export async function ${name}(`);
    assert.ok(start >= 0);
    const declaration = source
      .slice(start, source.indexOf("\n}", start) + 2)
      .replace("export ", "");
    const calls: { url: string; init?: RequestInit }[] = [];
    const authFetch = async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return new Response("{}", { status: 200 });
    };
    const request = new Function(
      "authFetch",
      "parseJsonOrThrow",
      `${
        ts.transpileModule(declaration, {
          compilerOptions: { target: ts.ScriptTarget.ES2020 },
        }).outputText
      }; return ${name};`,
    )(authFetch, (response: Response) => response.json());
    const path = "C:\\models\\hub\\models--Org--Model";
    await request("Org/Model", "Q8_0", path);
    assert.equal(calls.length, 1);
    if (method === "GET") {
      const params = new URL(calls[0].url, "http://localhost").searchParams;
      assert.equal(params.get("cache_path"), path);
      assert.equal(params.get("variant"), "Q8_0");
    } else {
      assert.equal(calls[0].init?.method, "POST");
      assert.deepEqual(JSON.parse(calls[0].init?.body as string), {
        repo_id: "Org/Model",
        variant: "Q8_0",
        cache_path: path,
      });
    }
  });
}
