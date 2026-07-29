// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/// <reference types="vite/client" />

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register(new URL("./module-alias-loader.mjs", import.meta.url));

class MemoryStorage {
  readonly #values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.#values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.#values.set(key, String(value));
  }

  removeItem(key: string): void {
    this.#values.delete(key);
  }
}

function token(subject: string): string {
  const payload = Buffer.from(JSON.stringify({ sub: subject })).toString(
    "base64url",
  );
  return `x.${payload}.x`;
}

test("guarded recipe reads stop when the authenticated subject changes", async () => {
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: new MemoryStorage(),
  });
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      addEventListener() {},
      dispatchEvent() {},
      location: { href: "/", pathname: "/", protocol: "http:" },
      removeEventListener() {},
    },
  });
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { onLine: true, platform: "linux", userAgent: "node" },
  });

  const { AuthSubjectChangedError } = await import(
    "../src/features/auth/api.ts"
  );
  const { getAuthSubjectKey, storeAuthTokens } = await import(
    "../src/features/auth/session.ts"
  );
  const { bootstrapUserAssets, listServerRecipeExecutions } = await import(
    "../src/features/user-assets/api.ts"
  );
  const { getRecipeJobStatus } = await import(
    "../src/features/recipe-studio/api/index.ts"
  );

  const accounts = {
    A: { access: token("A"), refresh: "refresh-A" },
    B: { access: token("B"), refresh: "refresh-B" },
  } as const;
  const select = (account: keyof typeof accounts): void => {
    storeAuthTokens(accounts[account].access, accounts[account].refresh);
  };

  const expectSubjectGuard = async (
    call: () => Promise<unknown>,
    label: string,
  ): Promise<void> => {
    select("A");
    let calls = 0;
    globalThis.fetch = (async () => {
      calls += 1;
      let switched = false;
      return {
        clone() {
          return this;
        },
        get status() {
          if (!switched) {
            switched = true;
            queueMicrotask(() => select("B"));
          }
          return 401;
        },
        statusText: "Unauthorized",
        async json() {
          return {};
        },
        async text() {
          return "";
        },
      } as unknown as Response;
    }) as typeof fetch;

    await assert.rejects(call, AuthSubjectChangedError, label);
    assert.equal(getAuthSubjectKey(), "subject:B");
    assert.equal(calls, 1, `${label} must stop before refresh or retry`);
  };

  await expectSubjectGuard(
    () => listServerRecipeExecutions("recipe-A"),
    "user-asset execution read",
  );
  await expectSubjectGuard(
    () => getRecipeJobStatus("job-A"),
    "recipe job status read",
  );

  select("A");
  const bootstrapUrls: string[] = [];
  const bootstrapPages = [
    {
      subject: "subject:A",
      importLedger: {
        source: "recipe-indexeddb-v1",
        recipes: ["r1"],
        executions: ["e1"],
        nextCursor: "page-2",
      },
    },
    {
      subject: "subject:A",
      importLedger: {
        source: "recipe-indexeddb-v1",
        recipes: ["r2"],
        executions: ["e2"],
        nextCursor: null,
      },
    },
  ];
  globalThis.fetch = (async (input) => {
    bootstrapUrls.push(String(input));
    return Response.json(bootstrapPages.shift());
  }) as typeof fetch;

  const bootstrap = await bootstrapUserAssets();
  assert.deepEqual(bootstrap.importLedger.recipes, ["r1", "r2"]);
  assert.deepEqual(bootstrap.importLedger.executions, ["e1", "e2"]);
  assert.deepEqual(bootstrapUrls, [
    "/api/user-assets/bootstrap",
    "/api/user-assets/bootstrap?cursor=page-2",
  ]);
});
