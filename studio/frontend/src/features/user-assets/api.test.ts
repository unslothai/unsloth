// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ authFetch: vi.fn() }));
vi.mock("@/features/auth", () => ({ authFetch: mocks.authFetch }));

import {
  importLegacyUserAssets,
  listServerRecipes,
  upsertServerRecipeExecution,
} from "./api";

describe("user-assets request safety", () => {
  beforeEach(() => {
    mocks.authFetch.mockReset();
  });

  it("rejects route-unsafe ids locally", () => {
    expect(() =>
      upsertServerRecipeExecution({
        recipeId: "recipe/one",
        executionId: "run-one",
        metadata: {},
      }),
    ).toThrow("one URL path segment");
    expect(mocks.authFetch).not.toHaveBeenCalled();
  });

  it("times out a half-open request and permits retry", async () => {
    vi.useFakeTimers();
    try {
      mocks.authFetch
        .mockImplementationOnce(
          (_url: string, init?: RequestInit) =>
            new Promise<Response>((_resolve, reject) => {
              init?.signal?.addEventListener("abort", () =>
                reject(init.signal?.reason),
              );
            }),
        )
        .mockResolvedValueOnce(
          new Response('{"recipes":[]}', {
            headers: { "Content-Type": "application/json" },
          }),
        );
      const first = expect(listServerRecipes()).rejects.toThrow("timed out");
      await vi.advanceTimersByTimeAsync(15_000);
      await first;
      await expect(listServerRecipes()).resolves.toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("composes caller cancellation into mutations", async () => {
    mocks.authFetch.mockImplementation(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
        }),
    );
    const controller = new AbortController();
    const request = importLegacyUserAssets(
      {
        source: "recipe-indexeddb-v1",
        confirmSubject: "alice",
        recipes: [],
        executions: [],
      },
      { signal: controller.signal },
    );
    controller.abort(new DOMException("cancelled", "AbortError"));
    await expect(request).rejects.toMatchObject({ name: "AbortError" });
  });
});
