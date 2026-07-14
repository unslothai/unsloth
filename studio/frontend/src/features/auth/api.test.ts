// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ fetch: vi.fn() }));
vi.mock("@/lib/api-base", () => ({ apiUrl: (path: string) => path, isTauri: true }));
vi.mock("./session", () => ({
  clearAuthTokens: vi.fn(),
  getAuthToken: () => "access",
  getRefreshToken: () => null,
  mustChangePassword: () => false,
  setMustChangePassword: vi.fn(),
  storeAuthTokens: vi.fn(),
}));

import { authFetch } from "./api";

describe("authFetch network retry safety", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    mocks.fetch.mockReset();
    vi.stubGlobal("fetch", mocks.fetch);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("never blindly replays a non-idempotent mutation after a lost response", async () => {
    mocks.fetch.mockRejectedValue(new TypeError("lost response"));
    await expect(authFetch("/mutation", { method: "POST" })).rejects.toThrow(
      "relaunch",
    );
    expect(mocks.fetch).toHaveBeenCalledTimes(1);
  });

  it("retains bounded Tauri retry for safe reads", async () => {
    mocks.fetch
      .mockRejectedValueOnce(new TypeError("temporary"))
      .mockResolvedValueOnce(new Response(null, { status: 200 }));
    const request = authFetch("/read");
    await vi.advanceTimersByTimeAsync(250);
    await expect(request).resolves.toMatchObject({ status: 200 });
    expect(mocks.fetch).toHaveBeenCalledTimes(2);
  });
});
