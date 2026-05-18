// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { beforeEach, describe, expect, it } from "vitest";

import {
  AUTH_MUST_CHANGE_PASSWORD_KEY,
  AUTH_REFRESH_TOKEN_KEY,
  AUTH_TOKEN_KEY,
  ONBOARDING_DONE_KEY,
  clearAuthTokens,
  getAuthToken,
  getRefreshToken,
  hasAuthToken,
  hasRefreshToken,
  isOnboardingDone,
  markOnboardingDone,
  mustChangePassword,
  resetOnboardingDone,
  setMustChangePassword,
  storeAuthTokens,
} from "@/features/auth/session";

beforeEach(() => {
  localStorage.clear();
});

describe("storeAuthTokens", () => {
  it("writes exactly the access + refresh keys, never the must-change-password key", () => {
    storeAuthTokens("access-x", "refresh-y");
    expect(localStorage.getItem(AUTH_TOKEN_KEY)).toBe("access-x");
    expect(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY)).toBe("refresh-y");
    // The must-change-password flag stays out of this function (CodeQL data-flow pin).
    expect(localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY)).toBeNull();
  });
});

describe("clearAuthTokens", () => {
  it("removes the access, refresh, and must-change keys", () => {
    localStorage.setItem(AUTH_TOKEN_KEY, "a");
    localStorage.setItem(AUTH_REFRESH_TOKEN_KEY, "b");
    localStorage.setItem(AUTH_MUST_CHANGE_PASSWORD_KEY, "1");
    clearAuthTokens();
    expect(localStorage.getItem(AUTH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(AUTH_REFRESH_TOKEN_KEY)).toBeNull();
    expect(localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY)).toBeNull();
  });
});

describe("hasAuthToken / getAuthToken", () => {
  it("reads back what storeAuthTokens wrote", () => {
    expect(hasAuthToken()).toBe(false);
    expect(getAuthToken()).toBeNull();
    storeAuthTokens("x", "y");
    expect(hasAuthToken()).toBe(true);
    expect(getAuthToken()).toBe("x");
  });
});

describe("hasRefreshToken / getRefreshToken", () => {
  it("reads back what storeAuthTokens wrote", () => {
    expect(hasRefreshToken()).toBe(false);
    expect(getRefreshToken()).toBeNull();
    storeAuthTokens("x", "rt");
    expect(hasRefreshToken()).toBe(true);
    expect(getRefreshToken()).toBe("rt");
  });
});

describe("mustChangePassword / setMustChangePassword (key-presence encoding)", () => {
  it("starts false when nothing is stored", () => {
    expect(mustChangePassword()).toBe(false);
  });

  it("setMustChangePassword(true) writes the literal '1' and the read returns true", () => {
    setMustChangePassword(true);
    // The stored value is a literal constant, not a derived boolean.
    // The CodeQL data-flow break depends on this remaining a constant.
    expect(localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY)).toBe("1");
    expect(mustChangePassword()).toBe(true);
  });

  it("setMustChangePassword(false) removes the key and the read returns false", () => {
    setMustChangePassword(true);
    setMustChangePassword(false);
    expect(localStorage.getItem(AUTH_MUST_CHANGE_PASSWORD_KEY)).toBeNull();
    expect(mustChangePassword()).toBe(false);
  });

  it("treats any present value as truthy (presence is the signal)", () => {
    // Defensive: even if some other code wrote a different value, presence wins.
    localStorage.setItem(AUTH_MUST_CHANGE_PASSWORD_KEY, "true");
    expect(mustChangePassword()).toBe(true);
    localStorage.setItem(AUTH_MUST_CHANGE_PASSWORD_KEY, "");
    // Empty string is still a present key.
    expect(mustChangePassword()).toBe(true);
  });
});

describe("onboarding flag", () => {
  it("starts unset and round-trips through mark/reset", () => {
    expect(isOnboardingDone()).toBe(false);
    markOnboardingDone();
    expect(localStorage.getItem(ONBOARDING_DONE_KEY)).toBe("true");
    expect(isOnboardingDone()).toBe(true);
    resetOnboardingDone();
    expect(localStorage.getItem(ONBOARDING_DONE_KEY)).toBeNull();
    expect(isOnboardingDone()).toBe(false);
  });

  it("only treats the literal 'true' as done (regression pin: not just truthy)", () => {
    localStorage.setItem(ONBOARDING_DONE_KEY, "1");
    expect(isOnboardingDone()).toBe(false);
    localStorage.setItem(ONBOARDING_DONE_KEY, "TRUE");
    expect(isOnboardingDone()).toBe(false);
  });
});
