import { describe, expect, it } from "vitest";

import {
  PLATFORM_AUTH_TOKEN_KEY,
  clearPlatformSession,
  clearPlatformSessionAndRedirectToLogin,
  getPlatformSessionToken,
  resetPlatformUnauthorizedRedirectForTests,
  storePlatformSessionToken,
  usePlatformSessionStore,
} from "../auth-session";

describe("PlatformSession", () => {
  it("persists exactly one opaque token and clears user state", () => {
    localStorage.clear();
    storePlatformSessionToken(" opaque-token ");
    usePlatformSessionStore.getState().setUser({
      active: true,
      avatar: null,
      colorScheme: "Bright",
      createdAt: 1_786_502_416_241,
      email: "user@example.test",
      id: "user-1",
      language: "English",
      loginChannel: "password",
      nickname: "User",
      superuser: false,
      timezone: "UTC",
      updatedAt: 1_786_502_416_398,
    });

    expect(getPlatformSessionToken()).toBe("opaque-token");
    expect(localStorage.length).toBe(1);
    expect(localStorage.key(0)).toBe(PLATFORM_AUTH_TOKEN_KEY);

    clearPlatformSession();
    expect(getPlatformSessionToken()).toBeNull();
    expect(usePlatformSessionStore.getState().user).toBeNull();
  });

  it("redirects a burst of 401 responses to login only once", () => {
    localStorage.clear();
    resetPlatformUnauthorizedRedirectForTests();
    window.history.replaceState(null, "", "/chat");
    let redirects = 0;
    const onPopState = () => {
      redirects += 1;
    };
    window.addEventListener("popstate", onPopState);
    storePlatformSessionToken("opaque-token");

    clearPlatformSessionAndRedirectToLogin();
    clearPlatformSessionAndRedirectToLogin();

    window.removeEventListener("popstate", onPopState);
    expect(window.location.pathname).toBe("/login");
    expect(redirects).toBe(1);
    expect(getPlatformSessionToken()).toBeNull();
  });
});
