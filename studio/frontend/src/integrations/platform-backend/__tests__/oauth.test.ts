import { afterEach, describe, expect, it } from "vitest";

import {
  consumePlatformOAuthRedirect,
  getPlatformOAuthLoginUrl,
  takePlatformOAuthError,
} from "../auth-api";
import {
  getPlatformSessionToken,
  PLATFORM_AUTH_TOKEN_KEY,
} from "../auth-session";

describe("platform OAuth browser bridge", () => {
  afterEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    document.cookie = "ragflow_auth=; Path=/; Max-Age=0";
  });

  it("consumes only the callback cookie and never accepts a query token", () => {
    document.cookie = "ragflow_auth=opaque-cookie-token; Path=/; SameSite=Lax";
    expect(
      consumePlatformOAuthRedirect("https://app.test/?auth=user-id"),
    ).toEqual({ handled: true, status: "success" });
    expect(getPlatformSessionToken()).toBe("opaque-cookie-token");
    expect(document.cookie).not.toContain("ragflow_auth");

    localStorage.removeItem(PLATFORM_AUTH_TOKEN_KEY);
    expect(
      consumePlatformOAuthRedirect(
        "https://app.test/?auth=attacker-controlled-token",
      ),
    ).toMatchObject({
      handled: true,
      status: "error",
      error: "oauth_session_missing",
    });
    expect(getPlatformSessionToken()).toBeNull();
  });

  it.each([
    ["invalid_state", "invalid_state"],
    ["access_denied", "oauth_failed"],
    ["https://evil.test/", "oauth_failed"],
  ])("normalizes callback failure %s without following redirects", (value, expected) => {
    const origin = window.location.origin;
    const result = consumePlatformOAuthRedirect(
      `https://app.test/?error=${encodeURIComponent(value)}`,
    );
    expect(result).toEqual({ handled: true, status: "error", error: expected });
    expect(takePlatformOAuthError()).toBe(expected);
    expect(window.location.origin).toBe(origin);
  });

  it("validates provider channel names and ignores unrelated URLs", () => {
    expect(consumePlatformOAuthRedirect("https://app.test/chat")).toEqual({
      handled: false,
    });
    expect(getPlatformOAuthLoginUrl("github")).toBe(
      "/api/v1/auth/login/github",
    );
    expect(() => getPlatformOAuthLoginUrl("../callback?next=https://evil.test")).toThrow(
      "Geçersiz giriş kanalı",
    );
  });
});
