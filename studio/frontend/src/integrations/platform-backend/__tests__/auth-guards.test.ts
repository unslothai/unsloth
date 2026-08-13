import { HttpResponse, http } from "msw";
import { isRedirect } from "@tanstack/react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { requireAuth, requireGuest } from "@/app/auth-guards";
import { resetPlatformAuthRequestsForTests } from "../auth-api";
import {
  getPlatformSessionToken,
  resetPlatformUnauthorizedRedirectForTests,
  storePlatformSessionToken,
} from "../auth-session";
import { platformTestServer } from "./test-server";

describe("platform auth guards", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_AUTH_ENABLED", "true");
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    localStorage.clear();
    resetPlatformAuthRequestsForTests();
    resetPlatformUnauthorizedRedirectForTests();
    window.history.replaceState(null, "", "/chat");
  });

  afterEach(() => vi.unstubAllEnvs());

  it("blocks protected routes without a token", async () => {
    const caught = await requireAuth().catch((error: unknown) => error);
    expect(isRedirect(caught)).toBe(true);
    expect((caught as Response & { options: { to: string } }).options.to).toBe(
      "/login",
    );
  });

  it("hydrates a persisted session and redirects authenticated guests", async () => {
    storePlatformSessionToken("valid-token");
    platformTestServer.use(
      http.get("http://platform.test/api/v1/users/me", ({ request }) => {
        expect(request.headers.get("authorization")).toBe("Bearer valid-token");
        return HttpResponse.json({
          code: 0,
          data: {
            id: "user-1",
            email: "user@example.test",
            nickname: "User",
            is_active: "1",
          },
        });
      }),
    );

    await expect(requireAuth()).resolves.toBeUndefined();
    const caught = await requireGuest().catch((error: unknown) => error);
    expect(isRedirect(caught)).toBe(true);
    expect((caught as Response & { options: { to: string } }).options.to).toBe(
      "/chat",
    );
  });

  it("does not create a login loop when the network is temporarily unavailable", async () => {
    storePlatformSessionToken("offline-token");
    platformTestServer.use(
      http.get("http://platform.test/api/v1/users/me", () => HttpResponse.error()),
    );

    await expect(requireAuth()).resolves.toBeUndefined();
    expect(getPlatformSessionToken()).toBe("offline-token");
    expect(window.location.pathname).toBe("/chat");
  });
});
