import { beforeEach, describe, expect, it } from "vitest";

import {
  platformAuthErrorMessage,
  platformOAuthErrorMessage,
} from "@/features/auth/platform-auth-errors";
import { setLocale } from "@/i18n";
import { PlatformApiError } from "../errors";

describe("safe Turkish auth errors", () => {
  beforeEach(() => setLocale("tr"));

  it("maps duplicate users, wrong passwords, permissions, and timeouts", () => {
    expect(
      platformAuthErrorMessage(
        new PlatformApiError("User already registered", {
          code: 100,
          endpoint: "/users",
          httpStatus: 200,
        }),
      ),
    ).toContain("zaten var");
    expect(
      platformAuthErrorMessage(
        new PlatformApiError("password error", {
          code: 100,
          endpoint: "/auth/login",
          httpStatus: 200,
        }),
      ),
    ).toContain("Mevcut parola");
    expect(
      platformAuthErrorMessage(
        new PlatformApiError("forbidden", {
          code: 403,
          endpoint: "/users/me/models",
          httpStatus: 403,
        }),
      ),
    ).toContain("yetkiniz yok");
    expect(
      platformAuthErrorMessage(
        new PlatformApiError("secret backend detail", {
          code: "CLIENT_TIMEOUT",
          endpoint: "/auth/login",
          httpStatus: null,
        }),
      ),
    ).not.toContain("secret backend detail");
  });

  it("keeps OAuth failures allow-listed and non-sensitive", () => {
    expect(platformOAuthErrorMessage("invalid_state")).toContain(
      "güvenlik kontrolünü",
    );
    expect(platformOAuthErrorMessage("provider-secret-error")).toBe(
      "Harici giriş tamamlanamadı.",
    );
  });

  it("provides the same safe error surface in English", () => {
    setLocale("en");

    expect(
      platformAuthErrorMessage(
        new PlatformApiError("password error", {
          code: 100,
          endpoint: "/auth/login",
          httpStatus: 200,
        }),
      ),
    ).toMatch(/current password/i);
    expect(platformOAuthErrorMessage("invalid_state")).toContain(
      "security check",
    );
    expect(platformOAuthErrorMessage("provider-secret-error")).toBe(
      "External sign-in could not be completed.",
    );
  });
});
