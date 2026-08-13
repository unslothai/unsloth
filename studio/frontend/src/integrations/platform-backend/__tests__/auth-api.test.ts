import forge from "node-forge";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import authFixtureText from "../../../../../../docs/rag-platform/fixtures/auth.json?raw";
import {
  changePlatformPassword,
  getCurrentPlatformTenantModels,
  getPlatformAuthCapabilities,
  hydratePlatformSession,
  loginPlatformUser,
  logoutPlatformUser,
  registerPlatformUser,
  requestForgotPasswordCaptcha,
  resetForgottenPlatformPassword,
  resetPlatformAuthRequestsForTests,
  sendForgotPasswordOtp,
  updateCurrentPlatformTenantModels,
  updatePlatformProfile,
  verifyForgotPasswordOtp,
} from "../auth-api";
import {
  getPlatformSessionToken,
  resetPlatformUnauthorizedRedirectForTests,
  storePlatformSessionToken,
} from "../auth-session";
import { platformTestServer } from "./test-server";

const fixture = JSON.parse(authFixtureText) as {
  interactions: Array<{
    name: string;
    response: { body: Record<string, unknown> };
  }>;
};
const loginBody = fixture.interactions.find(
  (interaction) => interaction.name === "auth.login",
)?.response.body;

function success(data: unknown, headers?: HeadersInit) {
  return HttpResponse.json({ code: 0, data, message: "success" }, { headers });
}

function decryptPassword(privateKey: forge.pki.rsa.PrivateKey, value: string) {
  const base64Password = privateKey.decrypt(
    forge.util.decode64(value),
    "RSAES-PKCS1-V1_5",
  );
  return forge.util.decodeUtf8(forge.util.decode64(base64Password));
}

describe("platform auth typed service", () => {
  let keyPair: forge.pki.rsa.KeyPair;
  let publicKeyPem: string;

  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    localStorage.clear();
    resetPlatformAuthRequestsForTests();
    resetPlatformUnauthorizedRedirectForTests();
    keyPair = forge.pki.rsa.generateKeyPair({ bits: 1024, e: 0x10001 });
    publicKeyPem = forge.pki.publicKeyToPem(keyPair.publicKey);
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    window.history.replaceState(null, "", "/");
  });

  it("encrypts email login, extracts the response header, and normalizes the user", async () => {
    let postedPassword = "";
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/auth/login",
        async ({ request }) => {
          const body = (await request.json()) as {
            email: string;
            password: string;
          };
          expect(body.email).toBe("user@example.test");
          postedPassword = body.password;
          return new HttpResponse(JSON.stringify(loginBody), {
            headers: {
              authorization: "Bearer opaque-login-token",
              "content-type": "application/json",
            },
          });
        },
      ),
    );

    const result = await loginPlatformUser(
      " user@example.test ",
      "şifre-1234",
      { publicKeyPem },
    );

    expect(decryptPassword(keyPair.privateKey, postedPassword)).toBe(
      "şifre-1234",
    );
    expect(result.token).toBe("opaque-login-token");
    expect(result.user).toMatchObject({
      active: true,
      createdAt: 1786502416241,
      loginChannel: "password",
      nickname: "Fixture Capture",
      updatedAt: 1786502416398,
    });
    expect(getPlatformSessionToken()).toBe("opaque-login-token");
  });

  it("fails login when Authorization is missing and surfaces code !== 0", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/auth/login", () =>
        HttpResponse.json(loginBody),
      ),
    );
    await expect(
      loginPlatformUser("user@example.test", "password-1", { publicKeyPem }),
    ).rejects.toMatchObject({ code: "AUTH_HEADER_MISSING" });

    platformTestServer.use(
      http.post("http://platform.test/api/v1/auth/login", () =>
        HttpResponse.json({ code: 100, data: null, message: "password error" }),
      ),
    );
    await expect(
      loginPlatformUser("user@example.test", "wrong-pass", { publicKeyPem }),
    ).rejects.toMatchObject({ code: 100, httpStatus: 200 });
  });

  it("probes registration/password-login flags and only returns real OAuth channels", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/system/config", () =>
        success({ disablePasswordLogin: false, registerEnabled: 1 }),
      ),
      http.get("http://platform.test/api/v1/auth/login/channels", () =>
        success([
          { channel: "github", display_name: "GitHub", icon: "github" },
          { channel: "", display_name: "Invalid" },
        ]),
      ),
    );

    await expect(getPlatformAuthCapabilities()).resolves.toEqual({
      loginChannels: [
        { channel: "github", displayName: "GitHub", icon: "github" },
      ],
      passwordLoginEnabled: true,
      registrationEnabled: true,
    });
  });

  it("covers registration and the complete forgot-password chain", async () => {
    const calls: string[] = [];
    platformTestServer.use(
      http.post("http://platform.test/api/v1/users", async ({ request }) => {
        const body = (await request.json()) as { password: string };
        expect(decryptPassword(keyPair.privateKey, body.password)).toBe(
          "new-pass-1",
        );
        calls.push("register");
        return success((loginBody as { data: unknown }).data, {
          authorization: "register-token",
        });
      }),
      http.post(
        "http://platform.test/api/v1/auth/password/forgot/captcha",
        ({ request }) => {
          expect(new URL(request.url).searchParams.get("email")).toBe(
            "user@example.test",
          );
          calls.push("captcha");
          return new HttpResponse("image", {
            headers: { "content-type": "image/jpeg" },
          });
        },
      ),
      http.post(
        "http://platform.test/api/v1/auth/password/forgot/otp",
        async ({ request }) => {
          expect(await request.json()).toEqual({
            email: "user@example.test",
            captcha: "ABCD",
          });
          calls.push("otp");
          return success(true);
        },
      ),
      http.post(
        "http://platform.test/api/v1/auth/password/forgot/otp/verify",
        async ({ request }) => {
          expect(await request.json()).toEqual({
            email: "user@example.test",
            otp: "OTP-1",
          });
          calls.push("verify");
          return success(true);
        },
      ),
      http.post(
        "http://platform.test/api/v1/auth/password/reset",
        async ({ request }) => {
          const body = (await request.json()) as {
            new_password: string;
            confirm_new_password: string;
          };
          expect(body.new_password).toBe(body.confirm_new_password);
          expect(decryptPassword(keyPair.privateKey, body.new_password)).toBe(
            "reset-pass-1",
          );
          calls.push("reset");
          return success((loginBody as { data: unknown }).data, {
            authorization: "reset-token",
          });
        },
      ),
    );

    await registerPlatformUser(
      { email: "user@example.test", nickname: "User", password: "new-pass-1" },
      { publicKeyPem },
    );
    const captchaImage =
      await requestForgotPasswordCaptcha("user@example.test");
    expect(captchaImage).toMatchObject({ size: 5, type: "image/jpeg" });
    await sendForgotPasswordOtp("user@example.test", "ABCD");
    await verifyForgotPasswordOtp("user@example.test", "OTP-1");
    await resetForgottenPlatformPassword(
      { email: "user@example.test", password: "reset-pass-1" },
      { publicKeyPem },
    );
    expect(calls).toEqual(["register", "captcha", "otp", "verify", "reset"]);
    expect(getPlatformSessionToken()).toBe("reset-token");
  });

  it("hydrates after reload, changes profile/models, and invalidates after password change", async () => {
    const user = (loginBody as { data: Record<string, unknown> }).data;
    const models = {
      tenant_id: "tenant-1",
      name: "Workspace",
      role: "owner",
      llm_id: "chat-1",
      embd_id: "embed-1",
      rerank_id: "rerank-1",
      asr_id: "asr-1",
      tts_id: "tts-1",
      img2txt_id: "vision-1",
      ocr_id: "ocr-1",
      parser_ids: "parser-1",
    };
    let currentUser = user;
    platformTestServer.use(
      http.get("http://platform.test/api/v1/users/me", ({ request }) => {
        expect(request.headers.get("authorization")).toBe(
          "Bearer persisted-token",
        );
        return success(currentUser);
      }),
      http.patch(
        "http://platform.test/api/v1/users/me",
        async ({ request }) => {
          const body = (await request.json()) as Record<string, string>;
          if (body.new_password) {
            expect(decryptPassword(keyPair.privateKey, body.password)).toBe(
              "old-pass-1",
            );
            expect(decryptPassword(keyPair.privateKey, body.new_password)).toBe(
              "new-pass-2",
            );
          } else {
            expect(body.nickname).toBe("Updated");
            currentUser = { ...user, nickname: "Updated" };
          }
          return success(true);
        },
      ),
      http.get("http://platform.test/api/v1/users/me/models", () =>
        success(models),
      ),
      http.patch(
        "http://platform.test/api/v1/users/me/models",
        async ({ request }) => {
          const body = (await request.json()) as Record<string, string>;
          expect(body.tenant_id).toBe("tenant-1");
          expect(body.llm_id).toBe("chat-2");
          models.llm_id = body.llm_id;
          return success(true);
        },
      ),
    );
    storePlatformSessionToken("persisted-token");

    await expect(hydratePlatformSession()).resolves.toMatchObject({
      email: "<redacted:email>",
    });
    await expect(
      updatePlatformProfile({ nickname: "Updated" }),
    ).resolves.toMatchObject({ nickname: "Updated" });
    const currentModels = await getCurrentPlatformTenantModels();
    await expect(
      updateCurrentPlatformTenantModels({
        ...currentModels,
        chatModelId: "chat-2",
      }),
    ).resolves.toMatchObject({ chatModelId: "chat-2", tenantId: "tenant-1" });
    await changePlatformPassword("old-pass-1", "new-pass-2", { publicKeyPem });
    expect(getPlatformSessionToken()).toBeNull();
  });

  it("clears local state on expired sessions and even when logout cannot reach the network", async () => {
    storePlatformSessionToken("expired-token");
    window.history.replaceState(null, "", "/chat");
    platformTestServer.use(
      http.get("http://platform.test/api/v1/users/me", () =>
        HttpResponse.json(
          { code: 401, data: null, message: "Unauthorized" },
          { status: 401 },
        ),
      ),
    );
    await expect(hydratePlatformSession()).rejects.toMatchObject({
      httpStatus: 401,
    });
    expect(getPlatformSessionToken()).toBeNull();
    expect(window.location.pathname).toBe("/login");

    storePlatformSessionToken("logout-token");
    platformTestServer.use(
      http.post("http://platform.test/api/v1/auth/logout", () =>
        HttpResponse.error(),
      ),
    );
    await expect(logoutPlatformUser()).rejects.toMatchObject({
      code: "NETWORK_ERROR",
    });
    expect(getPlatformSessionToken()).toBeNull();
  });
});
