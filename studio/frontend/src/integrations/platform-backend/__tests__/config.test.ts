import { describe, expect, it } from "vitest";

import {
  getPlatformAuthConfig,
  getPlatformBackendConfig,
  resolvePlatformUrl,
} from "../config";

describe("Rag Platform config", () => {
  it("uses a relative /api/v1 base by default", () => {
    const config = getPlatformBackendConfig({
      VITE_RAG_PLATFORM_ENABLED: "true",
      VITE_RAG_PLATFORM_BASE_URL: "",
      VITE_RAG_PLATFORM_API_PREFIX: "",
      VITE_RAG_PLATFORM_PROXY_TARGET: "http://127.0.0.1:9380/",
    });

    expect(resolvePlatformUrl("/system/ping", "api", config)).toBe(
      "/api/v1/system/ping",
    );
    expect(config.proxyTarget).toBe("http://127.0.0.1:9380");
    expect(config.enabled).toBe(true);
  });

  it("keeps root probes outside the API prefix", () => {
    const config = getPlatformBackendConfig({
      VITE_RAG_PLATFORM_ENABLED: "true",
      VITE_RAG_PLATFORM_BASE_URL: "https://platform.example.test/",
      VITE_RAG_PLATFORM_API_PREFIX: "api/v1/",
      VITE_RAG_PLATFORM_PROXY_TARGET: "",
    });

    expect(resolvePlatformUrl("/health", "root", config)).toBe(
      "https://platform.example.test/health",
    );
  });

  it("rejects absolute endpoint overrides to prevent bearer-token exfiltration", () => {
    const config = getPlatformBackendConfig({
      VITE_RAG_PLATFORM_ENABLED: "true",
      VITE_RAG_PLATFORM_BASE_URL: "",
      VITE_RAG_PLATFORM_API_PREFIX: "/api/v1",
      VITE_RAG_PLATFORM_PROXY_TARGET: "",
    });

    expect(() =>
      resolvePlatformUrl("https://attacker.example/api", "api", config),
    ).toThrow(TypeError);
    expect(() =>
      resolvePlatformUrl("//attacker.example/api", "api", config),
    ).toThrow(TypeError);
  });

  it("decodes the non-secret deployment public key and honors explicit rollout flags", () => {
    const config = getPlatformAuthConfig({
      VITE_RAG_PLATFORM_ENABLED: "true",
      VITE_RAG_PLATFORM_AUTH_ENABLED: "true",
      VITE_RAG_PLATFORM_AUTH_PUBLIC_KEY_B64: btoa("PUBLIC-KEY"),
      VITE_RAG_PLATFORM_REGISTRATION_ENABLED: "false",
      VITE_RAG_PLATFORM_PASSWORD_RECOVERY_ENABLED: "true",
      VITE_RAG_PLATFORM_OAUTH_ENABLED: "false",
    });

    expect(config).toEqual({
      enabled: true,
      oauthEnabled: false,
      passwordRecoveryEnabled: true,
      publicKeyPem: "PUBLIC-KEY",
      registrationEnabled: false,
    });
  });
});
