import { describe, expect, it } from "vitest";

import {
  getPlatformAuthConfig,
  getPlatformBackendConfig,
  getPlatformManagementConfig,
  isPlatformChatPersistenceEnabled,
  isPlatformModelToolsEnabled,
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

  it("keeps chat persistence on platform and hybrid backends only", () => {
    expect(isPlatformChatPersistenceEnabled({})).toBe(true);
    expect(
      isPlatformChatPersistenceEnabled({ VITE_BACKEND_MODE: "hybrid" }),
    ).toBe(true);
    expect(
      isPlatformChatPersistenceEnabled({ VITE_BACKEND_MODE: "legacy" }),
    ).toBe(false);
  });

  it("enables Phase 3 by default and honors only an explicit rollout disable", () => {
    expect(isPlatformModelToolsEnabled({})).toBe(true);
    expect(
      isPlatformModelToolsEnabled({
        VITE_RAG_PLATFORM_ENABLED: "true",
        VITE_RAG_PLATFORM_AUTH_ENABLED: "true",
        VITE_RAG_PLATFORM_MODEL_TOOLS_ENABLED: "false",
      }),
    ).toBe(false);
  });

  it("defaults every Phase 14 rollout area on and prevents disabled-area calls", () => {
    expect(getPlatformManagementConfig({})).toEqual({
      adminEnabled: true,
      adminOperationsEnabled: true,
      botsEnabled: true,
      channelsEnabled: true,
      tenantsEnabled: true,
    });
    expect(
      getPlatformManagementConfig({
        VITE_RAG_PLATFORM_ADMIN_ENABLED: "false",
        VITE_RAG_PLATFORM_ADMIN_OPERATIONS_ENABLED: "true",
        VITE_RAG_PLATFORM_TENANTS_ENABLED: "false",
        VITE_RAG_PLATFORM_BOTS_ENABLED: "false",
        VITE_RAG_PLATFORM_CHANNELS_ENABLED: "false",
      }),
    ).toEqual({
      adminEnabled: false,
      adminOperationsEnabled: false,
      botsEnabled: false,
      channelsEnabled: false,
      tenantsEnabled: false,
    });
  });
});
