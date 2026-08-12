import { describe, expect, it } from "vitest";

import { getPlatformBackendConfig, resolvePlatformUrl } from "../config";

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
});
