import { describe, expect, it } from "vitest";

import {
  createCapabilityRegistry,
  getBackendMode,
} from "./platform-capabilities";

describe("Rag Platform capability registry", () => {
  it("hides unsupported legacy surfaces only in platform mode", () => {
    const registry = createCapabilityRegistry({
      VITE_BACKEND_MODE: "platform",
      VITE_RAG_PLATFORM_ENABLED: "true",
    });
    expect(registry.projects.available).toBe(true);
    expect(registry.knowledge.visibleInNavigation).toBe(true);
    expect(registry.training).toMatchObject({
      available: false,
      visibleInNavigation: false,
    });
    expect(registry["image-generation"].available).toBe(false);
    expect(registry["audio-generation"].available).toBe(false);
    expect(registry["video-generation"].available).toBe(false);
    expect(registry.export.available).toBe(false);
    expect(registry["api-monitor"].available).toBe(false);
    expect(registry.agents).toMatchObject({
      available: true,
      visibleInNavigation: true,
    });
  });

  it("keeps legacy-only surfaces available outside platform-only mode", () => {
    const registry = createCapabilityRegistry({ VITE_BACKEND_MODE: "legacy" });
    expect(registry.training.available).toBe(true);
    expect(registry.export.available).toBe(true);
    expect(registry["image-generation"].available).toBe(true);
    expect(registry["audio-generation"].available).toBe(true);
    expect(registry["video-generation"].available).toBe(true);
    expect(registry["api-monitor"].available).toBe(true);
    expect(registry.agents.available).toBe(false);
    const hybrid = createCapabilityRegistry({ VITE_BACKEND_MODE: "hybrid" });
    expect(hybrid.agents).toMatchObject({
      available: true,
      visibleInNavigation: true,
    });
  });

  it("uses an explicit backend mode and safe defaults", () => {
    expect(getBackendMode({ VITE_BACKEND_MODE: "hybrid" })).toBe("hybrid");
    expect(getBackendMode({ VITE_RAG_PLATFORM_ENABLED: "false" })).toBe(
      "legacy",
    );
    expect(getBackendMode({ VITE_RAG_PLATFORM_ENABLED: "true" })).toBe(
      "platform",
    );
  });
});
