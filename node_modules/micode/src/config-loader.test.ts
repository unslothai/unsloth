// src/config-loader.test.ts
import { describe, expect, test } from "bun:test";

import { type MicodeConfig, type ProviderInfo, validateAgentModels } from "./config-loader";

// Helper to create a minimal ProviderInfo for testing
function createProvider(id: string, modelIds: string[]): ProviderInfo {
  const models: Record<string, unknown> = {};
  for (const modelId of modelIds) {
    models[modelId] = { id: modelId };
  }
  return { id, models };
}

describe("validateAgentModels", () => {
  test("returns config unchanged when all models are valid", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "openai/gpt-4" },
        brainstormer: { model: "anthropic/claude-3" },
      },
    };

    const providers: ProviderInfo[] = [
      createProvider("openai", ["gpt-4", "gpt-3.5"]),
      createProvider("anthropic", ["claude-3", "claude-2"]),
    ];

    const result = validateAgentModels(userConfig, providers);

    expect(result.agents?.commander?.model).toBe("openai/gpt-4");
    expect(result.agents?.brainstormer?.model).toBe("anthropic/claude-3");
  });

  test("removes model override when provider does not exist", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "nonexistent/gpt-4" },
      },
    };

    const providers: ProviderInfo[] = [createProvider("openai", ["gpt-4"])];

    const result = validateAgentModels(userConfig, providers);

    // Model should be removed, falling back to default
    expect(result.agents?.commander?.model).toBeUndefined();
  });

  test("removes model override when model does not exist in provider", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "openai/nonexistent-model" },
      },
    };

    const providers: ProviderInfo[] = [createProvider("openai", ["gpt-4", "gpt-3.5"])];

    const result = validateAgentModels(userConfig, providers);

    // Model should be removed, falling back to default
    expect(result.agents?.commander?.model).toBeUndefined();
  });

  test("preserves other properties when model is invalid", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: {
          model: "nonexistent/model",
          temperature: 0.7,
          maxTokens: 4000,
        },
      },
    };

    const providers: ProviderInfo[] = [createProvider("openai", ["gpt-4"])];

    const result = validateAgentModels(userConfig, providers);

    // Model removed but other properties preserved
    expect(result.agents?.commander?.model).toBeUndefined();
    expect(result.agents?.commander?.temperature).toBe(0.7);
    expect(result.agents?.commander?.maxTokens).toBe(4000);
  });

  test("handles config with no agents", () => {
    const userConfig: MicodeConfig = {};

    const providers: ProviderInfo[] = [createProvider("openai", ["gpt-4"])];

    const result = validateAgentModels(userConfig, providers);

    expect(result).toEqual({});
  });

  test("handles agent override with no model specified", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { temperature: 0.5 },
      },
    };

    const providers: ProviderInfo[] = [createProvider("openai", ["gpt-4"])];

    const result = validateAgentModels(userConfig, providers);

    // No model to validate, config unchanged
    expect(result.agents?.commander?.temperature).toBe(0.5);
    expect(result.agents?.commander?.model).toBeUndefined();
  });

  test("handles empty providers list", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "openai/gpt-4" },
      },
    };

    const providers: ProviderInfo[] = [];

    const result = validateAgentModels(userConfig, providers);

    // No providers available, config should remain unchanged
    expect(result).toEqual(userConfig);
  });

  test("handles providers with no models", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "openai/gpt-4" },
      },
    };

    const providers: ProviderInfo[] = [{ id: "openai", models: {} }];

    const result = validateAgentModels(userConfig, providers);

    // No provider models available, config should remain unchanged
    expect(result).toEqual(userConfig);
  });

  test("validates multiple agents with mixed valid/invalid models", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "openai/gpt-4" }, // valid
        brainstormer: { model: "fake/model" }, // invalid provider
        planner: { model: "openai/fake-model" }, // invalid model
        reviewer: { model: "anthropic/claude-3" }, // valid
      },
    };

    const providers: ProviderInfo[] = [
      createProvider("openai", ["gpt-4", "gpt-3.5"]),
      createProvider("anthropic", ["claude-3"]),
    ];

    const result = validateAgentModels(userConfig, providers);

    expect(result.agents?.commander?.model).toBe("openai/gpt-4");
    expect(result.agents?.brainstormer?.model).toBeUndefined();
    expect(result.agents?.planner?.model).toBeUndefined();
    expect(result.agents?.reviewer?.model).toBe("anthropic/claude-3");
  });

  test("removes empty string model", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "", temperature: 0.5 },
      },
    };

    const providers: ProviderInfo[] = [createProvider("openai", ["gpt-4"])];

    const result = validateAgentModels(userConfig, providers);

    // Empty string model should be removed as invalid
    expect(result.agents?.commander?.model).toBeUndefined();
    expect(result.agents?.commander?.temperature).toBe(0.5);
  });

  test("removes model string without slash (malformed)", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "gpt-4-no-provider" },
      },
    };

    const providers: ProviderInfo[] = [createProvider("openai", ["gpt-4"])];

    const result = validateAgentModels(userConfig, providers);

    // Malformed model (no slash) should be removed
    expect(result.agents?.commander?.model).toBeUndefined();
  });

  test("handles model with multiple slashes in model ID", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "openai/gpt-4/turbo" },
      },
    };

    // Model ID is "gpt-4/turbo" (contains slash)
    const providers: ProviderInfo[] = [createProvider("openai", ["gpt-4/turbo"])];

    const result = validateAgentModels(userConfig, providers);

    // Should be valid - "gpt-4/turbo" is the full model ID
    expect(result.agents?.commander?.model).toBe("openai/gpt-4/turbo");
  });

  test("returns consistent shape when all agents have invalid models", () => {
    const userConfig: MicodeConfig = {
      agents: {
        commander: { model: "invalid/model" },
      },
    };

    const providers: ProviderInfo[] = [createProvider("openai", ["gpt-4"])];

    const result = validateAgentModels(userConfig, providers);

    // Should return { agents: {} } for consistency, not {}
    expect(result).toEqual({ agents: {} });
  });
});
