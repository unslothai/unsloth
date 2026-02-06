// src/config-loader.ts
import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

import type { AgentConfig } from "@opencode-ai/sdk";

// Minimal type for provider validation - only what we need
export interface ProviderInfo {
  id: string;
  models: Record<string, unknown>;
}

/**
 * OpenCode config structure for reading default model and available models
 */
interface OpencodeConfig {
  model?: string;
  provider?: Record<string, { models?: Record<string, unknown> }>;
}

/**
 * Load opencode.json config file (synchronous)
 * Returns the parsed config or null if unavailable
 */
function loadOpencodeConfig(configDir?: string): OpencodeConfig | null {
  const baseDir = configDir ?? join(homedir(), ".config", "opencode");

  try {
    const configPath = join(baseDir, "opencode.json");
    const content = readFileSync(configPath, "utf-8");
    return JSON.parse(content) as OpencodeConfig;
  } catch {
    return null;
  }
}

/**
 * Load available models from opencode.json config file (synchronous)
 * Returns a Set of "provider/model" strings
 */
export function loadAvailableModels(configDir?: string): Set<string> {
  const availableModels = new Set<string>();
  const config = loadOpencodeConfig(configDir);

  if (config?.provider) {
    for (const [providerId, providerConfig] of Object.entries(config.provider)) {
      if (providerConfig.models) {
        for (const modelId of Object.keys(providerConfig.models)) {
          availableModels.add(`${providerId}/${modelId}`);
        }
      }
    }
  }

  return availableModels;
}

/**
 * Load the default model from opencode.json config file (synchronous)
 * Returns the model string in "provider/model" format or null if not set
 */
export function loadDefaultModel(configDir?: string): string | null {
  const config = loadOpencodeConfig(configDir);
  return config?.model ?? null;
}

// Safe properties that users can override
const SAFE_AGENT_PROPERTIES = ["model", "temperature", "maxTokens"] as const;

// Built-in OpenCode models that don't require validation (always available)
const BUILTIN_MODELS = new Set(["opencode/big-pickle"]);

export interface AgentOverride {
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface MicodeFeatures {
  mindmodelInjection?: boolean;
}

export interface MicodeConfig {
  agents?: Record<string, AgentOverride>;
  features?: MicodeFeatures;
  compactionThreshold?: number;
  fragments?: Record<string, string[]>;
}

/**
 * Load micode.json from ~/.config/opencode/micode.json
 * Returns null if file doesn't exist or is invalid JSON
 * @param configDir - Optional override for config directory (for testing)
 */
export async function loadMicodeConfig(configDir?: string): Promise<MicodeConfig | null> {
  const baseDir = configDir ?? join(homedir(), ".config", "opencode");
  const configPath = join(baseDir, "micode.json");

  try {
    const content = await readFile(configPath, "utf-8");
    const parsed = JSON.parse(content) as Record<string, unknown>;

    const result: MicodeConfig = {};

    // Sanitize agents - only allow safe properties
    if (parsed.agents && typeof parsed.agents === "object") {
      const sanitizedAgents: Record<string, AgentOverride> = {};

      for (const [agentName, agentConfig] of Object.entries(parsed.agents)) {
        if (agentConfig && typeof agentConfig === "object") {
          const sanitized: AgentOverride = {};
          const config = agentConfig as Record<string, unknown>;

          for (const prop of SAFE_AGENT_PROPERTIES) {
            if (prop in config) {
              (sanitized as Record<string, unknown>)[prop] = config[prop];
            }
          }

          sanitizedAgents[agentName] = sanitized;
        }
      }

      result.agents = sanitizedAgents;
    }

    // Parse features
    if (parsed.features && typeof parsed.features === "object") {
      const features = parsed.features as Record<string, unknown>;
      result.features = {
        mindmodelInjection: features.mindmodelInjection === true,
      };
    }

    // Parse compactionThreshold (must be number between 0 and 1)
    if (typeof parsed.compactionThreshold === "number") {
      const threshold = parsed.compactionThreshold;
      if (threshold >= 0 && threshold <= 1) {
        result.compactionThreshold = threshold;
      }
    }

    // Parse fragments
    if (parsed.fragments && typeof parsed.fragments === "object") {
      const fragments = parsed.fragments as Record<string, unknown>;
      const sanitizedFragments: Record<string, string[]> = {};

      for (const [agentName, fragmentList] of Object.entries(fragments)) {
        if (Array.isArray(fragmentList)) {
          const validFragments = fragmentList.filter((f): f is string => typeof f === "string" && f.trim().length > 0);
          if (validFragments.length > 0) {
            sanitizedFragments[agentName] = validFragments;
          }
        }
      }

      result.fragments = sanitizedFragments;
    }

    return result;
  } catch {
    return null;
  }
}

/**
 * Load model context limits from opencode.json
 * Returns a Map of "provider/model" -> context limit (tokens)
 */
export function loadModelContextLimits(configDir?: string): Map<string, number> {
  const limits = new Map<string, number>();
  const baseDir = configDir ?? join(homedir(), ".config", "opencode");

  try {
    const configPath = join(baseDir, "opencode.json");
    const content = readFileSync(configPath, "utf-8");
    const config = JSON.parse(content) as {
      provider?: Record<string, { models?: Record<string, { limit?: { context?: number } }> }>;
    };

    if (config.provider) {
      for (const [providerId, providerConfig] of Object.entries(config.provider)) {
        if (providerConfig.models) {
          for (const [modelId, modelConfig] of Object.entries(providerConfig.models)) {
            const contextLimit = modelConfig?.limit?.context;
            if (typeof contextLimit === "number" && contextLimit > 0) {
              limits.set(`${providerId}/${modelId}`, contextLimit);
            }
          }
        }
      }
    }
  } catch {
    // Config doesn't exist or can't be parsed - return empty map
  }

  return limits;
}

/**
 * Merge user config overrides into plugin agent configs
 * Model overrides are validated against available models from opencode.json
 * Invalid models are logged and skipped (agent uses opencode default)
 *
 * Model resolution priority:
 * 1. Per-agent override in micode.json (highest)
 * 2. Default model from opencode.json "model" field
 * 3. Plugin default (hardcoded in agent definitions)
 */
export function mergeAgentConfigs(
  pluginAgents: Record<string, AgentConfig>,
  userConfig: MicodeConfig | null,
  availableModels?: Set<string>,
  defaultModel?: string | null,
): Record<string, AgentConfig> {
  const models = availableModels ?? loadAvailableModels();
  const shouldValidateModels = models.size > 0;
  const opencodeDefaultModel = defaultModel ?? loadDefaultModel();

  // Helper to validate a model string
  const isValidModel = (model: string): boolean => {
    if (BUILTIN_MODELS.has(model)) return true;
    if (!shouldValidateModels) return true;
    return models.has(model);
  };

  const merged: Record<string, AgentConfig> = {};

  for (const [name, agentConfig] of Object.entries(pluginAgents)) {
    const userOverride = userConfig?.agents?.[name];

    // Start with the base agent config
    let finalConfig: AgentConfig = { ...agentConfig };

    // Apply opencode default model if available and valid (overrides plugin default)
    if (opencodeDefaultModel && isValidModel(opencodeDefaultModel)) {
      finalConfig = { ...finalConfig, model: opencodeDefaultModel };
    }

    // Apply user overrides from micode.json (highest priority)
    if (userOverride) {
      if (userOverride.model) {
        if (isValidModel(userOverride.model)) {
          // Model is valid - apply all overrides including model
          finalConfig = { ...finalConfig, ...userOverride };
        } else {
          // Model is invalid - log warning and apply other overrides only
          const fallbackModel = finalConfig.model || "plugin default";
          console.warn(
            `[micode] Model "${userOverride.model}" for agent "${name}" is not available. Using ${fallbackModel}.`,
          );
          const { model: _ignored, ...safeOverrides } = userOverride;
          finalConfig = { ...finalConfig, ...safeOverrides };
        }
      } else {
        // No model in override - apply other overrides (keep resolved model)
        finalConfig = { ...finalConfig, ...userOverride };
      }
    }

    merged[name] = finalConfig;
  }

  return merged;
}

/**
 * Validate that configured models exist in available providers
 * Removes invalid model overrides and logs warnings
 */
export function validateAgentModels(userConfig: MicodeConfig, providers: ProviderInfo[]): MicodeConfig {
  if (!userConfig.agents) {
    return userConfig;
  }

  const hasAnyModels = providers.some((provider) => Object.keys(provider.models).length > 0);
  if (!hasAnyModels) {
    return userConfig;
  }

  // Build lookup map for providers and their models
  const providerMap = new Map<string, Set<string>>();
  for (const provider of providers) {
    providerMap.set(provider.id, new Set(Object.keys(provider.models)));
  }

  const validatedAgents: Record<string, AgentOverride> = {};

  for (const [agentName, override] of Object.entries(userConfig.agents)) {
    // No model specified - keep other properties as-is
    if (override.model === undefined) {
      validatedAgents[agentName] = override;
      continue;
    }

    // Empty or whitespace-only model - treat as invalid
    const trimmedModel = override.model.trim();
    if (!trimmedModel) {
      const { model: _removed, ...otherProps } = override;
      console.warn(`[micode] Empty model for agent "${agentName}". Using default model.`);
      if (Object.keys(otherProps).length > 0) {
        validatedAgents[agentName] = otherProps;
      }
      continue;
    }

    // Skip validation for built-in models
    if (BUILTIN_MODELS.has(trimmedModel)) {
      validatedAgents[agentName] = override;
      continue;
    }

    // Parse "provider/model" format
    const [providerID, ...rest] = trimmedModel.split("/");
    const modelID = rest.join("/");

    const providerModels = providerMap.get(providerID);
    const isValid = providerModels?.has(modelID) ?? false;

    if (isValid) {
      validatedAgents[agentName] = override;
    } else {
      // Remove invalid model but keep other properties
      const { model: _removed, ...otherProps } = override;
      console.warn(`[micode] Model "${override.model}" not found for agent "${agentName}". Using default model.`);
      if (Object.keys(otherProps).length > 0) {
        validatedAgents[agentName] = otherProps;
      }
    }
  }

  return { agents: validatedAgents };
}
