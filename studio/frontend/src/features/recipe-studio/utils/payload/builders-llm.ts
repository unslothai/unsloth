import type { LlmConfig, LlmMcpProviderConfig, LlmToolConfig } from "../../types";

export function buildLlmColumn(
  config: LlmConfig,
  errors: string[],
): Record<string, unknown> {
  const toolAlias = config.tool_alias?.trim();
  const base = {
    name: config.name,
    drop: config.drop ?? false,
    // biome-ignore lint/style/useNamingConvention: api schema
    model_alias: config.model_alias,
    prompt: config.prompt,
    // biome-ignore lint/style/useNamingConvention: api schema
    system_prompt: config.system_prompt || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    tool_alias: toolAlias || undefined,
  };

  if (config.llm_type === "code") {
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      column_type: "llm-code",
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      code_lang: config.code_lang || "python",
    };
  }
  if (config.llm_type === "structured") {
    let outputFormat: unknown = config.output_format || undefined;
    if (typeof outputFormat === "string" && outputFormat.trim()) {
      try {
        outputFormat = JSON.parse(outputFormat);
      } catch {
        errors.push(`LLM ${config.name}: output_format is not valid JSON.`);
      }
    }
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      column_type: "llm-structured",
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      output_format: outputFormat,
    };
  }
  if (config.llm_type === "judge") {
    const scores = (config.scores ?? [])
      .map((score) => {
        const options: Record<string, string> = {};
        for (const option of score.options ?? []) {
          const key = option.value.trim();
          const value = option.description.trim();
          if (!key || !value) {
            continue;
          }
          options[key] = value;
        }
        return {
          name: score.name.trim(),
          description: score.description.trim(),
          options,
        };
      })
      .filter(
        (score) =>
          score.name && score.description && Object.keys(score.options).length > 0,
      );
    if (scores.length === 0) {
      errors.push(`LLM ${config.name}: scores required for LLM Judge.`);
    }
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      column_type: "llm-judge",
      ...base,
      scores,
    };
  }
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    column_type: "llm-text",
    ...base,
    // biome-ignore lint/style/useNamingConvention: api schema
    with_trace: "none",
  };
}

export function buildLlmMcpProvider(
  provider: LlmMcpProviderConfig,
  errors: string[],
): Record<string, unknown> | null {
  const name = provider.name.trim();
  if (!name) {
    errors.push("MCP provider: name is required.");
    return null;
  }
  if (provider.provider_type === "stdio") {
    const command = provider.command?.trim() ?? "";
    if (!command) {
      errors.push(`MCP provider ${name}: command is required for stdio.`);
      return null;
    }
    const env: Record<string, string> = {};
    for (const item of provider.env ?? []) {
      const key = item.key.trim();
      const value = item.value.trim();
      if (key && value) {
        env[key] = value;
      }
    }
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      provider_type: "stdio",
      name,
      command,
      args: (provider.args ?? []).map((value) => value.trim()).filter(Boolean),
      env,
    };
  }
  const endpoint = provider.endpoint?.trim() ?? "";
  if (!endpoint) {
    errors.push(`MCP provider ${name}: endpoint is required.`);
    return null;
  }
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    provider_type: "streamable_http",
    name,
    endpoint,
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key: provider.api_key?.trim() || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key_env: provider.api_key_env?.trim() || undefined,
  };
}

export function buildLlmToolConfig(
  config: LlmToolConfig,
  errors: string[],
): Record<string, unknown> | null {
  const toolAlias = config.tool_alias.trim();
  if (!toolAlias) {
    errors.push("Tool config: tool_alias is required.");
    return null;
  }
  const providers = config.providers
    .map((value) => value.trim())
    .filter(Boolean);
  if (providers.length === 0) {
    errors.push(`Tool config ${toolAlias}: at least one provider is required.`);
    return null;
  }
  const allowTools = (config.allow_tools ?? [])
    .map((value) => value.trim())
    .filter(Boolean);
  const maxToolCallTurnsRaw = config.max_tool_call_turns?.trim();
  const maxToolCallTurns =
    maxToolCallTurnsRaw && Number.isFinite(Number(maxToolCallTurnsRaw))
      ? Number(maxToolCallTurnsRaw)
      : 5;
  const timeoutRaw = config.timeout_sec?.trim();
  const timeoutSec =
    timeoutRaw && Number.isFinite(Number(timeoutRaw))
      ? Number(timeoutRaw)
      : undefined;
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    tool_alias: toolAlias,
    providers,
    // biome-ignore lint/style/useNamingConvention: api schema
    allow_tools: allowTools.length > 0 ? allowTools : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    max_tool_call_turns: maxToolCallTurns,
    // biome-ignore lint/style/useNamingConvention: api schema
    timeout_sec: timeoutSec,
  };
}
