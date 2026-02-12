import { type ReactElement, useMemo, useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { listRecipeTools } from "../../api";
import type {
  LlmConfig,
  LlmMcpProviderConfig,
  LlmToolConfig,
  McpEnvVar,
} from "../../types";
import { toastError, toastSuccess } from "@/shared/toast";
import { McpProvidersSection } from "./mcp-tools/mcp-providers-section";
import {
  createMcpProviderId,
  createToolConfigId,
  isProviderReadyForToolFetch,
  resolveLlmToolAlias,
  toApiProvider,
} from "./mcp-tools/helpers";
import { ToolConfigsSection } from "./mcp-tools/tool-configs-section";

type LlmMcpToolsTabProps = {
  config: LlmConfig;
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

const EMPTY_MCP_PROVIDERS: LlmMcpProviderConfig[] = [];
const EMPTY_TOOL_CONFIGS: LlmToolConfig[] = [];

function uniqueTrimmed(values: string[]): string[] {
  return Array.from(
    new Set(values.map((value) => value.trim()).filter(Boolean)),
  );
}

export function LlmMcpToolsTab({
  config,
  onUpdate,
}: LlmMcpToolsTabProps): ReactElement {
  const providers = config.mcp_providers ?? EMPTY_MCP_PROVIDERS;
  const toolConfigs = config.tool_configs ?? EMPTY_TOOL_CONFIGS;
  const [loadingTools, setLoadingTools] = useState(false);
  const [toolsByProvider, setToolsByProvider] = useState<Record<string, string[]>>(
    {},
  );

  function updateProviders(nextProviders: LlmMcpProviderConfig[]): void {
    onUpdate({ mcp_providers: nextProviders });
  }

  function updateToolConfigs(nextToolConfigs: LlmToolConfig[]): void {
    onUpdate({
      tool_configs: nextToolConfigs,
      // biome-ignore lint/style/useNamingConvention: api schema
      tool_alias: resolveLlmToolAlias(nextToolConfigs, config.tool_alias),
    });
  }

  function updateProviderAt(
    index: number,
    patch: Partial<LlmMcpProviderConfig>,
  ): void {
    updateProviders(
      providers.map((provider, currentIndex) =>
        currentIndex === index ? { ...provider, ...patch } : provider,
      ),
    );
  }

  function mutateProviderAt(
    index: number,
    mapProvider: (provider: LlmMcpProviderConfig) => Partial<LlmMcpProviderConfig>,
  ): void {
    const provider = providers[index];
    if (!provider) {
      return;
    }
    updateProviderAt(index, mapProvider(provider));
  }

  function removeProvider(index: number): void {
    updateProviders(
      providers.filter((_, currentIndex) => currentIndex !== index),
    );
  }

  function addProvider(): void {
    updateProviders([
      ...providers,
      {
        id: createMcpProviderId(config.id, providers.length),
        name: "",
        // biome-ignore lint/style/useNamingConvention: ui schema
        provider_type: "stdio",
        command: "",
        args: [""],
        env: [{ key: "", value: "" }],
        endpoint: "",
        // biome-ignore lint/style/useNamingConvention: api schema
        api_key: "",
        // biome-ignore lint/style/useNamingConvention: api schema
        api_key_env: "",
      },
    ]);
  }

  function addProviderArg(providerIndex: number): void {
    mutateProviderAt(providerIndex, (provider) => ({
      args: [...(provider.args ?? []), ""],
    }));
  }

  function updateProviderArg(
    providerIndex: number,
    argIndex: number,
    value: string,
  ): void {
    mutateProviderAt(providerIndex, (provider) => {
      const nextArgs = [...(provider.args ?? [])];
      nextArgs[argIndex] = value;
      return { args: nextArgs };
    });
  }

  function removeProviderArg(providerIndex: number, argIndex: number): void {
    mutateProviderAt(providerIndex, (provider) => ({
      args: (provider.args ?? []).filter(
        (_, currentIndex) => currentIndex !== argIndex,
      ),
    }));
  }

  function addProviderEnv(providerIndex: number): void {
    mutateProviderAt(providerIndex, (provider) => ({
      env: [...(provider.env ?? []), { key: "", value: "" }],
    }));
  }

  function updateProviderEnv(
    providerIndex: number,
    envIndex: number,
    patch: Partial<McpEnvVar>,
  ): void {
    mutateProviderAt(providerIndex, (provider) => ({
      env: (provider.env ?? []).map((item, currentIndex) =>
        currentIndex === envIndex ? { ...item, ...patch } : item,
      ),
    }));
  }

  function removeProviderEnv(providerIndex: number, envIndex: number): void {
    mutateProviderAt(providerIndex, (provider) => ({
      env: (provider.env ?? []).filter(
        (_, currentIndex) => currentIndex !== envIndex,
      ),
    }));
  }

  function updateToolConfigAt(
    index: number,
    patch: Partial<LlmToolConfig>,
  ): void {
    updateToolConfigs(
      toolConfigs.map((toolConfig, currentIndex) =>
        currentIndex === index ? { ...toolConfig, ...patch } : toolConfig,
      ),
    );
  }

  function addToolConfig(): void {
    updateToolConfigs([
      ...toolConfigs,
      {
        id: createToolConfigId(config.id, toolConfigs.length),
        // biome-ignore lint/style/useNamingConvention: api schema
        tool_alias: "",
        providers: [],
        // biome-ignore lint/style/useNamingConvention: api schema
        allow_tools: [],
        // biome-ignore lint/style/useNamingConvention: api schema
        max_tool_call_turns: "5",
        // biome-ignore lint/style/useNamingConvention: api schema
        timeout_sec: "",
      },
    ]);
  }

  function removeToolConfig(index: number): void {
    updateToolConfigs(
      toolConfigs.filter((_, currentIndex) => currentIndex !== index),
    );
  }

  async function loadToolNames(): Promise<void> {
    const apiProviders = providers
      .filter(isProviderReadyForToolFetch)
      .map(toApiProvider)
      .filter((provider) => Boolean(provider.name));

    if (apiProviders.length === 0) {
      toastError(
        "No MCP servers configured",
        "Add server name and command/endpoint first.",
      );
      return;
    }

    setLoadingTools(true);
    try {
      const response = await listRecipeTools({
        recipe: {
          // biome-ignore lint/style/useNamingConvention: api schema
          model_providers: [],
          // biome-ignore lint/style/useNamingConvention: api schema
          mcp_providers: apiProviders,
          // biome-ignore lint/style/useNamingConvention: api schema
          model_configs: [],
          // biome-ignore lint/style/useNamingConvention: api schema
          tool_configs: [],
          columns: [],
          processors: [],
        },
      });
      // biome-ignore lint/style/useNamingConvention: api schema
      setToolsByProvider(response.tools_by_provider ?? {});
      toastSuccess("Fetched MCP tools");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Tool fetch failed.";
      toastError("Failed to fetch tools", message);
    } finally {
      setLoadingTools(false);
    }
  }

  const providerNameSuggestions = useMemo(
    () => uniqueTrimmed(providers.map((provider) => provider.name)),
    [providers],
  );
  const toolAliasOptions = useMemo(
    () => uniqueTrimmed(toolConfigs.map((item) => item.tool_alias)),
    [toolConfigs],
  );
  const activeToolAlias = useMemo(() => {
    const currentAlias = config.tool_alias?.trim() ?? "";
    if (toolAliasOptions.includes(currentAlias)) {
      return currentAlias;
    }
    return toolAliasOptions[0] ?? "";
  }, [config.tool_alias, toolAliasOptions]);

  return (
    <div className="space-y-5">
      <div className="grid gap-2">
        <label className="text-xs font-semibold uppercase text-muted-foreground">
          Active tool alias
        </label>
        {toolAliasOptions.length > 0 ? (
          <Select
            value={activeToolAlias}
            onValueChange={(value) =>
              onUpdate({
                // biome-ignore lint/style/useNamingConvention: api schema
                tool_alias: value,
              })
            }
          >
            <SelectTrigger className="nodrag w-full">
              <SelectValue placeholder="Select active tool alias" />
            </SelectTrigger>
            <SelectContent>
              {toolAliasOptions.map((alias) => (
                <SelectItem key={alias} value={alias}>
                  {alias}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : (
          <p className="text-xs text-muted-foreground">
            Add tool config alias first.
          </p>
        )}
      </div>
      <ToolConfigsSection
        toolConfigs={toolConfigs}
        providerNameSuggestions={providerNameSuggestions}
        toolsByProvider={toolsByProvider}
        loadingTools={loadingTools}
        onFetchTools={() => {
          void loadToolNames();
        }}
        onAddToolConfig={addToolConfig}
        onUpdateToolConfig={updateToolConfigAt}
        onRemoveToolConfig={removeToolConfig}
      />
      <McpProvidersSection
        providers={providers}
        onAddProvider={addProvider}
        onUpdateProviderAt={updateProviderAt}
        onRemoveProvider={removeProvider}
        onAddProviderArg={addProviderArg}
        onUpdateProviderArg={updateProviderArg}
        onRemoveProviderArg={removeProviderArg}
        onAddProviderEnv={addProviderEnv}
        onUpdateProviderEnv={updateProviderEnv}
        onRemoveProviderEnv={removeProviderEnv}
      />
    </div>
  );
}
