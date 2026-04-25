// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useI18n } from "@/features/i18n";
import { toastError } from "@/shared/toast";
import {
  ArrowRight01Icon,
  Delete02Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import { listMcpTools } from "../../api";
import { ChipInput } from "../../components/chip-input";
import { CollapsibleSectionTriggerButton } from "../shared/collapsible-section-trigger";
import type { LlmMcpProviderConfig, McpEnvVar, ToolProfileConfig } from "../../types";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";
import {
  addUnique,
  collectToolSuggestions,
  createMcpProviderId,
  isProviderReadyForToolFetch,
  toApiProvider,
} from "./helpers";

type ToolProfileDialogProps = {
  config: ToolProfileConfig;
  onUpdate: (patch: Partial<ToolProfileConfig>) => void;
};

function EmptyState({
  title,
  description,
}: {
  title: string;
  description: string;
}): ReactElement {
  return (
    <div className="rounded-2xl border border-dashed border-border/70 bg-muted/15 px-4 py-5 text-sm">
      <p className="font-semibold text-foreground">{title}</p>
      <p className="mt-1 text-xs text-muted-foreground">{description}</p>
    </div>
  );
}

function isProviderConfigured(provider: LlmMcpProviderConfig): boolean {
  const hasName = provider.name.trim().length > 0;
  if (!hasName) {
    return false;
  }
  if (provider.provider_type === "stdio") {
    return (provider.command?.trim().length ?? 0) > 0;
  }
  return (provider.endpoint?.trim().length ?? 0) > 0;
}

function McpServerCard({
  provider,
  index,
  toolsCount,
  error,
  open,
  onOpenChange,
  onUpdateProviderAt,
  onRemoveProvider,
  onAddProviderArg,
  onUpdateProviderArg,
  onRemoveProviderArg,
  onAddProviderEnv,
  onUpdateProviderEnv,
  onRemoveProviderEnv,
}: {
  provider: LlmMcpProviderConfig;
  index: number;
  toolsCount?: number;
  error?: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUpdateProviderAt: (
    index: number,
    patch: Partial<LlmMcpProviderConfig>,
  ) => void;
  onRemoveProvider: (index: number) => void;
  onAddProviderArg: (index: number) => void;
  onUpdateProviderArg: (index: number, argIndex: number, value: string) => void;
  onRemoveProviderArg: (index: number, argIndex: number) => void;
  onAddProviderEnv: (index: number) => void;
  onUpdateProviderEnv: (
    index: number,
    envIndex: number,
    patch: Partial<McpEnvVar>,
  ) => void;
  onRemoveProviderEnv: (index: number, envIndex: number) => void;
}): ReactElement {
  const { t } = useI18n();
  const args = provider.args && provider.args.length > 0 ? provider.args : [""];
  const envVars =
    provider.env && provider.env.length > 0
      ? provider.env
      : [{ key: "", value: "" }];
  const summaryTitle =
    provider.name.trim() || `${t("recipe.toolProfile.server.title")} ${index + 1}`;
  const transportLabel =
    provider.provider_type === "stdio"
      ? t("recipe.toolProfile.transport.localCommand")
      : t("recipe.toolProfile.transport.http");
  const toolsLabel =
    typeof toolsCount === "number"
      ? `${toolsCount} ${t("recipe.toolProfile.toolsCount")}`
      : null;
  const description =
    provider.provider_type === "stdio"
      ? t("recipe.toolProfile.server.localDescription")
      : t("recipe.toolProfile.server.remoteDescription");

  return (
    <Collapsible open={open} onOpenChange={onOpenChange}>
      <div className="rounded-2xl border border-border/60 bg-background/80">
        <div className="flex items-start gap-2 px-4 py-4">
          <CollapsibleTrigger asChild={true}>
            <button
              type="button"
              className="flex min-w-0 flex-1 items-start gap-3 text-left"
            >
              <HugeiconsIcon
                icon={ArrowRight01Icon}
                className={`mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform ${
                  open ? "rotate-90" : ""
                }`}
              />
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="truncate text-sm font-semibold text-foreground">
                    {summaryTitle}
                  </p>
                  <Badge variant="outline" className="rounded-full text-[10px] uppercase">
                    {transportLabel}
                  </Badge>
                  {toolsLabel ? (
                    <Badge variant="secondary" className="rounded-full text-[10px]">
                      {toolsLabel}
                    </Badge>
                  ) : null}
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{description}</p>
              </div>
            </button>
          </CollapsibleTrigger>
          <Button
            type="button"
            size="icon-sm"
            variant="ghost"
            onClick={() => onRemoveProvider(index)}
          >
            <HugeiconsIcon icon={Delete02Icon} className="size-4" />
          </Button>
        </div>

        <CollapsibleContent className="space-y-4 border-t border-border/50 px-4 pt-4 pb-4">
          {error && (
            <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          )}

          <div className="grid gap-1.5">
            <FieldLabel
              label={t("recipe.toolProfile.serverName.label")}
              hint={t("recipe.toolProfile.serverName.hint")}
            />
            <Input
              className="nodrag"
              value={provider.name}
              placeholder={t("recipe.toolProfile.serverName.placeholder")}
              onChange={(event) =>
                onUpdateProviderAt(index, { name: event.target.value })
              }
            />
          </div>

          <Tabs
            value={provider.provider_type}
            onValueChange={(value) =>
              onUpdateProviderAt(index, {
                // biome-ignore lint/style/useNamingConvention: ui schema
                provider_type: value === "stdio" ? "stdio" : "streamable_http",
              })
            }
          >
              <TabsList className="w-full">
                <TabsTrigger value="stdio">
                  {t("recipe.toolProfile.transport.localCommand")}
                </TabsTrigger>
                <TabsTrigger value="streamable_http">
                  {t("recipe.toolProfile.transport.httpEndpoint")}
                </TabsTrigger>
              </TabsList>
          </Tabs>

          {provider.provider_type === "stdio" ? (
            <div className="space-y-4">
              <div className="grid gap-1.5">
                <FieldLabel
                  label={t("recipe.toolProfile.command.label")}
                  hint={t("recipe.toolProfile.command.hint")}
                />
                <Input
                  className="nodrag"
                  value={provider.command ?? ""}
                  placeholder={t("recipe.toolProfile.command.placeholder")}
                  onChange={(event) =>
                    onUpdateProviderAt(index, { command: event.target.value })
                  }
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <FieldLabel
                    label={t("recipe.toolProfile.arguments.label")}
                    hint={t("recipe.toolProfile.arguments.hint")}
                  />
                  <Button
                    type="button"
                    size="xs"
                    variant="outline"
                    onClick={() => onAddProviderArg(index)}
                  >
                    <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
                    {t("recipe.toolProfile.arguments.add")}
                  </Button>
                </div>
                {args.map((arg, argIndex) => (
                  <div key={`${provider.id}-arg-${argIndex}`} className="flex gap-2">
                    <Input
                      className="nodrag"
                      value={arg}
                      placeholder={
                        argIndex === 0
                          ? t("recipe.toolProfile.arguments.firstPlaceholder")
                          : t("recipe.toolProfile.arguments.placeholder")
                      }
                      onChange={(event) =>
                        onUpdateProviderArg(index, argIndex, event.target.value)
                      }
                    />
                    <Button
                      type="button"
                      size="icon-sm"
                      variant="ghost"
                      onClick={() => onRemoveProviderArg(index, argIndex)}
                    >
                      <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                    </Button>
                  </div>
                ))}
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <FieldLabel
                    label={t("recipe.toolProfile.env.label")}
                    hint={t("recipe.toolProfile.env.hint")}
                  />
                  <Button
                    type="button"
                    size="xs"
                    variant="outline"
                    onClick={() => onAddProviderEnv(index)}
                  >
                    <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
                    {t("recipe.toolProfile.env.add")}
                  </Button>
                </div>
                {envVars.map((item, envIndex) => (
                  <div
                    key={`${provider.id}-env-${envIndex}`}
                    className="grid grid-cols-[1fr_1fr_auto] gap-2"
                  >
                    <Input
                      className="nodrag"
                      value={item.key}
                      placeholder={t("recipe.toolProfile.env.keyPlaceholder")}
                      onChange={(event) =>
                        onUpdateProviderEnv(index, envIndex, {
                          key: event.target.value,
                        })
                      }
                    />
                    <Input
                      className="nodrag"
                      value={item.value}
                      placeholder={t("recipe.toolProfile.env.valuePlaceholder")}
                      onChange={(event) =>
                        onUpdateProviderEnv(index, envIndex, {
                          value: event.target.value,
                        })
                      }
                    />
                    <Button
                      type="button"
                      size="icon-sm"
                      variant="ghost"
                      onClick={() => onRemoveProviderEnv(index, envIndex)}
                    >
                      <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid gap-1.5">
                <FieldLabel
                  label={t("recipe.toolProfile.endpoint.label")}
                  hint={t("recipe.toolProfile.endpoint.hint")}
                />
                <Input
                  className="nodrag"
                  value={provider.endpoint ?? ""}
                  placeholder={t("recipe.toolProfile.endpoint.placeholder")}
                  onChange={(event) =>
                    onUpdateProviderAt(index, { endpoint: event.target.value })
                  }
                />
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                <div className="grid gap-1.5">
                  <FieldLabel
                    label={t("recipe.toolProfile.apiKeyEnv.label")}
                    hint={t("recipe.toolProfile.apiKeyEnv.hint")}
                  />
                  <Input
                    className="nodrag"
                    value={provider.api_key_env ?? ""}
                    placeholder={t("recipe.toolProfile.apiKeyEnv.placeholder")}
                    onChange={(event) =>
                      onUpdateProviderAt(index, {
                        // biome-ignore lint/style/useNamingConvention: api schema
                        api_key_env: event.target.value,
                      })
                    }
                  />
                </div>
                <div className="grid gap-1.5">
                  <FieldLabel
                    label={t("recipe.toolProfile.apiKey.label")}
                    hint={t("recipe.toolProfile.apiKey.hint")}
                  />
                  <Input
                    className="nodrag"
                    value={provider.api_key ?? ""}
                    placeholder={t("recipe.toolProfile.apiKey.placeholder")}
                    onChange={(event) =>
                      onUpdateProviderAt(index, {
                        // biome-ignore lint/style/useNamingConvention: api schema
                        api_key: event.target.value,
                      })
                    }
                  />
                </div>
              </div>
            </div>
          )}
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

export function ToolProfileDialog({
  config,
  onUpdate,
}: ToolProfileDialogProps): ReactElement {
  const { t } = useI18n();
  const providers = config.mcp_providers;
  const [activeTab, setActiveTab] = useState<"profile" | "servers">(
    providers.length > 0 ? "profile" : "servers",
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [loadingTools, setLoadingTools] = useState(false);
  const [toolsByProvider, setToolsByProvider] = useState<Record<string, string[]>>(
    config.fetched_tools_by_provider ?? {},
  );
  const [providerErrors, setProviderErrors] = useState<Record<string, string>>({});
  const [duplicateTools, setDuplicateTools] = useState<Record<string, string[]>>({});
  const [openProviders, setOpenProviders] = useState<Record<string, boolean>>({});
  const previousProviderSignatureRef = useRef<string | null>(null);

  const providerSignature = useMemo(
    () =>
      JSON.stringify(
        providers.map((provider) => ({
          name: provider.name,
          // biome-ignore lint/style/useNamingConvention: ui schema
          provider_type: provider.provider_type,
          command: provider.command,
          args: provider.args,
          env: provider.env,
          endpoint: provider.endpoint,
          // biome-ignore lint/style/useNamingConvention: api schema
          api_key: provider.api_key,
          // biome-ignore lint/style/useNamingConvention: api schema
          api_key_env: provider.api_key_env,
        })),
      ),
    [providers],
  );

  useEffect(() => {
    const previousSignature = previousProviderSignatureRef.current;
    previousProviderSignatureRef.current = providerSignature;
    if (previousSignature === null) {
      setToolsByProvider(config.fetched_tools_by_provider ?? {});
      return;
    }
    if (previousSignature === providerSignature) {
      return;
    }
    setToolsByProvider({});
    setProviderErrors({});
    setDuplicateTools({});
    if (Object.keys(config.fetched_tools_by_provider ?? {}).length > 0) {
      onUpdate({
        // biome-ignore lint/style/useNamingConvention: ui schema
        fetched_tools_by_provider: {},
      });
    }
  }, [config.fetched_tools_by_provider, onUpdate, providerSignature]);

  useEffect(() => {
    const tools = config.fetched_tools_by_provider ?? {};
    setToolsByProvider(tools);
  }, [config.fetched_tools_by_provider]);

  useEffect(() => {
    setOpenProviders((current) => {
      const next: Record<string, boolean> = {};
      for (const provider of providers) {
        next[provider.id] =
          current[provider.id] ?? !isProviderConfigured(provider);
      }
      return next;
    });
  }, [providers]);

  function updateProviders(nextProviders: LlmMcpProviderConfig[]): void {
    onUpdate({
      // biome-ignore lint/style/useNamingConvention: ui schema
      mcp_providers: nextProviders,
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
    updateProviders(providers.filter((_, currentIndex) => currentIndex !== index));
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
        args: [],
        env: [],
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
      const nextArgs =
        provider.args && provider.args.length > 0 ? [...provider.args] : [""];
      nextArgs[argIndex] = value;
      return { args: nextArgs };
    });
  }

  function removeProviderArg(providerIndex: number, argIndex: number): void {
    mutateProviderAt(providerIndex, (provider) => ({
      args: (provider.args ?? []).filter((_, currentIndex) => currentIndex !== argIndex),
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
      env: (
        provider.env && provider.env.length > 0
          ? provider.env
          : [{ key: "", value: "" }]
      ).map((item, currentIndex) =>
        currentIndex === envIndex ? { ...item, ...patch } : item,
      ),
    }));
  }

  function removeProviderEnv(providerIndex: number, envIndex: number): void {
    mutateProviderAt(providerIndex, (provider) => ({
      env: (provider.env ?? []).filter((_, currentIndex) => currentIndex !== envIndex),
    }));
  }

  async function loadTools(): Promise<void> {
    const readyProviders = providers.filter(isProviderReadyForToolFetch);
    if (readyProviders.length === 0) {
      toastError(
        t("recipe.toolProfile.toast.noServersReadyTitle"),
        t("recipe.toolProfile.toast.noServersReadyDescription"),
      );
      return;
    }

    setLoadingTools(true);
    try {
      const timeoutRaw = config.timeout_sec?.trim();
      const timeoutSec =
        timeoutRaw && Number.isFinite(Number(timeoutRaw))
          ? Number(timeoutRaw)
          : 15;
      const response = await listMcpTools({
        // biome-ignore lint/style/useNamingConvention: api schema
        mcp_providers: readyProviders.map(toApiProvider),
        // biome-ignore lint/style/useNamingConvention: api schema
        timeout_sec: timeoutSec,
      });
      const nextToolsByProvider = Object.fromEntries(
        response.providers
          .filter((provider) => provider.name.trim())
          .map((provider) => [provider.name.trim(), provider.tools]),
      );
      setToolsByProvider(nextToolsByProvider);
      onUpdate({
        // biome-ignore lint/style/useNamingConvention: ui schema
        fetched_tools_by_provider: nextToolsByProvider,
      });
      setProviderErrors(
        Object.fromEntries(
          response.providers
            .filter((provider) => provider.name.trim() && provider.error)
            .map((provider) => [
              provider.name.trim(),
              provider.error ?? t("recipe.toolProfile.toast.loadToolsFailedShort"),
            ]),
        ),
      );
      setDuplicateTools(response.duplicate_tools ?? {});
    } catch (error) {
      toastError(
        t("recipe.toolProfile.toast.couldNotLoadToolsTitle"),
        error instanceof Error
          ? error.message
          : t("recipe.toolProfile.toast.couldNotLoadToolsDescription"),
      );
    } finally {
      setLoadingTools(false);
    }
  }

  const providerNames = useMemo(
    () =>
      Array.from(
        new Set(providers.map((provider) => provider.name.trim()).filter(Boolean)),
      ),
    [providers],
  );
  const availableTools = useMemo(
    () => collectToolSuggestions(providerNames, toolsByProvider),
    [providerNames, toolsByProvider],
  );
  const hasProviders = providers.length > 0;

  useEffect(() => {
    if (!hasProviders && activeTab === "profile") {
      setActiveTab("servers");
    }
  }, [activeTab, hasProviders]);

  return (
    <Tabs
      value={activeTab}
      onValueChange={(value) =>
        setActiveTab(value === "servers" ? "servers" : "profile")
      }
      className="w-full"
    >
      <TabsList className="w-full">
        <TabsTrigger value="servers">
          {t("recipe.toolProfile.tab.addServers")}
        </TabsTrigger>
        <TabsTrigger value="profile">
          {t("recipe.toolProfile.tab.chooseTools")}
        </TabsTrigger>
      </TabsList>

      <TabsContent value="profile" className="space-y-4 pt-3">
        <NameField
          label={t("recipe.toolProfile.accessName.label")}
          value={config.name}
          onChange={(value) => onUpdate({ name: value })}
        />

        {!hasProviders ? (
          <div className="space-y-3">
            <EmptyState
              title={t("recipe.toolProfile.empty.addServerTitle")}
              description={t("recipe.toolProfile.empty.addServerDescription")}
            />
            <Button
              type="button"
              variant="outline"
              onClick={() => setActiveTab("servers")}
            >
              {t("recipe.toolProfile.action.addServersFirst")}
            </Button>
          </div>
        ) : (
          <>
            <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3">
              <p className="text-sm font-semibold text-foreground">
                {t("recipe.toolProfile.pickTools.title")}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {t("recipe.toolProfile.pickTools.description")}
              </p>
            </div>
            <div className="space-y-3 rounded-2xl border border-border/60 bg-muted/10 p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-foreground">
                    {t("recipe.toolProfile.availableTools.title")}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {t("recipe.toolProfile.availableTools.description")}
                  </p>
                </div>
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  disabled={loadingTools}
                  onClick={() => {
                    void loadTools();
                  }}
                >
                  {loadingTools
                    ? t("recipe.common.loading")
                    : t("recipe.toolProfile.action.loadTools")}
                </Button>
              </div>

              {Object.keys(toolsByProvider).length === 0 &&
                Object.keys(providerErrors).length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    {t("recipe.toolProfile.availableTools.empty")}
                  </p>
                )}

              {Object.entries(toolsByProvider).map(([providerName, toolNames]) => (
                <div key={providerName} className="space-y-2">
                  <div className="flex items-center gap-2">
                    <p className="text-xs font-semibold uppercase text-muted-foreground">
                      {providerName}
                    </p>
                    <Badge variant="outline" className="rounded-full text-[10px]">
                      {toolNames.length}
                    </Badge>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {toolNames.map((toolName) => (
                      <Badge key={`${providerName}-${toolName}`} variant="secondary">
                        {toolName}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}

              {Object.entries(duplicateTools).length > 0 && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                  {t("recipe.toolProfile.duplicates.prefix")}{" "}
                  {Object.entries(duplicateTools)
                    .map(([toolName, providerList]) => `${toolName} (${providerList.join(", ")})`)
                    .join("; ")}
                </div>
              )}
            </div>

            <div className="grid gap-1.5">
              <FieldLabel
                label={t("recipe.toolProfile.allowTools.label")}
                hint={t("recipe.toolProfile.allowTools.hint")}
              />
              <ChipInput
                values={config.allow_tools ?? []}
                suggestions={availableTools}
                onAdd={(value) =>
                  onUpdate({
                    // biome-ignore lint/style/useNamingConvention: api schema
                    allow_tools: addUnique(config.allow_tools ?? [], value),
                  })
                }
                onRemove={(toolIndex) =>
                  onUpdate({
                    // biome-ignore lint/style/useNamingConvention: api schema
                    allow_tools: (config.allow_tools ?? []).filter(
                      (_, currentIndex) => currentIndex !== toolIndex,
                    ),
                  })
                }
                placeholder={t("recipe.toolProfile.allowTools.placeholder")}
              />
            </div>

            <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
              <CollapsibleTrigger asChild={true}>
                <CollapsibleSectionTriggerButton
                  label={t("recipe.toolProfile.limits.label")}
                  open={advancedOpen}
                />
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3">
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="grid gap-1.5">
                    <FieldLabel
                      label={t("recipe.toolProfile.limits.maxTurns")}
                      hint={t("recipe.toolProfile.limits.maxTurnsHint")}
                    />
                    <Input
                      className="nodrag"
                      value={config.max_tool_call_turns ?? ""}
                      onChange={(event) =>
                        onUpdate({
                          // biome-ignore lint/style/useNamingConvention: api schema
                          max_tool_call_turns: event.target.value,
                        })
                      }
                    />
                  </div>
                  <div className="grid gap-1.5">
                    <FieldLabel
                      label={t("recipe.toolProfile.limits.timeout")}
                      hint={t("recipe.toolProfile.limits.timeoutHint")}
                    />
                    <Input
                      className="nodrag"
                      value={config.timeout_sec ?? ""}
                      onChange={(event) =>
                        onUpdate({
                          // biome-ignore lint/style/useNamingConvention: api schema
                          timeout_sec: event.target.value,
                        })
                      }
                    />
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
          </>
        )}
      </TabsContent>

      <TabsContent value="servers" className="space-y-4 pt-3">
        <div className="rounded-2xl border border-border/60 bg-muted/10 px-4 py-3">
          <p className="text-sm font-semibold text-foreground">
            {t("recipe.toolProfile.servers.title")}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {t("recipe.toolProfile.servers.description")}
          </p>
        </div>
        <div className="flex items-center justify-between gap-3">
          <FieldLabel
            label={t("recipe.toolProfile.servers.label")}
            hint={t("recipe.toolProfile.servers.hint")}
          />
          <Button type="button" size="xs" variant="outline" onClick={addProvider}>
            <HugeiconsIcon icon={PlusSignIcon} className="size-3.5" />
            {t("recipe.toolProfile.action.addServer")}
          </Button>
        </div>

        {!hasProviders ? (
          <EmptyState
            title={t("recipe.toolProfile.servers.emptyTitle")}
            description={t("recipe.toolProfile.servers.emptyDescription")}
          />
        ) : (
          <div className="space-y-3">
            {providers.map((provider, index) => (
              <McpServerCard
                key={provider.id}
                provider={provider}
                index={index}
                toolsCount={
                  provider.name.trim()
                    ? (toolsByProvider[provider.name.trim()] ?? []).length
                    : undefined
                }
                error={provider.name.trim() ? providerErrors[provider.name.trim()] : undefined}
                open={openProviders[provider.id] ?? !isProviderConfigured(provider)}
                onOpenChange={(open) =>
                  setOpenProviders((current) => ({
                    ...current,
                    [provider.id]: open,
                  }))
                }
                onUpdateProviderAt={updateProviderAt}
                onRemoveProvider={removeProvider}
                onAddProviderArg={addProviderArg}
                onUpdateProviderArg={updateProviderArg}
                onRemoveProviderArg={removeProviderArg}
                onAddProviderEnv={addProviderEnv}
                onUpdateProviderEnv={updateProviderEnv}
                onRemoveProviderEnv={removeProviderEnv}
              />
            ))}
          </div>
        )}
      </TabsContent>
    </Tabs>
  );
}
