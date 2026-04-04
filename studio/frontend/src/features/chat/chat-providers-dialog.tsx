// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import { CloudIcon, Delete02Icon, Wifi02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  createProviderConfig,
  deleteProviderConfig,
  listProviderConfigs,
  listProviderModels,
  listProviderRegistry,
  testProviderConnection,
  updateProviderConfig,
  type ProviderRegistryEntry,
} from "./api/providers-api";
import type { ExternalProviderConfig } from "./external-providers";
import {
  getExternalProviderApiKey,
  removeExternalProviderApiKey,
  setExternalProviderApiKey,
} from "./external-providers";
import { ApiProviderLogo } from "./api-provider-logo";

/** Matches navbar / thread layout easing (see index.css --ease-out-quart) */
const PROVIDER_FORM_EASE: [number, number, number, number] = [0.165, 0.84, 0.44, 1];
const PROVIDER_FORM_DURATION = 0.2;

function parseManualModelIds(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of text.split(/[\n,]+/)) {
    const id = raw.trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }
  return out;
}

interface ChatProvidersDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  providers: ExternalProviderConfig[];
  onProvidersChange: (providers: ExternalProviderConfig[]) => void;
}

export function ChatProvidersDialog({
  open,
  onOpenChange,
  providers,
  onProvidersChange,
}: ChatProvidersDialogProps) {
  const [providerType, setProviderType] = useState<string>("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrlDraft, setBaseUrlDraft] = useState("");
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null);
  const [registry, setRegistry] = useState<ProviderRegistryEntry[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [syncingProviders, setSyncingProviders] = useState(false);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [mutatingProvider, setMutatingProvider] = useState(false);
  const [manualModelIds, setManualModelIds] = useState("");
  const reduceMotion = useReducedMotion();

  const registryByType = useMemo(
    () => new Map(registry.map((entry) => [entry.provider_type, entry])),
    [registry],
  );

  const isCuratedModelList = useMemo(() => {
    return registryByType.get(providerType)?.model_list_mode === "curated";
  }, [registryByType, providerType]);

  const modelsPanelKey = isCuratedModelList ? "curated" : "remote";

  useEffect(() => {
    if (!providerType || editingProviderId) return;
    const entry = registryByType.get(providerType);
    if (entry?.model_list_mode === "curated") {
      setAvailableModels([...entry.default_models]);
      setSelectedModelIds([...entry.default_models]);
      setManualModelIds("");
    } else if (entry) {
      setAvailableModels([]);
      setSelectedModelIds([]);
      setManualModelIds("");
    }
  }, [providerType, editingProviderId, registryByType]);

  const totalModels = useMemo(
    () => providers.reduce((count, provider) => count + provider.models.length, 0),
    [providers],
  );

  useEffect(() => {
    if (!open) return;
    let isMounted = true;
    const syncFromBackend = async () => {
      setRegistryLoading(true);
      setSyncingProviders(true);
      try {
        const [registryRows, configRows] = await Promise.all([
          listProviderRegistry(),
          listProviderConfigs(),
        ]);
        if (!isMounted) return;
        setRegistry(registryRows);
        setProviderType((current) => {
          if (current && registryRows.some((entry) => entry.provider_type === current)) {
            return current;
          }
          return registryRows[0]?.provider_type ?? "";
        });
        const existingById = new Map(providers.map((provider) => [provider.id, provider]));
        const syncedProviders: ExternalProviderConfig[] = configRows
          .filter((config) => config.is_enabled)
          .map((config) => {
            const existing = existingById.get(config.id);
            const createdAt = Number.isFinite(Date.parse(config.created_at))
              ? Date.parse(config.created_at)
              : Date.now();
            const updatedAt = Number.isFinite(Date.parse(config.updated_at))
              ? Date.parse(config.updated_at)
              : Date.now();
            return {
              id: config.id,
              providerType: config.provider_type,
              name: config.display_name,
              baseUrl: config.base_url ?? "",
              models: existing?.models ?? [],
              createdAt: existing?.createdAt ?? createdAt,
              updatedAt,
            };
          });
        onProvidersChange(syncedProviders);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        toast.error(`Failed to load providers: ${message}`);
      } finally {
        if (isMounted) {
          setRegistryLoading(false);
          setSyncingProviders(false);
        }
      }
    };
    void syncFromBackend();
    return () => {
      isMounted = false;
    };
  }, [open, onProvidersChange]);

  function resetForm() {
    setEditingProviderId(null);
    setApiKey("");
    setBaseUrlDraft("");
    setAvailableModels([]);
    setSelectedModelIds([]);
    setManualModelIds("");
  }

  function toggleModel(modelId: string) {
    setSelectedModelIds((prev) =>
      prev.includes(modelId) ? prev.filter((id) => id !== modelId) : [...prev, modelId],
    );
  }

  function selectAllModels() {
    setSelectedModelIds([...availableModels]);
  }

  function clearModelSelection() {
    setSelectedModelIds([]);
  }

  function parseOptionalBaseUrl(input: string): string | null {
    const trimmed = input.trim();
    if (!trimmed) return null;
    let parsed: URL;
    try {
      parsed = new URL(trimmed);
    } catch {
      throw new Error("Base URL must be a valid URL.");
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error("Base URL must use http or https.");
    }
    return parsed.toString().replace(/\/+$/, "");
  }

  async function loadModels() {
    if (!providerType) {
      toast.error("Choose a provider first.");
      return;
    }
    if (isCuratedModelList) {
      toast.info(
        "This provider has a very large model catalog. Use the suggestions and add model IDs manually — full list is not fetched.",
      );
      return;
    }
    if (!apiKey.trim()) {
      toast.error("Add an API key first.");
      return;
    }
    setModelsLoading(true);
    try {
      const baseUrl = parseOptionalBaseUrl(baseUrlDraft);
      const models = await listProviderModels({
        providerType,
        apiKey: apiKey.trim(),
        baseUrl,
      });
      const modelIds = [...new Set(models
        .map((model) => model.id.trim())
        .filter((id) => id.length > 0))];
      setAvailableModels(modelIds);
      setSelectedModelIds([...modelIds]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Could not load models: ${message}`);
    } finally {
      setModelsLoading(false);
    }
  }

  async function addProvider() {
    if (!providerType) {
      toast.error("Choose a provider first.");
      return;
    }
    const selectedRegistryEntry = registryByType.get(providerType);
    const displayName = selectedRegistryEntry?.display_name ?? providerType;
    if (!apiKey.trim()) {
      toast.error("API key is required.");
      return;
    }
    const curated = selectedRegistryEntry?.model_list_mode === "curated";
    const modelsToSave = curated
      ? [...new Set([...selectedModelIds, ...parseManualModelIds(manualModelIds)])]
      : [...selectedModelIds];
    if (curated) {
      if (modelsToSave.length === 0) {
        toast.error(
          "Select at least one suggested model and/or add model IDs in the box below.",
        );
        return;
      }
    } else {
      if (availableModels.length === 0) {
        toast.error("Load available models first, then choose which to enable.");
        return;
      }
      if (selectedModelIds.length === 0) {
        toast.error("Select at least one model.");
        return;
      }
    }
    setMutatingProvider(true);
    try {
      const baseUrl = parseOptionalBaseUrl(baseUrlDraft);
      const created = await createProviderConfig({
        providerType,
        displayName,
        baseUrl,
      });
      const createdAt = Number.isFinite(Date.parse(created.created_at))
        ? Date.parse(created.created_at)
        : Date.now();
      const updatedAt = Number.isFinite(Date.parse(created.updated_at))
        ? Date.parse(created.updated_at)
        : Date.now();
      const provider: ExternalProviderConfig = {
        id: created.id,
        providerType: created.provider_type,
        name: created.display_name,
        baseUrl: created.base_url ?? "",
        models: modelsToSave,
        createdAt,
        updatedAt,
      };
      await setExternalProviderApiKey(created.id, apiKey.trim());
      onProvidersChange([...providers.filter((p) => p.id !== created.id), provider]);
      resetForm();
      toast.success("Provider added.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to add provider: ${message}`);
    } finally {
      setMutatingProvider(false);
    }
  }

  async function saveProviderEdits() {
    if (!editingProviderId) return;
    const existing = providers.find((provider) => provider.id === editingProviderId);
    if (!existing) {
      toast.error("Provider not found.");
      return;
    }
    if (!apiKey.trim()) {
      toast.error("API key is required.");
      return;
    }
    const entry = registryByType.get(existing.providerType);
    const curated = entry?.model_list_mode === "curated";
    const modelsToSave = curated
      ? [...new Set([...selectedModelIds, ...parseManualModelIds(manualModelIds)])]
      : [...selectedModelIds];
    if (curated) {
      if (modelsToSave.length === 0) {
        toast.error(
          "Select at least one suggested model and/or add model IDs in the box below.",
        );
        return;
      }
    } else {
      if (availableModels.length === 0) {
        toast.error("Load available models first, then choose which to enable.");
        return;
      }
      if (selectedModelIds.length === 0) {
        toast.error("Select at least one model.");
        return;
      }
    }
    setMutatingProvider(true);
    try {
      const baseUrl = parseOptionalBaseUrl(baseUrlDraft);
      const updated = await updateProviderConfig(editingProviderId, {
        displayName: existing.name,
        baseUrl,
      });
      await setExternalProviderApiKey(editingProviderId, apiKey.trim());
      const updatedAt = Number.isFinite(Date.parse(updated.updated_at))
        ? Date.parse(updated.updated_at)
        : Date.now();
      onProvidersChange(
        providers.map((provider) =>
          provider.id === editingProviderId
            ? {
                ...provider,
                baseUrl: updated.base_url ?? "",
                models: modelsToSave,
                updatedAt,
              }
            : provider,
        ),
      );
      toast.success("Provider updated.");
      resetForm();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to update provider: ${message}`);
    } finally {
      setMutatingProvider(false);
    }
  }

  async function editProvider(provider: ExternalProviderConfig) {
    setEditingProviderId(provider.id);
    setProviderType(provider.providerType);
    setApiKey(await getExternalProviderApiKey(provider.id));
    setBaseUrlDraft(provider.baseUrl);
    const entry = registryByType.get(provider.providerType);
    if (entry?.model_list_mode === "curated") {
      const defaults = new Set(entry.default_models);
      const inDefaults = provider.models.filter((m) => defaults.has(m));
      const custom = provider.models.filter((m) => !defaults.has(m));
      setAvailableModels([...entry.default_models]);
      setSelectedModelIds(inDefaults);
      setManualModelIds(custom.join("\n"));
    } else {
      setAvailableModels([...provider.models]);
      setSelectedModelIds([...provider.models]);
      setManualModelIds("");
    }
  }

  async function deleteProvider(providerId: string) {
    setMutatingProvider(true);
    try {
      await deleteProviderConfig(providerId);
      removeExternalProviderApiKey(providerId);
      onProvidersChange(providers.filter((provider) => provider.id !== providerId));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to delete provider: ${message}`);
    } finally {
      setMutatingProvider(false);
    }
  }

  async function testProvider(provider: ExternalProviderConfig) {
    const savedKey = (await getExternalProviderApiKey(provider.id)).trim();
    if (!savedKey) {
      await editProvider(provider);
      toast.info(`No API key found for ${provider.name}. Add one and save.`);
      return;
    }
    try {
      const result = await testProviderConnection({
        providerType: provider.providerType,
        apiKey: savedKey,
        baseUrl: provider.baseUrl || null,
      });
      if (result.success) {
        toast.success(result.message);
      } else {
        toast.error(result.message);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Test failed: ${message}`);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        overlayClassName="bg-black/50 backdrop-blur-sm"
        className={
          // 20% narrower than prior viewport-based widths (multiply by 0.8).
          "flex max-h-[90dvh] w-[calc(60vw-1.2rem)] sm:w-[56.4vw] lg:w-[55.2vw] xl:w-[54vw] sm:max-w-none flex-col gap-0 overflow-y-auto p-0"
        }
      >
        <DialogHeader className="shrink-0 space-y-2 px-8 pb-5 pt-8 text-left">
          <DialogTitle className="flex items-center gap-2.5 text-lg">
            <HugeiconsIcon icon={CloudIcon} className="size-5 text-muted-foreground" />
            API Providers
          </DialogTitle>
          <DialogDescription className="text-pretty">
            Choose a backend registry provider type, add your API key, then load models through
            the studio proxy.
          </DialogDescription>
        </DialogHeader>
        <Separator />
        <div className="grid min-w-0 grid-cols-1 gap-8 p-8 pt-6 sm:gap-10 md:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] md:gap-12 lg:gap-14">
          <div className="min-w-0">
            <div className="mb-5 flex flex-wrap items-end justify-between gap-2">
              <h3 className="text-sm font-semibold tracking-tight">Configured providers</h3>
              <span className="text-xs tabular-nums text-muted-foreground">
                {providers.length} providers · {totalModels} models
              </span>
            </div>
            <div className="flex flex-col gap-4">
              {providers.length === 0 ? (
                <div className="rounded-xl border border-dashed px-6 py-10 text-center text-sm leading-relaxed text-muted-foreground">
                  No providers yet. Add one using the form on the right.
                </div>
              ) : (
                providers.map((provider) => {
                  const registryEntry = registryByType.get(provider.providerType);
                  const detail = provider.baseUrl || registryEntry?.base_url || "";
                  const visibleModels = provider.models.slice(0, 5);
                  const hiddenModelsCount = Math.max(0, provider.models.length - visibleModels.length);
                  return (
                    <div key={provider.id} className="rounded-xl border p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex min-w-0 gap-3">
                          <ApiProviderLogo
                            providerType={provider.providerType}
                            className="mt-0.5 size-[calc(2rem*0.95)]"
                            title={provider.name}
                          />
                          <div className="min-w-0">
                            <div className="truncate text-sm font-medium">
                              {provider.name}
                            </div>
                            <div className="truncate text-xs text-muted-foreground">
                              <span className="font-mono">{provider.providerType}</span>
                              {" · "}
                              {registryEntry?.display_name ?? provider.providerType}
                              {detail ? ` · ${detail}` : ""}
                            </div>
                          </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-1.5">
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="h-7"
                            disabled={mutatingProvider}
                            onClick={() => editProvider(provider)}
                          >
                            Edit
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="h-7"
                            disabled={mutatingProvider}
                            onClick={() => void testProvider(provider)}
                          >
                            <HugeiconsIcon
                              icon={Wifi02Icon}
                              className="mr-1 size-[calc(0.875rem*0.95)]"
                            />
                            Check
                          </Button>
                          <Button
                            type="button"
                            size="icon-sm"
                            variant="ghost"
                            className="text-muted-foreground hover:text-destructive"
                            disabled={mutatingProvider}
                            onClick={() => void deleteProvider(provider.id)}
                            title="Delete provider"
                          >
                            <HugeiconsIcon
                              icon={Delete02Icon}
                              className="size-[calc(1rem*0.95)]"
                            />
                          </Button>
                        </div>
                      </div>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {visibleModels.map((model) => (
                          <span
                            key={`${provider.id}-${model}`}
                            className="rounded-md bg-muted px-2 py-1 text-[11px] text-muted-foreground"
                          >
                            {model}
                          </span>
                        ))}
                        {hiddenModelsCount > 0 ? (
                          <span
                            className="rounded-md bg-muted px-2 py-1 text-[11px] text-muted-foreground"
                            title={`${hiddenModelsCount} more model(s) hidden`}
                          >
                            +{hiddenModelsCount}
                          </span>
                        ) : null}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
          <div className="min-w-0 border-t border-border/80 pt-8 md:flex md:flex-col md:border-l md:border-t-0 md:pl-10 md:pt-0 lg:pl-12">
            <h3 className="mb-6 text-sm font-semibold tracking-tight">
              {editingProviderId ? "Edit provider" : "Add provider"}
            </h3>
            <div className="flex flex-col gap-6 md:flex-1">
              <div className="space-y-2">
                <Label htmlFor="provider-preset" className="text-sm font-medium">
                  Provider
                </Label>
                <Select
                  value={providerType}
                  onValueChange={(value) => {
                    if (editingProviderId) return;
                    setProviderType(value);
                    setAvailableModels([]);
                    setSelectedModelIds([]);
                    setManualModelIds("");
                  }}
                >
                  <SelectTrigger
                    id="provider-preset"
                    className="h-10 w-full text-sm"
                    disabled={editingProviderId != null}
                  >
                    <SelectValue placeholder="Choose a provider" />
                  </SelectTrigger>
                  <SelectContent>
                    {registry.map((entry) => (
                      <SelectItem key={entry.provider_type} value={entry.provider_type}>
                        <span className="flex items-center gap-2">
                          <ApiProviderLogo
                            providerType={entry.provider_type}
                            className="size-4"
                            title={entry.display_name}
                          />
                          {entry.display_name}
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Choose a provider from Studio's supported list. Your API key stays on this
                  device and is encrypted before each request.
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="provider-api-key" className="text-sm font-medium">
                  API key
                </Label>
                <Input
                  id="provider-api-key"
                  type="password"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  placeholder="Enter API key"
                  className="h-10 text-sm"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="provider-base-url" className="text-sm font-medium">
                  Base URL (optional)
                </Label>
                <Input
                  id="provider-base-url"
                  type="text"
                  value={baseUrlDraft}
                  onChange={(event) => setBaseUrlDraft(event.target.value)}
                  placeholder="https://my-vllm-server.com/v1"
                  className="h-10 text-sm"
                />
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Leave blank to use the registry default for this provider type. Set this to
                  target a custom OpenAI-compatible endpoint.
                </p>
              </div>

              <div className="space-y-3">
                <AnimatePresence initial={false} mode="wait">
                  <motion.div
                    key={modelsPanelKey}
                    className="origin-top space-y-3 overflow-hidden"
                    initial={
                      reduceMotion
                        ? false
                        : { opacity: 0, height: 0 }
                    }
                    animate={{ opacity: 1, height: "auto" }}
                    exit={
                      reduceMotion
                        ? undefined
                        : { opacity: 0, height: 0 }
                    }
                    transition={{
                      height: {
                        duration: PROVIDER_FORM_DURATION,
                        ease: PROVIDER_FORM_EASE,
                      },
                      opacity: {
                        duration: reduceMotion ? 0 : 0.14,
                        ease: PROVIDER_FORM_EASE,
                      },
                    }}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <Label className="text-sm font-medium">Models</Label>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-8"
                        disabled={modelsLoading || mutatingProvider || isCuratedModelList}
                        title={
                          isCuratedModelList
                            ? "Full catalog is not fetched for this provider"
                            : undefined
                        }
                        onClick={() => void loadModels()}
                      >
                        {modelsLoading ? (
                          <>
                            <Spinner className="mr-2 size-3.5" />
                            Loading…
                          </>
                        ) : (
                          "Load available models"
                        )}
                      </Button>
                    </div>
                    {isCuratedModelList ? (
                      <div className="space-y-3">
                        <p className="text-sm leading-relaxed text-muted-foreground">
                          This provider lists a huge number of models. Studio does not download the
                          full catalog — pick suggestions below and/or enter exact model IDs.
                        </p>
                        {availableModels.length > 0 ? (
                          <div className="space-y-3 rounded-lg border bg-muted/20 p-3">
                            <div className="flex flex-wrap gap-2">
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="h-7 text-xs"
                                onClick={selectAllModels}
                              >
                                Select all suggestions
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="h-7 text-xs"
                                onClick={() => {
                                  clearModelSelection();
                                  setManualModelIds("");
                                }}
                              >
                                Clear
                              </Button>
                            </div>
                            <ul className="max-h-48 space-y-2 overflow-y-auto pr-1">
                              {availableModels.map((model, index) => (
                                <li key={model} className="flex items-center gap-2.5">
                                  <Checkbox
                                    id={`provider-model-curated-${modelsPanelKey}-${index}`}
                                    checked={selectedModelIds.includes(model)}
                                    onCheckedChange={() => toggleModel(model)}
                                  />
                                  <label
                                    htmlFor={`provider-model-curated-${modelsPanelKey}-${index}`}
                                    className="min-w-0 cursor-pointer break-all text-sm leading-tight"
                                  >
                                    {model}
                                  </label>
                                </li>
                              ))}
                            </ul>
                          </div>
                        ) : null}
                        <div className="space-y-2">
                          <Label htmlFor="provider-manual-models" className="text-sm font-medium">
                            Model IDs (one per line or comma-separated)
                          </Label>
                          <Textarea
                            id="provider-manual-models"
                            value={manualModelIds}
                            onChange={(event) => setManualModelIds(event.target.value)}
                            placeholder={
                              "openai/gpt-4o-mini\nanthropic/claude-sonnet-4-5\ngoogle/gemini-2.5-flash"
                            }
                            rows={5}
                            className="min-h-[100px] resize-y font-mono text-sm"
                          />
                        </div>
                      </div>
                    ) : availableModels.length === 0 ? (
                      <p className="text-sm leading-relaxed text-muted-foreground">
                        Add your API key, then load models to choose which ones appear in chat.
                      </p>
                    ) : (
                      <div className="space-y-3 rounded-lg border bg-muted/20 p-3">
                        <div className="flex flex-wrap gap-2">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs"
                            onClick={selectAllModels}
                          >
                            Select all
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="h-7 text-xs"
                            onClick={clearModelSelection}
                          >
                            Clear
                          </Button>
                        </div>
                        <ul className="max-h-48 space-y-2 overflow-y-auto pr-1">
                          {availableModels.map((model, index) => (
                            <li key={model} className="flex items-center gap-2.5">
                              <Checkbox
                                id={`provider-model-remote-${modelsPanelKey}-${index}`}
                                checked={selectedModelIds.includes(model)}
                                onCheckedChange={() => toggleModel(model)}
                              />
                              <label
                                htmlFor={`provider-model-remote-${modelsPanelKey}-${index}`}
                                className="min-w-0 cursor-pointer break-all text-sm leading-tight"
                              >
                                {model}
                              </label>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </motion.div>
                </AnimatePresence>
              </div>

              <div className="mt-auto flex flex-wrap gap-3 pt-2">
                <Button
                  type="button"
                  disabled={registryLoading || syncingProviders || modelsLoading || mutatingProvider}
                  onClick={() =>
                    editingProviderId
                      ? void saveProviderEdits()
                      : void addProvider()
                  }
                >
                  {editingProviderId ? "Save provider" : "Add provider"}
                </Button>
                <Button type="button" variant="outline" onClick={resetForm}>
                  {editingProviderId ? "Cancel edit" : "Clear"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
