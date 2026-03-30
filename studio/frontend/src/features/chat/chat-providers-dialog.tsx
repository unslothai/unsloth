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
import { CloudIcon, Delete02Icon, TestTubeIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { API_PROVIDER_PRESETS, getApiProviderPreset } from "./api-provider-presets";
import type { ExternalProviderConfig } from "./external-providers";

interface ChatProvidersDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  providers: ExternalProviderConfig[];
  onProvidersChange: (providers: ExternalProviderConfig[]) => void;
}

/** UI-only stand-in until the proxy exposes GET …/models. */
function stubFetchModelsForPreset(presetId: string): Promise<string[]> {
  return new Promise((resolve) => {
    window.setTimeout(() => {
      const tables: Record<string, string[]> = {
        openai: ["gpt-4o", "gpt-4o-mini", "o3-mini", "gpt-4-turbo"],
        anthropic: ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
        google: ["gemini-2.0-flash", "gemini-1.5-pro"],
        groq: ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        together: ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
      };
      resolve(tables[presetId] ?? ["default-model"]);
    }, 400);
  });
}

export function ChatProvidersDialog({
  open,
  onOpenChange,
  providers,
  onProvidersChange,
}: ChatProvidersDialogProps) {
  const [presetId, setPresetId] = useState<string>("openai");
  const [apiKey, setApiKey] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);

  const totalModels = useMemo(
    () => providers.reduce((count, provider) => count + provider.models.length, 0),
    [providers],
  );

  function resetForm() {
    setPresetId("openai");
    setApiKey("");
    setAvailableModels([]);
    setSelectedModelIds([]);
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

  async function loadModels() {
    if (!apiKey.trim()) {
      toast.error("Add an API key first.");
      return;
    }
    setModelsLoading(true);
    try {
      // Replace with real proxy call: GET /v1/models (or studio route) using preset + key.
      const models = await stubFetchModelsForPreset(presetId);
      setAvailableModels(models);
      setSelectedModelIds([...models]);
    } catch {
      toast.error("Could not load models.");
    } finally {
      setModelsLoading(false);
    }
  }

  function addProvider() {
    const preset = getApiProviderPreset(presetId);
    const name = preset?.label ?? presetId;
    if (!apiKey.trim()) {
      toast.error("API key is required.");
      return;
    }
    if (availableModels.length === 0) {
      toast.error('Load available models first, then choose which to enable.');
      return;
    }
    if (selectedModelIds.length === 0) {
      toast.error("Select at least one model.");
      return;
    }
    const now = Date.now();
    const provider: ExternalProviderConfig = {
      id: crypto.randomUUID(),
      presetId,
      name,
      baseUrl: "",
      apiKey: apiKey.trim(),
      models: [...selectedModelIds],
      createdAt: now,
      updatedAt: now,
    };
    onProvidersChange([...providers, provider]);
    resetForm();
    toast.success("Provider added.");
  }

  function deleteProvider(providerId: string) {
    onProvidersChange(providers.filter((provider) => provider.id !== providerId));
  }

  function testProvider(providerName: string) {
    toast.info(`Test for ${providerName} will use the backend proxy when wired.`);
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
            Choose a preset provider and your API key — we load models for you via the studio
            proxy.
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
                  const preset = getApiProviderPreset(provider.presetId);
                  const detail = preset?.description ?? provider.baseUrl ?? "";
                  return (
                    <div key={provider.id} className="rounded-xl border p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="truncate text-sm font-medium">
                            {preset?.label ?? provider.name}
                          </div>
                          <div className="truncate text-xs text-muted-foreground">{detail}</div>
                        </div>
                        <div className="flex shrink-0 items-center gap-1.5">
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            className="h-7"
                            onClick={() => testProvider(provider.name)}
                          >
                            <HugeiconsIcon icon={TestTubeIcon} className="mr-1 size-3.5" />
                            Test
                          </Button>
                          <Button
                            type="button"
                            size="icon-sm"
                            variant="ghost"
                            className="text-muted-foreground hover:text-destructive"
                            onClick={() => deleteProvider(provider.id)}
                            title="Delete provider"
                          >
                            <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                          </Button>
                        </div>
                      </div>
                      <div className="mt-3 flex flex-wrap gap-2">
                        {provider.models.map((model) => (
                          <span
                            key={`${provider.id}-${model}`}
                            className="rounded-md bg-muted px-2 py-1 text-[11px] text-muted-foreground"
                          >
                            {model}
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
          <div className="min-w-0 border-t border-border/80 pt-8 md:border-l md:border-t-0 md:pl-10 md:pt-0 lg:pl-12">
            <h3 className="mb-6 text-sm font-semibold tracking-tight">Add provider</h3>
            <div className="flex flex-col gap-6">
              <div className="space-y-2">
                <Label htmlFor="provider-preset" className="text-xs font-medium">
                  Provider
                </Label>
                <Select
                  value={presetId}
                  onValueChange={(value) => {
                    setPresetId(value);
                    setAvailableModels([]);
                    setSelectedModelIds([]);
                  }}
                >
                  <SelectTrigger id="provider-preset" className="h-10 w-full text-sm">
                    <SelectValue placeholder="Choose a provider" />
                  </SelectTrigger>
                  <SelectContent>
                    {API_PROVIDER_PRESETS.map((p) => (
                      <SelectItem key={p.id} value={p.id}>
                        {p.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs leading-relaxed text-muted-foreground">
                  Presets use the studio proxy — you do not need to enter a base URL.
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="provider-api-key" className="text-xs font-medium">
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

              <div className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <Label className="text-xs font-medium">Models</Label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="h-8"
                    disabled={modelsLoading}
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
                {availableModels.length === 0 ? (
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    After you add your API key, load models so you can pick which ones appear in
                    chat. This will call the proxy <code className="rounded bg-muted px-1 text-[11px]">/models</code>{" "}
                    route when wired.
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
                            id={`provider-model-${index}`}
                            checked={selectedModelIds.includes(model)}
                            onCheckedChange={() => toggleModel(model)}
                          />
                          <label
                            htmlFor={`provider-model-${index}`}
                            className="min-w-0 cursor-pointer break-all text-sm leading-tight"
                          >
                            {model}
                          </label>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-3 pt-2">
                <Button type="button" onClick={addProvider}>
                  Add provider
                </Button>
                <Button type="button" variant="outline" onClick={resetForm}>
                  Clear
                </Button>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
