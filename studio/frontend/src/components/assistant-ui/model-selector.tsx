// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { usePlatformStore } from "@/config/env";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  CloudIcon,
  FolderSearchIcon,
  Logout01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";
import type {
  DeletedModelRef,
  ExternalModelOption,
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./model-selector/types";
import { HubModelPicker, LoraModelPicker } from "./model-selector/pickers";
import { Input } from "../ui/input";

const PROVIDER_LOGO_EXT: Record<string, "svg" | "png" | "jpg"> = {
  openai: "svg",
  mistral: "svg",
  gemini: "svg",
  anthropic: "svg",
  deepseek: "svg",
  huggingface: "svg",
  kimi: "jpg",
  qwen: "png",
  openrouter: "svg",
};

function providerLogoSrc(providerType: string | undefined): string | undefined {
  if (!providerType) return undefined;
  const ext = PROVIDER_LOGO_EXT[providerType];
  if (!ext) return undefined;
  return `${import.meta.env.BASE_URL}provider-logos/${providerType}.${ext}`;
}

function ExternalProviderLogo({
  providerType,
  className,
  title,
}: {
  providerType: string | undefined;
  className?: string;
  title?: string;
}) {
  const src = providerLogoSrc(providerType);
  if (!src) return null;
  return (
    <img
      src={src}
      alt=""
      title={title}
      aria-hidden={true}
      className={cn(
        "shrink-0 object-contain",
        providerType === "openai" && "dark:invert",
        className,
      )}
    />
  );
}

export type {
  DeletedModelRef,
  ExternalModelOption,
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./model-selector/types";

interface ModelSelectorProps {
  models: ModelOption[];
  loraModels?: LoraModelOption[];
  externalModels?: ExternalModelOption[];
  value?: string;
  defaultValue?: string;
  activeGgufVariant?: string | null;
  onValueChange?: (value: string, meta: ModelSelectorChangeMeta) => void;
  onEject?: () => void;
  onFoldersChange?: () => void;
  onPickLocalModel?: () => void | Promise<void>;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
  contentClassName?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  triggerDataTour?: string;
  contentDataTour?: string;
  showCloudIndicator?: boolean;
}

function ModelSelectorTrigger({
  currentModel,
  isLoaded,
  showCloudIndicator = false,
  variant = "outline",
  size = "default",
  className,
  dataTour,
}: {
  currentModel?: ModelOption;
  isLoaded: boolean;
  showCloudIndicator?: boolean;
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
  dataTour?: string;
}) {
  return (
    <PopoverTrigger asChild={true}>
      <button
        type="button"
        data-tour={dataTour}
        className={cn(
          "flex min-w-0 items-center gap-2 transition-colors",
          variant === "outline" &&
          "rounded-[10px] border border-border/60 hover:bg-[#ececec] dark:hover:bg-[#2d2e32]",
          variant === "ghost" && "rounded-[10px] hover:bg-[#ececec] dark:hover:bg-[#2d2e32]",
          variant === "muted" && "rounded-[10px] bg-muted hover:bg-muted/80",
          size === "sm" && "h-8 px-3 text-xs",
          size === "default" && "h-9 px-3.5 text-sm",
          size === "lg" && "h-10 px-4 text-sm",
          className,
        )}
      >
        {isLoaded && (
          <span className="size-2 shrink-0 rounded-full bg-emerald-500" />
        )}
        {currentModel?.icon ? (
          <span className="flex shrink-0 items-center">{currentModel.icon}</span>
        ) : null}
        <span className="flex min-w-0 flex-1 items-baseline">
          <span className="min-w-0 flex flex-1 items-baseline truncate font-heading text-[16px] font-medium leading-tight text-black dark:text-white">
            {currentModel?.name ?? "Select model"}
            {showCloudIndicator ? (
              <HugeiconsIcon
                icon={CloudIcon}
                strokeWidth={1.75}
                className="relative top-[0.15625rem] ml-1.5 mr-1.5 size-3.5 shrink-0 text-muted-foreground"
              />
            ) : null}
          </span>
          {currentModel?.description && (
            <span
              className={cn(
                "shrink-0 text-xs leading-none text-muted-foreground",
                showCloudIndicator ? "" : "ml-2",
              )}
            >
              {currentModel.description}
            </span>
          )}
        </span>
        <span className="flex size-4 shrink-0 items-center justify-center">
          <HugeiconsIcon
            icon={ArrowDown01Icon}
            strokeWidth={1.75}
            className="relative top-0.5 size-3.5 text-muted-foreground"
          />
        </span>
      </button>
    </PopoverTrigger>
  );
}

function ModelSelectorContent({
  models,
  loraModels,
  externalModels,
  value,
  onSelect,
  onEject,
  onFoldersChange,
  onPickLocalModel,
  onModelsChange,
  deleteDisabled,
  className,
  dataTour,
}: {
  models: ModelOption[];
  loraModels: LoraModelOption[];
  externalModels: ExternalModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  onEject?: () => void;
  onFoldersChange?: () => void;
  onPickLocalModel?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  className?: string;
  dataTour?: string;
}) {
  const hasSelection = Boolean(value);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const hasExternal = externalModels.length > 0;
  const chatOnlyTabsDefault = useMemo(
    () => (value && externalModels.some((model) => model.id === value) ? "external" : "hub"),
    [externalModels, value],
  );
  const studioTabsDefault = useMemo((): "hub" | "lora" | "external" => {
    if (value && externalModels.some((model) => model.id === value)) {
      return "external";
    }
    if (value && loraModels.some((model) => model.id === value)) {
      return "lora";
    }
    return "hub";
  }, [externalModels, loraModels, value]);

  return (
    <PopoverContent
      align="start"
      data-tour={dataTour}
      className={cn(
        "menu-soft-surface ring-0 w-[min(440px,calc(100vw-1rem))] max-w-[calc(100vw-1rem)] min-w-0 gap-0 p-2",
        className,
      )}
    >
      {chatOnly ? (
        hasExternal ? (
          <Tabs defaultValue={chatOnlyTabsDefault} className="w-full">
            <TabsList className="mb-2 w-full">
              <TabsTrigger value="hub">Hub models</TabsTrigger>
              <TabsTrigger value="external">External</TabsTrigger>
            </TabsList>
            <TabsContent value="hub" className="m-0">
              <HubModelPicker models={models} value={value} onSelect={onSelect} onFoldersChange={onFoldersChange} />
            </TabsContent>
            <TabsContent value="external" className="m-0">
              <ExternalModelPicker
                externalModels={externalModels}
                value={value}
                onSelect={onSelect}
              />
            </TabsContent>
          </Tabs>
        ) : (
          <HubModelPicker models={models} value={value} onSelect={onSelect} onFoldersChange={onFoldersChange} />
        )
      ) : (
        <Tabs defaultValue={studioTabsDefault} className="w-full">
          <TabsList className="mb-2 w-full">
            <TabsTrigger value="hub">Hub models</TabsTrigger>
            <TabsTrigger value="lora">Fine-tuned</TabsTrigger>
            {hasExternal ? <TabsTrigger value="external">External</TabsTrigger> : null}
          </TabsList>

          <TabsContent value="hub" className="m-0">
            <HubModelPicker models={models} value={value} onSelect={onSelect} onFoldersChange={onFoldersChange} />
          </TabsContent>

          <TabsContent value="lora" className="m-0">
            <LoraModelPicker
              loraModels={loraModels}
              value={value}
              onSelect={onSelect}
              onModelsChange={onModelsChange}
              deleteDisabled={deleteDisabled}
            />
          </TabsContent>

          {hasExternal ? (
            <TabsContent value="external" className="m-0">
              <ExternalModelPicker
                externalModels={externalModels}
                value={value}
                onSelect={onSelect}
              />
            </TabsContent>
          ) : null}
        </Tabs>
      )}

      {onPickLocalModel ? (
        <div className="mt-2 border-t border-border/70 pt-2">
          <button
            type="button"
            onClick={onPickLocalModel}
            className="flex w-full items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted/60"
            title="Pick a model file from disk"
          >
            <HugeiconsIcon icon={FolderSearchIcon} className="size-3.5" />
            Pick a model file from disk
          </button>
        </div>
      ) : null}
      {hasSelection && onEject ? (
        <div className="mt-2 border-t border-border/70 pt-2">
          <button
            type="button"
            onClick={onEject}
            className="flex w-full items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-destructive transition-colors hover:bg-destructive/10"
            title="Eject model"
          >
            <HugeiconsIcon icon={Logout01Icon} className="size-3.5" />
            Eject loaded model
          </button>
        </div>
      ) : null}
    </PopoverContent>
  );
}

export function ModelSelector({
  models,
  loraModels = [],
  externalModels = [],
  value,
  defaultValue,
  activeGgufVariant,
  onValueChange,
  onEject,
  onFoldersChange,
  onPickLocalModel,
  onModelsChange,
  deleteDisabled,
  variant = "outline",
  size = "default",
  className,
  contentClassName,
  open: controlledOpen,
  onOpenChange,
  triggerDataTour,
  contentDataTour,
  showCloudIndicator = false,
}: ModelSelectorProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = onOpenChange ?? setUncontrolledOpen;
  const [uncontrolled, setUncontrolled] = useState(defaultValue ?? "");

  const selected = value ?? uncontrolled;
  const isLoaded = selected !== "";

  const optionById = useMemo(() => {
    const all = new Map<string, ModelOption>();
    for (const model of models) {
      all.set(model.id, model);
    }
    for (const lora of loraModels) {
      // Strip "/ suffix" from display name (e.g. "foo_123/foo" → "foo_123")
      const displayName = lora.name.includes("/")
        ? lora.name.split("/")[0].trim()
        : lora.name;
      // Show type tag instead of base model name
      const isLocal = lora.source === "local";
      const isTraining = lora.source === "training";
      const isExported = lora.source === "exported";
      const isMerged = lora.exportType === "merged";
      const isGguf = lora.exportType === "gguf";
      const tag = isLocal
        ? isGguf
          ? "GGUF"
          : "Local"
        : isTraining && isMerged
          ? "Full finetune"
          : isExported
            ? isMerged
              ? "Merged · Exported"
              : "LoRA · Exported"
            : "LoRA";
      all.set(lora.id, {
        ...lora,
        name: displayName,
        description: tag,
      });
    }
    for (const externalModel of externalModels) {
      all.set(externalModel.id, {
        ...externalModel,
        description: externalModel.providerName,
        icon: (
          <ExternalProviderLogo
            providerType={externalModel.providerType}
            className="size-4"
            title={externalModel.providerName}
          />
        ),
      });
    }
    return all;
  }, [externalModels, loraModels, models]);

  const currentModel = useMemo(() => {
    if (!selected) return undefined;
    const found = optionById.get(selected);
    if (activeGgufVariant) {
      const desc = `GGUF · ${activeGgufVariant}`;
      return found ? { ...found, description: desc } : { id: selected, name: selected, description: desc };
    }
    return found ?? { id: selected, name: selected };
  }, [selected, optionById, activeGgufVariant]);

  function handleSelect(id: string, meta: ModelSelectorChangeMeta) {
    if (onValueChange) {
      onValueChange(id, meta);
    } else {
      setUncontrolled(id);
    }
    setOpen(false);
  }

  function handleEject() {
    onEject?.();
    setOpen(false);
  }

  function handlePickLocalModel() {
    setOpen(false);
    void onPickLocalModel?.();
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <ModelSelectorTrigger
        currentModel={currentModel}
        isLoaded={isLoaded}
        showCloudIndicator={showCloudIndicator}
        variant={variant}
        size={size}
        className={className}
        dataTour={triggerDataTour}
      />
      <ModelSelectorContent
        models={models}
        loraModels={loraModels}
        externalModels={externalModels}
        value={selected}
        onSelect={handleSelect}
        onEject={onEject ? handleEject : undefined}
        onFoldersChange={onFoldersChange}
        onPickLocalModel={onPickLocalModel ? handlePickLocalModel : undefined}
        onModelsChange={onModelsChange}
        deleteDisabled={deleteDisabled}
        className={contentClassName}
        dataTour={contentDataTour}
      />
    </Popover>
  );
}

ModelSelector.Trigger = ModelSelectorTrigger;
ModelSelector.Content = ModelSelectorContent;

function normalizeForSearch(value: string): string {
  return value.toLowerCase().replace(/[\s_.-]/g, "");
}

function ExternalModelPicker({
  externalModels,
  value,
  onSelect,
}: {
  externalModels: ExternalModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
}) {
  const [query, setQuery] = useState("");
  const grouped = useMemo(() => {
    const needle = normalizeForSearch(query.trim());
    const byProvider = new Map<
      string,
      { providerName: string; models: ExternalModelOption[] }
    >();
    for (const model of externalModels) {
      const searchText = normalizeForSearch(
        `${model.name} ${model.providerName} ${model.id}`,
      );
      if (needle && !searchText.includes(needle)) continue;
      const prev = byProvider.get(model.providerId);
      if (prev) {
        prev.models.push(model);
      } else {
        byProvider.set(model.providerId, {
          providerName: model.providerName,
          models: [model],
        });
      }
    }
    return [...byProvider.entries()]
      .map(([providerId, group]) => ({
        providerId,
        providerName: group.providerName,
        models: group.models.sort((a, b) => a.name.localeCompare(b.name)),
      }))
      .sort((a, b) => a.providerName.localeCompare(b.providerName));
  }, [externalModels, query]);

  return (
    <div className="space-y-2">
      <div className="relative">
        <HugeiconsIcon
          icon={Search01Icon}
          className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search external models"
          className="h-9 pl-8"
        />
      </div>
      <div className="max-h-64 overflow-y-auto">
        <div className="space-y-2 p-1">
          {grouped.length === 0 ? (
            <div className="px-2.5 py-2 text-xs text-muted-foreground">
              No external models configured.
            </div>
          ) : (
            grouped.map((group) => (
              <div key={group.providerId}>
                <div className="flex items-center gap-2 px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  <ExternalProviderLogo
                    providerType={group.models[0]?.providerType}
                    className="size-3.5"
                    title={group.providerName}
                  />
                  <span className="min-w-0 truncate">{group.providerName}</span>
                </div>
                {group.models.map((model) => (
                  <button
                    key={model.id}
                    type="button"
                    onClick={() =>
                      onSelect(model.id, {
                        source: "external",
                        isLora: false,
                      })
                    }
                    className={cn(
                      "flex w-full items-center rounded-md px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-accent",
                      value === model.id && "bg-accent/60",
                    )}
                  >
                    <span className="min-w-0 truncate">{model.name}</span>
                  </button>
                ))}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
