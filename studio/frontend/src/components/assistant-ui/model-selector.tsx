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
import { ApiProviderLogo } from "@/features/chat/api-provider-logo";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  Logout01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";
import type {
  ExternalModelOption,
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./model-selector/types";
import { HubModelPicker, LoraModelPicker } from "./model-selector/pickers";
import { Input } from "../ui/input";

export type {
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
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
  contentClassName?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  triggerDataTour?: string;
  contentDataTour?: string;
}

function ModelSelectorTrigger({
  currentModel,
  isLoaded,
  variant = "outline",
  size = "default",
  className,
  dataTour,
}: {
  currentModel?: ModelOption;
  isLoaded: boolean;
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
          "flex items-center gap-2 transition-colors",
          variant === "outline" &&
          "rounded-full border border-border/60 hover:bg-accent",
          variant === "ghost" && "rounded-md hover:bg-accent",
          variant === "muted" && "rounded-md bg-muted hover:bg-muted/80",
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
        <span className={isLoaded ? "text-foreground" : "text-muted-foreground"}>
          {currentModel?.name ?? "Select model..."}
        </span>
        {currentModel?.description && (
          <span className="text-muted-foreground text-xs">{currentModel.description}</span>
        )}
        <HugeiconsIcon
          icon={ArrowDown01Icon}
          className="size-3 shrink-0 text-muted-foreground"
        />
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
  className?: string;
  dataTour?: string;
}) {
  const hasSelection = Boolean(value);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const hasExternal = externalModels.length > 0;

  const chatOnlyTabsDefault = useMemo(
    () => (value && externalModels.some((m) => m.id === value) ? "external" : "hub"),
    [value, externalModels],
  );

  const studioTabsDefault = useMemo((): "hub" | "lora" | "external" => {
    if (value && externalModels.some((m) => m.id === value)) return "external";
    if (value && loraModels.some((l) => l.id === value)) return "lora";
    return "hub";
  }, [value, externalModels, loraModels]);

  return (
    <PopoverContent
      align="start"
      data-tour={dataTour}
      className={cn(
        "w-[min(440px,calc(100vw-1rem))] max-w-[calc(100vw-1rem)] min-w-0 gap-0 p-2",
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
  variant = "outline",
  size = "default",
  className,
  contentClassName,
  open: controlledOpen,
  onOpenChange,
  triggerDataTour,
  contentDataTour,
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
      const isExported = lora.source === "exported";
      const isMerged = lora.exportType === "merged";
      const tag = isExported
        ? isMerged ? "Merged · Exported" : "LoRA"
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
        description: `External · ${externalModel.providerName}`,
        icon: (
          <ApiProviderLogo
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

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <ModelSelectorTrigger
        currentModel={currentModel}
        isLoaded={isLoaded}
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
        className={contentClassName}
        dataTour={contentDataTour}
      />
    </Popover>
  );
}

ModelSelector.Trigger = ModelSelectorTrigger;
ModelSelector.Content = ModelSelectorContent;

function normalizeForSearch(value: string): string {
  return value.toLowerCase().replace(/[\s\-_\.]/g, "");
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
    const byProvider = new Map<string, { providerName: string; models: ExternalModelOption[] }>();
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
                  <ApiProviderLogo
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
                    onClick={() => onSelect(model.id, { source: "external", isLora: false })}
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
