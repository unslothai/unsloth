// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { usePlatformStore } from "@/config/env";
import {
  isCustomProviderType,
  savePerModelConfig,
  touchRecentModel,
  validateChatTemplate,
  type PerModelConfig,
} from "@/features/chat";
import { apiProviderLogoSrc } from "@/features/chat/api-provider-logo";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  CloudIcon,
  DashboardSquare01Icon,
  FolderSearchIcon,
  Logout01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useMemo, useState } from "react";
import type {
  DeletedModelRef,
  ExternalModelOption,
  LoraModelOption,
  ModelOption,
  ModelPickTarget,
  ModelSelectorChangeMeta,
} from "./model-selector/types";
import { HubModelPicker, LoraModelPicker } from "./model-selector/pickers";
import { trainedModelDescription } from "./model-selector/model-output-labels";
import { toast } from "sonner";
import { ModelConfigPage } from "./model-selector/model-config-page";
import { normalizeForSearch } from "@/lib/search-text";
import { Input } from "../ui/input";

function ExternalProviderLogo({
  providerType,
  className,
  title,
}: {
  providerType: string | undefined;
  className?: string;
  title?: string;
}) {
  const src = apiProviderLogoSrc(providerType);
  if (!src && isCustomProviderType(providerType)) {
    return (
      <span title={title} aria-hidden={true} className="inline-flex shrink-0">
        <HugeiconsIcon
          icon={DashboardSquare01Icon}
          className={cn("shrink-0", className)}
        />
      </span>
    );
  }

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
  localModels?: ModelOption[];
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
        title={currentModel?.id ?? currentModel?.name}
        className={cn(
          "flex min-w-0 items-center gap-2 transition-colors",
          variant === "outline" &&
          "rounded-[10px] border border-border/60 hover:bg-[#ececec] dark:hover:bg-[#2a2b2f]",
          variant === "ghost" && "rounded-[10px] hover:bg-[#ececec] dark:hover:bg-[#2a2b2f]",
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
        <span className="flex min-w-0 flex-1 items-baseline gap-2">
          <span className="min-w-0 flex-1 truncate font-heading text-[16px] font-medium leading-tight text-[#232528] dark:text-white">
            {currentModel?.name ?? "Select model"}
            {showCloudIndicator ? (
              <HugeiconsIcon
                icon={CloudIcon}
                strokeWidth={1.75}
                className="relative top-[0.15625rem] ml-1.5 mr-[0.36rem] size-3.5 shrink-0 text-muted-foreground"
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

type PickerTab = "hub" | "lora" | "external";

const PICKER_TAB_LABELS: Record<PickerTab, string> = {
  hub: "On Device",
  lora: "Train",
  external: "External",
};

function PickerTabToggle({
  tabs,
  tab,
  onTabChange,
}: {
  tabs: PickerTab[];
  tab: PickerTab;
  onTabChange: (tab: PickerTab) => void;
}) {
  const activeIndex = Math.max(0, tabs.indexOf(tab));
  return (
    <div
      className="menu-trigger tab-toggle relative mb-2 inline-flex h-9 w-full items-center rounded-full"
      role="radiogroup"
      aria-label="Model source"
    >
      <span
        aria-hidden="true"
        className="tab-toggle-pill pointer-events-none absolute inset-y-0 left-0 rounded-full transition-transform duration-200 ease-out"
        style={{
          width: `${100 / tabs.length}%`,
          transform: `translateX(${activeIndex * 100}%)`,
        }}
      />
      {tabs.map((t) => (
        <button
          key={t}
          type="button"
          role="radio"
          aria-checked={tab === t}
          onClick={() => onTabChange(t)}
          className={cn(
            "relative z-10 inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-[12.5px] transition-colors",
            tab === t
              ? "text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {PICKER_TAB_LABELS[t]}
        </button>
      ))}
    </div>
  );
}

function ModelSelectorListView({
  loraModels,
  externalModels,
  value,
  onPick,
  onPickExternal,
  onEject,
  onPickLocalModel,
  tab,
  onTabChange,
  enabled,
}: {
  models: ModelOption[];
  loraModels: LoraModelOption[];
  externalModels: ExternalModelOption[];
  value?: string;
  onPick: (target: ModelPickTarget) => void;
  onPickExternal: (id: string, meta: ModelSelectorChangeMeta) => void;
  onEject?: () => void;
  onFoldersChange?: () => void;
  onPickLocalModel?: () => void;
  tab: PickerTab;
  onTabChange: (tab: PickerTab) => void;
  enabled: boolean;
}) {
  const hasSelection = Boolean(value);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const hasExternal = externalModels.length > 0;

  const tabs = useMemo<PickerTab[]>(() => {
    if (chatOnly) return hasExternal ? ["hub", "external"] : ["hub"];
    return hasExternal ? ["hub", "lora", "external"] : ["hub", "lora"];
  }, [chatOnly, hasExternal]);
  const activeTab = tabs.includes(tab) ? tab : "hub";

  return (
    <>
      {tabs.length > 1 ? (
        <>
          <PickerTabToggle tabs={tabs} tab={activeTab} onTabChange={onTabChange} />
          {activeTab === "hub" ? (
            <HubModelPicker
              value={value}
              onPick={onPick}
              trainedModels={loraModels}
              enabled={enabled}
            />
          ) : activeTab === "lora" ? (
            <LoraModelPicker
              loraModels={loraModels}
              value={value}
              onPick={onPick}
            />
          ) : (
            <ExternalModelPicker
              externalModels={externalModels}
              value={value}
              onSelect={onPickExternal}
            />
          )}
        </>
      ) : (
        <HubModelPicker
          value={value}
          onPick={onPick}
          trainedModels={loraModels}
          enabled={enabled}
        />
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
    </>
  );
}

export function ModelSelector({
  models,
  localModels = [],
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
    for (const localModel of localModels) {
      all.set(localModel.id, localModel);
    }
    for (const lora of loraModels) {
      const runName = lora.runDisplayName?.trim();
      const displayName =
        runName && runName.length > 0
          ? runName
          : lora.name.includes("/")
            ? (lora.name.split("/").pop()?.trim() ?? lora.name)
            : lora.name;
      const isLocal = lora.source === "local";
      const isGguf = lora.exportType === "gguf";
      const tag = isLocal
        ? isGguf
          ? "GGUF"
          : "Local"
        : trainedModelDescription(lora);
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
  }, [externalModels, localModels, loraModels, models]);

  const currentModel = useMemo(() => {
    if (!selected) return undefined;
    const found = optionById.get(selected);
    if (activeGgufVariant) {
      const desc = `GGUF · ${activeGgufVariant}`;
      return found ? { ...found, description: desc } : { id: selected, name: selected, description: desc };
    }
    return found ?? { id: selected, name: selected };
  }, [selected, optionById, activeGgufVariant]);

  const [view, setView] = useState<"list" | "config">("list");
  const [target, setTarget] = useState<ModelPickTarget | null>(null);
  const [pickerTab, setPickerTab] = useState<PickerTab>("hub");

  const resetView = useCallback(() => {
    setView("list");
    setTarget(null);
  }, []);

  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!next) {
        resetView();
        setPickerTab("hub");
      }
      setOpen(next);
    },
    [resetView, setOpen],
  );

  const handlePick = useCallback((next: ModelPickTarget) => {
    setTarget(next);
    setView("config");
  }, []);

  const handlePickExternal = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      if (onValueChange) {
        onValueChange(id, meta);
      } else {
        setUncontrolled(id);
      }
      handleOpenChange(false);
    },
    [handleOpenChange, onValueChange],
  );

  const handleEject = useCallback(() => {
    onEject?.();
    handleOpenChange(false);
  }, [handleOpenChange, onEject]);

  const handlePickLocalModel = useCallback(() => {
    handleOpenChange(false);
    void onPickLocalModel?.();
  }, [handleOpenChange, onPickLocalModel]);

  const handleRun = useCallback(
    async (config: PerModelConfig, remember: boolean) => {
      if (!target) return;
      if (config.chatTemplateOverride != null) {
        try {
          const result = await validateChatTemplate(config.chatTemplateOverride);
          if (!result.valid) {
            toast.error("Invalid chat template", {
              description:
                result.error ?? "Check the Jinja syntax and try again.",
            });
            return;
          }
        } catch (error) {
          toast.error("Couldn't validate chat template", {
            description: error instanceof Error ? error.message : undefined,
          });
          return;
        }
      }
      if (remember) {
        const saved = savePerModelConfig(
          target.id,
          target.meta.ggufVariant ?? null,
          config,
        );
        if (!saved) {
          toast.error("Couldn't save these settings", {
            description:
              "Browser storage may be full or this model may have settings from a newer app version. Nothing was changed.",
          });
          return;
        }
      }
      touchRecentModel({
        id: target.id,
        ggufVariant: target.meta.ggufVariant ?? null,
      });
      const meta: ModelSelectorChangeMeta = { ...target.meta, config };
      if (onValueChange) {
        onValueChange(target.id, meta);
      } else {
        setUncontrolled(target.id);
      }
      handleOpenChange(false);
    },
    [handleOpenChange, onValueChange, target],
  );

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <ModelSelectorTrigger
        currentModel={currentModel}
        isLoaded={isLoaded}
        showCloudIndicator={showCloudIndicator}
        variant={variant}
        size={size}
        className={className}
        dataTour={triggerDataTour}
      />
      <PopoverContent
        align="start"
        data-tour={contentDataTour}
        className={cn(
          "menu-soft-surface picker-scope ring-0 w-[min(440px,calc(100vw-1rem))] max-w-[calc(100vw-1rem)] min-w-0 gap-0 rounded-[20px] p-3 overflow-hidden",
          contentClassName,
        )}
      >
        {view === "list" ? (
          <div
            key="list"
            className="animate-in fade-in-0 slide-in-from-left-2 duration-150"
          >
            <ModelSelectorListView
              models={models}
              loraModels={loraModels}
              externalModels={externalModels}
              value={selected}
              onPick={handlePick}
              onPickExternal={handlePickExternal}
              onEject={onEject ? handleEject : undefined}
              onFoldersChange={onFoldersChange}
              onPickLocalModel={onPickLocalModel ? handlePickLocalModel : undefined}
              tab={pickerTab}
              onTabChange={setPickerTab}
              enabled={open}
            />
          </div>
        ) : target ? (
          <div
            key={`config-${target.id}-${target.meta.ggufVariant ?? ""}`}
            className="animate-in fade-in-0 slide-in-from-right-3 duration-150"
          >
            <ModelConfigPage
              target={target}
              onBack={resetView}
              onCancel={() => handleOpenChange(false)}
              onRun={handleRun}
              onDeleted={onModelsChange}
              deleteDisabled={deleteDisabled}
            />
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}

ModelSelector.Trigger = ModelSelectorTrigger;

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
