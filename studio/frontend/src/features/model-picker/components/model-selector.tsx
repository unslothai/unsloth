// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { TooltipProvider } from "@/components/ui/tooltip";
import { usePlatformStore } from "@/config/env";
import { isCustomProviderType } from "@/features/chat";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import {
  CheckmarkCircle02Icon,
  CloudIcon,
  DashboardSquare01Icon,
  Download01Icon,
  FolderSearchIcon,
  RemoveCircleIcon,
  Search01Icon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type PerModelConfig,
  resolveInitialConfig,
} from "../model-config/per-model-config";
import { ModelConfigPage } from "./model-config-page";
import { HubModelPicker, hasDownloadedModels } from "./model-selector/pickers";
import { PillTabs } from "./model-selector/pill-tabs";
import {
  buildSourceTabs,
  isFineTunedSource,
} from "./model-selector/source-tabs";
import type {
  DeletedModelRef,
  ExternalModelOption,
  LoraModelOption,
  ModelOption,
  ModelPickTarget,
  ModelSelectorChangeMeta,
} from "./model-selector/types";

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
  vllm: "svg",
  ollama: "svg",
  llama_cpp: "svg",
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
  loraModels?: LoraModelOption[];
  externalModels?: ExternalModelOption[];
  value?: string;
  defaultValue?: string;
  activeGgufVariant?: string | null;
  activeModelConfig?: PerModelConfig | null;
  activeGgufContextLength?: number | null;
  selectedConfig?: PerModelConfig | null;
  selectedGgufVariant?: string | null;
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
  onEject,
}: {
  currentModel?: ModelOption;
  isLoaded: boolean;
  showCloudIndicator?: boolean;
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
  dataTour?: string;
  onEject?: () => void;
}) {
  return (
    <PopoverTrigger asChild={true}>
      <button
        type="button"
        data-tour={dataTour}
        className={cn(
          "unsloth-model-selector-trigger group/trigger flex min-w-0 items-center gap-2 transition-colors",
          // Suppress the pill's hover background while the eject hit area is
          // hovered, so only the dot's own circle reacts.
          variant === "outline" &&
            "rounded-full border border-border/60 hover:bg-accent has-[[data-eject-hit]:hover]:!bg-transparent",
          variant === "ghost" &&
            "rounded-full hover:bg-accent has-[[data-eject-hit]:hover]:!bg-transparent",
          variant === "muted" &&
            "rounded-full bg-muted hover:bg-muted/80 has-[[data-eject-hit]:hover]:!bg-muted",
          // More left padding than right; the chevron is pulled close to the
          // label (below) so the trigger reads balanced around the text.
          size === "sm" && "h-8 pl-3 pr-1.5 text-xs",
          size === "default" && "h-9 pl-4 pr-2 text-sm",
          size === "lg" && "h-10 pl-4.5 pr-2.5 text-sm",
          className,
        )}
      >
        {isLoaded &&
          (onEject ? (
            // Loaded status doubles as a mouse eject shortcut: green checkmark
            // at rest, red eject icon on pill hover, click to eject. A plain
            // span (no role/tabIndex) keeps it out of the trigger button's
            // content model, which forbids focusable descendants. Keyboard and
            // screen-reader users eject via the picker's "Eject model" button.
            // aria-hidden marks it decorative; stopPropagation stops the
            // popover from toggling. On touch (no hover) the eject icon and
            // tooltip never reveal, so pointer-events-none disables the
            // shortcut there and taps open the picker instead of ejecting.
            <span
              aria-hidden={true}
              title="Eject model"
              data-eject-hit={true}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                onEject();
              }}
              // Hit area larger than the icon, with a hover circle. Negative
              // margin keeps the icon in the dot's original spot.
              className="-m-1 flex size-5 shrink-0 cursor-pointer items-center justify-center rounded-full transition-colors hover:bg-black/10 dark:hover:bg-white/10 [@media(hover:none)]:pointer-events-none"
            >
              <HugeiconsIcon
                icon={CheckmarkCircle02Icon}
                strokeWidth={1.75}
                className="size-3.5 text-emerald-500 group-hover/trigger:hidden"
              />
              <HugeiconsIcon
                icon={RemoveCircleIcon}
                strokeWidth={1.75}
                className="hidden size-3.5 text-red-500 group-hover/trigger:block"
              />
            </span>
          ) : (
            <span className="size-2 shrink-0 rounded-full bg-emerald-500" />
          ))}
        {currentModel?.icon ? (
          <span className="flex shrink-0 items-center">
            {currentModel.icon}
          </span>
        ) : null}
        <span className="flex min-w-0 flex-1 items-baseline">
          <span className="min-w-0 flex flex-1 items-baseline truncate font-heading text-[16px] font-medium leading-tight text-black dark:text-white">
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
        <span className="-ml-1 flex size-4 shrink-0 items-center justify-center">
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={1.75}
            className="size-3.5 text-muted-foreground"
          />
        </span>
      </button>
    </PopoverTrigger>
  );
}

type HubSection = "downloaded" | "recommended" | "custom" | "connected";

// The user's most recently clicked Hub section, restored on every open so the
// selector returns to the tab they last used.
const HUB_SECTION_KEY = "unsloth_model_selector_section";
// Last tab the user actually clicked, or null when none is stored yet. Only
// On Device / Recommended persist (Connected is provider-conditional).
function loadLastHubSection(): HubSection | null {
  try {
    const raw = localStorage.getItem(HUB_SECTION_KEY);
    return raw === "downloaded" || raw === "recommended" ? raw : null;
  } catch {
    return null;
  }
}
function saveLastHubSection(section: HubSection): void {
  if (section !== "downloaded" && section !== "recommended") return;
  try {
    localStorage.setItem(HUB_SECTION_KEY, section);
  } catch {
    // Ignore unavailable storage.
  }
}
// Default the Hub section: the last tab the user clicked; first time, On Device
// when they have downloads, else Recommended.
function defaultHubSection(): HubSection {
  return (
    loadLastHubSection() ??
    (hasDownloadedModels() ? "downloaded" : "recommended")
  );
}

const HUB_SECTION_TABS: { value: string; label: string; icon?: ReactNode }[] = [
  {
    value: "recommended",
    label: "Recommended",
    icon: <HugeiconsIcon icon={StarIcon} className="size-3.5" />,
  },
  {
    value: "downloaded",
    label: "On Device",
    icon: <HugeiconsIcon icon={Download01Icon} className="size-3.5" />,
  },
];

function ModelSelectorContent({
  open,
  models,
  loraModels,
  externalModels,
  value,
  activeGgufVariant,
  activeModelConfig,
  activeGgufContextLength,
  selectedConfig,
  selectedGgufVariant,
  onSelect,
  onEject,
  onFoldersChange,
  onPickLocalModel,
  onBrowseHub,
  onModelsChange,
  deleteDisabled,
  className,
  dataTour,
}: {
  open: boolean;
  models: ModelOption[];
  loraModels: LoraModelOption[];
  externalModels: ExternalModelOption[];
  value?: string;
  activeGgufVariant?: string | null;
  activeModelConfig?: PerModelConfig | null;
  activeGgufContextLength?: number | null;
  selectedConfig?: PerModelConfig | null;
  selectedGgufVariant?: string | null;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  onEject?: () => void;
  onFoldersChange?: () => void;
  onPickLocalModel?: () => void;
  onBrowseHub?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  className?: string;
  dataTour?: string;
}) {
  const hasSelection = Boolean(value);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const hasExternal = externalModels.length > 0;
  // The Fine-tuned tab is for fine-tuned models only. Local models (LM Studio,
  // Ollama, custom folders) carry source "local" and live in the Hub tab's
  // Downloaded / Custom sections instead.
  const fineTunedModels = useMemo(
    () => loraModels.filter((model) => isFineTunedSource(model.source)),
    [loraModels],
  );
  const chatOnlyTabsDefault = useMemo(
    () =>
      value && externalModels.some((model) => model.id === value)
        ? "external"
        : "hub",
    [externalModels, value],
  );
  const studioTabsDefault = useMemo((): "hub" | "external" => {
    if (value && externalModels.some((model) => model.id === value)) {
      return "external";
    }
    return "hub";
  }, [externalModels, value]);

  const tabs = useMemo(() => buildSourceTabs(), []);
  // Connected sits in the section toggle, shown only with external providers.
  const hubSectionTabs = useMemo(
    () =>
      hasExternal
        ? [
            ...HUB_SECTION_TABS,
            {
              value: "connected",
              label: "Connected",
              icon: <HugeiconsIcon icon={CloudIcon} className="size-3.5" />,
            },
          ]
        : HUB_SECTION_TABS,
    [hasExternal],
  );

  const [activeTab, setActiveTab] = useState<string>(() =>
    chatOnly ? chatOnlyTabsDefault : studioTabsDefault,
  );
  // Fall back to the first tab if the active one disappears.
  const effectiveTab = tabs.some((tab) => tab.value === activeTab)
    ? activeTab
    : tabs[0].value;
  // Open on Connected when the active model comes from a connected provider.
  const wantsConnectedDefault =
    (chatOnly ? chatOnlyTabsDefault : studioTabsDefault) === "external";
  const [hubSection, setHubSection] = useState<HubSection>(() =>
    wantsConnectedDefault ? "connected" : defaultHubSection(),
  );
  // Connected is only valid while external providers exist; fall back otherwise.
  const effectiveHubSection: HubSection =
    hubSection === "connected" && !hasExternal ? "recommended" : hubSection;

  const [configTarget, setConfigTarget] = useState<ModelPickTarget | null>(
    null,
  );

  // The picker below remounts on each open, but this tab state does not, so a
  // persisted selection that lands in lora/external after async load would
  // reopen on Hub. Re-derive the default tab on the open edge.
  const wasOpen = useRef(open);
  useEffect(() => {
    if (open && !wasOpen.current) {
      setActiveTab(chatOnly ? chatOnlyTabsDefault : studioTabsDefault);
      // Connected when an external model is active, else On Device when the
      // user has downloads, else their last section.
      setHubSection(wantsConnectedDefault ? "connected" : defaultHubSection());
    }
    if (!open && wasOpen.current) {
      setConfigTarget(null);
    }
    wasOpen.current = open;
  }, [
    open,
    chatOnly,
    chatOnlyTabsDefault,
    studioTabsDefault,
    wantsConnectedDefault,
  ]);

  function focusActiveModelOption(root: HTMLElement): boolean {
    const option =
      root.querySelector<HTMLElement>(
        '[role="tabpanel"]:not([hidden]) [data-model-picker-active-option="true"]',
      ) ??
      root.querySelector<HTMLElement>(
        '[data-model-picker-active-option="true"]',
      ) ??
      root.querySelector<HTMLElement>(
        '[role="tabpanel"]:not([hidden]) [data-model-picker-option]',
      ) ??
      root.querySelector<HTMLElement>("[data-model-picker-option]");
    if (!option) {
      return false;
    }
    option.focus();
    return true;
  }

  function handlePickerEntryKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key !== "ArrowDown") {
      return;
    }

    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const isPickerSearchInput = target.matches(
      "[data-model-picker-search-input]",
    );
    const isTabTrigger = Boolean(target.closest('[role="tab"]'));
    if (!isPickerSearchInput && !isTabTrigger) {
      return;
    }

    if (focusActiveModelOption(event.currentTarget)) {
      event.preventDefault();
    }
  }

  const visibleConfigTarget = open ? configTarget : null;
  const openConfigPage = (id: string, meta: ModelSelectorChangeMeta) => {
    const leaf = id.includes("/") ? id.slice(id.lastIndexOf("/") + 1) : id;
    setConfigTarget({
      id,
      displayName: meta.ggufVariant ? `${leaf} · ${meta.ggufVariant}` : leaf,
      ggufVariant: meta.ggufVariant ?? null,
      isGguf: meta.isGguf ?? Boolean(meta.ggufVariant),
      meta,
    });
  };
  const handlePick = (id: string, meta: ModelSelectorChangeMeta) => {
    if (meta.source === "external") {
      onSelect(id, meta);
      return;
    }
    const resolved = resolveInitialConfig(id, meta.ggufVariant);
    onSelect(id, {
      ...meta,
      ...(resolved.remembered ? { config: resolved.config } : {}),
    });
  };

  return (
    <PopoverContent
      align="start"
      alignOffset={10}
      data-tour={dataTour}
      onKeyDown={handlePickerEntryKeyDown}
      className={cn(
        "unsloth-model-selector-menu menu-soft-surface ring-0 max-w-[calc(100vw-1rem)] min-w-0 gap-0",
        visibleConfigTarget
          ? "w-[min(468px,calc(100vw-1rem))] px-4 pt-4 pb-4"
          : cn(
              "pt-4 pb-0 pl-4",
              // Sized so the left-packed row keeps uniform gaps and the last
              // dropdown's right gap matches the pill's left gap (pl-4 vs pr-4).
              hasExternal
                ? "w-[min(614px,calc(100vw-1rem))] pr-4"
                : "w-[min(506px,calc(100vw-1rem))] pr-2",
            ),
        className,
      )}
    >
      {/* Local provider so popover tooltips open instantly, including when the
          cursor moves between icons. disableHoverableContent drops the grace
          area between a trigger and its tooltip, so moving from one icon to the
          next switches the tooltip at once instead of keeping the old one up. */}
      <TooltipProvider
        delayDuration={0}
        skipDelayDuration={0}
        disableHoverableContent={true}
      >
        {visibleConfigTarget ? (
          <ModelConfigPage
            key={`${visibleConfigTarget.id}::${visibleConfigTarget.ggufVariant ?? ""}`}
            target={visibleConfigTarget}
            onBack={() => setConfigTarget(null)}
            onRun={(config) =>
              onSelect(visibleConfigTarget.id, {
                ...visibleConfigTarget.meta,
                config,
                forceReload: true,
              })
            }
            loadedConfig={
              value === visibleConfigTarget.id &&
              (activeGgufVariant ?? null) ===
                (visibleConfigTarget.ggufVariant ?? null)
                ? (activeModelConfig ?? null)
                : null
            }
            loadedContextLength={
              value === visibleConfigTarget.id &&
              (activeGgufVariant ?? null) ===
                (visibleConfigTarget.ggufVariant ?? null)
                ? (activeGgufContextLength ?? null)
                : null
            }
            initialConfig={
              value === visibleConfigTarget.id &&
              (selectedGgufVariant ?? null) ===
                (visibleConfigTarget.ggufVariant ?? null)
                ? (selectedConfig ?? null)
                : null
            }
          />
        ) : (
          <>
        {tabs.length > 1 ? (
          <PillTabs
            ariaLabel="Model source"
            tabs={tabs}
            value={effectiveTab}
            onValueChange={setActiveTab}
            fit={true}
            className="mb-2"
          />
        ) : null}

        {effectiveTab === "hub" ? (
          <HubModelPicker
            models={models}
            loraModels={fineTunedModels}
            externalModels={externalModels}
            value={value}
            onSelect={handlePick}
            onFoldersChange={onFoldersChange}
            onBrowseHub={onBrowseHub}
            onModelsChange={onModelsChange}
            onConfigure={openConfigPage}
            deleteDisabled={deleteDisabled}
            onEject={hasSelection && onEject ? onEject : undefined}
            section={effectiveHubSection}
            sectionToggle={
              <PillTabs
                ariaLabel="Hub section"
                tabs={hubSectionTabs}
                value={effectiveHubSection}
                onValueChange={(next) => {
                  const section = next as HubSection;
                  setHubSection(section);
                  saveLastHubSection(section);
                }}
                fit={true}
              />
            }
          />
        ) : null}

        {effectiveTab === "external" ? (
          <ExternalModelPicker
            externalModels={externalModels}
            value={value}
            onSelect={onSelect}
          />
        ) : null}

        {onPickLocalModel ? (
          <div className="mt-1.5 border-t border-border/70 pt-1.5">
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
        {effectiveTab !== "hub" && hasSelection && onEject ? (
          <div className="mt-1.5 border-t border-border/70 pt-1.5 pb-2">
            <button
              type="button"
              onClick={onEject}
              className="flex w-full items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-destructive transition-colors hover:bg-destructive/10"
              title="Eject model"
            >
              <HugeiconsIcon icon={RemoveCircleIcon} className="size-3.5" />
              Eject loaded model
            </button>
          </div>
        ) : null}
          </>
        )}
      </TooltipProvider>
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
  activeModelConfig,
  activeGgufContextLength,
  selectedConfig,
  selectedGgufVariant,
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
  const navigate = useNavigate();
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
      return found
        ? { ...found, description: desc }
        : { id: selected, name: selected, description: desc };
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

  function handleBrowseHub() {
    setOpen(false);
    void navigate({ to: "/hub", search: { tab: "discover" } });
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
        onEject={onEject ? handleEject : undefined}
      />
      <ModelSelectorContent
        open={open}
        models={models}
        loraModels={loraModels}
        externalModels={externalModels}
        value={selected}
        activeGgufVariant={activeGgufVariant}
        activeModelConfig={activeModelConfig}
        activeGgufContextLength={activeGgufContextLength}
        selectedConfig={selectedConfig}
        selectedGgufVariant={selectedGgufVariant}
        onSelect={handleSelect}
        onEject={onEject ? handleEject : undefined}
        onFoldersChange={onFoldersChange}
        onPickLocalModel={onPickLocalModel ? handlePickLocalModel : undefined}
        onBrowseHub={handleBrowseHub}
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
          placeholder="Search models"
          className="h-9 pl-8"
        />
      </div>
      <div className="-mr-1.5 max-h-72 overflow-y-auto pr-1.5">
        <div className="space-y-2 p-1">
          {grouped.length === 0 ? (
            <div className="px-2.5 py-2 text-xs leading-relaxed text-muted-foreground">
              {externalModels.length === 0 ? (
                <>
                  No models from your connections. Set up in Settings →
                  Connections.
                </>
              ) : (
                "No models match your search."
              )}
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
