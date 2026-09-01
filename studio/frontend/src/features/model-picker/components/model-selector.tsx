// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { TooltipProvider } from "@/components/ui/tooltip";
import { isCustomProviderType } from "@/features/chat";

import type { HfTaskFilter } from "@/features/hub/hooks/use-hub-model-search";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import {
  CheckmarkCircle02Icon,
  CloudIcon,
  DashboardSquare01Icon,
  Download01Icon,
  RemoveCircleIcon,
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
  isOllamaLinkPath,
  modelDisplayName,
} from "../model-config/model-identity";
import {
  type PerModelConfig,
  resolveInitialConfig,
} from "../model-config/per-model-config";
import { ModelConfigPage } from "./model-config-page";
import {
  type ExternalConnectionRef,
  missingExternalModel,
} from "./model-selector/missing-external-model";
import type { CommunityModelPolicy } from "./model-selector/audio-picker-policy";
import type { CatalogGroup } from "./model-selector/model-catalog";
import { HubModelPicker, hasDownloadedModels } from "./model-selector/pickers";
import { PillTabs } from "./model-selector/pill-tabs";
import { loraOptionLabel } from "./model-selector/row-meta";
import { isFineTunedSource } from "./model-selector/source-tabs";
import type {
  DeletedModelRef,
  ExternalModelOption,
  LoraModelOption,
  ModelDownloadFootprintResolver,
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
export type { ExternalConnectionRef } from "./model-selector/missing-external-model";

interface ModelSelectorProps {
  models: ModelOption[];
  /** Models a task-specific runtime confirms are locally loadable even when
   * their specialized cache layout is absent from the generic Hub inventory. */
  additionalOnDeviceModels?: ModelOption[];
  /** Task-owned runtime residency when it is separate from Chat's main slot. */
  loadedModelIdOverride?: string;
  loraModels?: LoraModelOption[];
  externalModels?: ExternalModelOption[];
  /**
   * The connections behind `externalModels`, carrying each one's cached catalogue.
   * `externalModels` lists only the models the user ticked, so it cannot tell a model the
   * user turned off from one the provider withdrew; the catalogue can.
   */
  externalConnections?: ExternalConnectionRef[];
  value?: string;
  defaultValue?: string;
  /**
   * Whether the selection is actually resident. Omitted means "a selection is
   * a load", which is what every caller assumed until an image or video load
   * started evicting the chat model out from under this tick.
   */
  loaded?: boolean;
  activeGgufVariant?: string | null;
  activeModelConfig?: PerModelConfig | null;
  activeLoadedContextLength?: number | null;
  selectedConfig?: PerModelConfig | null;
  selectedGgufVariant?: string | null;
  onValueChange?: (value: string, meta: ModelSelectorChangeMeta) => void;
  /** Optional task-specific resolver for companion assets a GGUF row alone cannot describe. */
  resolveDownloadFootprint?: ModelDownloadFootprintResolver;
  onEject?: () => void;
  onFoldersChange?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
  /** Responsive text sizing for headers that have to share a constrained row. */
  triggerLabelClassName?: string;
  contentClassName?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  triggerDataTour?: string;
  contentDataTour?: string;
  showCloudIndicator?: boolean;
  /** Restrict the Hub tab to a pipeline task (e.g. text-to-image). */
  task?: HfTaskFilter;
  /** Canonical model groups (Images / Video pages): collapses a model's artifact repos into one row with a format second level and device-aware routing. Undefined (chat) changes nothing. */
  catalog?: CatalogGroup[];
  /** Also list community (non-unsloth) models for `task`. Opt-in: only pages
   *  whose runtime loads arbitrary publishers. */
  communityModelPolicy?: CommunityModelPolicy;
  /** Trigger text when nothing is loaded. Defaults to "Select model"; task pages name what they pick so it reads as separate from the chat model. */
  placeholder?: string;
}

function ModelSelectorTrigger({
  currentModel,
  isLoaded,
  showCloudIndicator = false,
  variant = "outline",
  size = "default",
  className,
  triggerLabelClassName,
  dataTour,
  onEject,
  // Task pages name what they pick ("Select image model"), so the choice reads as separate from the chat model.
  placeholder = "Select model",
}: {
  currentModel?: ModelOption;
  isLoaded: boolean;
  showCloudIndicator?: boolean;
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
  triggerLabelClassName?: string;
  dataTour?: string;
  onEject?: () => void;
  placeholder?: string;
}) {
  return (
    <PopoverTrigger asChild={true}>
      <button
        type="button"
        data-tour={dataTour}
        className={cn(
          "unsloth-model-selector-trigger group/trigger flex min-w-0 items-center gap-2 transition-colors",
          // Suppress the pill's hover background while the eject hit area is hovered.
          variant === "outline" &&
            "rounded-full border border-border/60 hover:bg-accent has-[[data-eject-hit]:hover]:!bg-transparent",
          variant === "ghost" &&
            "rounded-full hover:bg-accent has-[[data-eject-hit]:hover]:!bg-transparent",
          variant === "muted" &&
            "rounded-full bg-muted hover:bg-muted/80 has-[[data-eject-hit]:hover]:!bg-muted",
          // More left padding than right; the chevron is pulled close to the label so the trigger reads
          // balanced around the text. Height stays pinned to --studio-chat-control-height.
          size === "sm" && "h-8 pl-3 pr-1.5 text-xs",
          size === "default" && "h-9 pl-4 pr-2 text-sm",
          size === "lg" && "h-10 pl-4.5 pr-2.5 text-sm",
          className,
        )}
      >
        {isLoaded &&
          (onEject ? (
            // Loaded status doubles as a mouse eject shortcut (checkmark at rest, eject on hover). A plain
            // span keeps it out of the trigger button's content model; keyboard/SR users eject via the
            // "Eject model" button. stopPropagation stops the popover toggling, and on touch (no hover)
            // pointer-events-none disables it so taps open the picker instead.
            <span
              aria-hidden={true}
              title="Eject model"
              data-eject-hit={true}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                onEject();
              }}
              // Hit area larger than the icon, with a hover circle; negative margin keeps the icon in place.
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
          <span
            className={cn(
              "min-w-0 flex flex-1 items-baseline truncate font-heading text-ui-16 font-medium leading-tight text-black dark:text-white",
              triggerLabelClassName,
            )}
          >
            {currentModel?.name ?? placeholder}
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

type HubSection = "downloaded" | "recommended" | "connected";

// The user's most recently clicked Hub section, restored on every open.
const HUB_SECTION_KEY = "unsloth_model_selector_section";
// Last tab the user actually clicked, or null. Only On Device / Recommended persist.
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
// Default the Hub section: the last tab clicked; first time, On Device with downloads else Recommended.
function defaultHubSection(hasAdditionalOnDeviceModels = false): HubSection {
  return (
    loadLastHubSection() ??
    (hasDownloadedModels() || hasAdditionalOnDeviceModels
      ? "downloaded"
      : "recommended")
  );
}

const HUB_SECTION_TABS: { value: string; label: string; icon?: ReactNode }[] = [
  {
    value: "recommended",
    label: "Recommended",
    icon: <HugeiconsIcon icon={StarIcon} className="size-3.5 shrink-0" />,
  },
  {
    value: "downloaded",
    label: "On Device",
    icon: <HugeiconsIcon icon={Download01Icon} className="size-3.5 shrink-0" />,
  },
];

function ModelSelectorContent({
  open,
  models,
  additionalOnDeviceModels,
  loadedModelIdOverride,
  loraModels,
  externalModels,
  value,
  activeGgufVariant,
  activeModelConfig,
  activeLoadedContextLength,
  selectedConfig,
  selectedGgufVariant,
  onSelect,
  resolveDownloadFootprint,
  onEject,
  onFoldersChange,
  onBrowseHub,
  onModelsChange,
  deleteDisabled,
  className,
  dataTour,
  task,
  catalog,
  communityModelPolicy,
}: {
  open: boolean;
  models: ModelOption[];
  additionalOnDeviceModels?: ModelOption[];
  loadedModelIdOverride?: string;
  loraModels: LoraModelOption[];
  externalModels: ExternalModelOption[];
  value?: string;
  activeGgufVariant?: string | null;
  activeModelConfig?: PerModelConfig | null;
  activeLoadedContextLength?: number | null;
  selectedConfig?: PerModelConfig | null;
  selectedGgufVariant?: string | null;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  resolveDownloadFootprint?: ModelDownloadFootprintResolver;
  onEject?: () => void;
  onFoldersChange?: () => void;
  onBrowseHub?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  className?: string;
  dataTour?: string;
  task?: HfTaskFilter;
  catalog?: CatalogGroup[];
  communityModelPolicy?: CommunityModelPolicy;
}) {
  const t = useT();
  const hasSelection = Boolean(value);
  const hasExternal = externalModels.length > 0;
  // The Fine-tuned tab is for fine-tuned models only; local models (LM Studio, Ollama, custom folders) live in Hub.
  const fineTunedModels = useMemo(
    () => loraModels.filter((model) => isFineTunedSource(model.source)),
    [loraModels],

  );
  // Connected sits in the section toggle, shown only with external providers.
  const hubSectionTabs = useMemo(
    () =>
      hasExternal
        ? [
            ...HUB_SECTION_TABS,
            {
              value: "connected",
              label: "Connected",
              icon: (
                <HugeiconsIcon icon={CloudIcon} className="size-3.5 shrink-0" />
              ),
            },
          ]
        : HUB_SECTION_TABS,
    [hasExternal],

  );
  const wantsConnectedDefault = Boolean(
    value && externalModels.some((model) => model.id === value),
  );
  const hasAdditionalOnDeviceModels =
    (additionalOnDeviceModels?.length ?? 0) > 0;
  const [hubSection, setHubSection] = useState<HubSection>(() =>
    wantsConnectedDefault
      ? "connected"
      : defaultHubSection(hasAdditionalOnDeviceModels),
  );
  // Connected is only valid while external providers exist; fall back otherwise.
  const effectiveHubSection: HubSection =
    hubSection === "connected" && !hasExternal ? "recommended" : hubSection;

  const [configTarget, setConfigTarget] = useState<ModelPickTarget | null>(
    null,
  );

  // The picker remounts on each open but this section state does not, so
  // re-derive the default section on the open edge.
  const wasOpen = useRef(open);
  useEffect(() => {
    if (open && !wasOpen.current) {
      setHubSection(
        wantsConnectedDefault
          ? "connected"
          : defaultHubSection(hasAdditionalOnDeviceModels),
      );
    }
    if (!open && wasOpen.current) {
      setConfigTarget(null);
    }
    wasOpen.current = open;
  }, [open, wantsConnectedDefault, hasAdditionalOnDeviceModels]);

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
    const isGguf = meta.isGguf ?? Boolean(meta.ggufVariant);
    setConfigTarget({
      id,
      displayName: meta.ggufVariant ? `${leaf} · ${meta.ggufVariant}` : leaf,
      ggufVariant: meta.ggufVariant ?? null,
      isGguf,
      // Ollama's models sit under a link dir the resolver skips, so mirroring their settings would advertise an impossible load.
      apiLoadable: isGguf && !isOllamaLinkPath(id),
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
          ? "max-h-[var(--radix-popover-content-available-height)] w-[min(468px,calc(100vw-1rem))] overflow-y-auto px-4 pt-4 pb-4"
          : cn(
              "pt-4 pb-0 pl-4",
              // Sized so the left-packed row keeps uniform gaps and the last dropdown's right gap matches the
              // pill's left gap. Widths track the controls they hold so the single-line row does not wrap.
              hasExternal
                ? "w-[min(var(--picker-panel-w-external),calc(100vw-1rem))] pr-4"
                : "w-[min(var(--picker-panel-w),calc(100vw-1rem))] pr-2",
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
            onRun={(config, isDiffusion) =>
              onSelect(visibleConfigTarget.id, {
                ...visibleConfigTarget.meta,
                config,
                isDiffusion,
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
                ? (activeLoadedContextLength ?? null)
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
            <HubModelPicker
              models={models}
              additionalOnDeviceModels={additionalOnDeviceModels}
              loadedModelIdOverride={loadedModelIdOverride}
              loraModels={fineTunedModels}
              externalModels={externalModels}
              value={value}
              onSelect={handlePick}
              resolveDownloadFootprint={resolveDownloadFootprint}
              onFoldersChange={onFoldersChange}
              onBrowseHub={onBrowseHub}
              onModelsChange={onModelsChange}
              onConfigure={openConfigPage}
              deleteDisabled={deleteDisabled}
              onEject={hasSelection && onEject ? onEject : undefined}
              task={task}
              catalog={catalog}
              communityModelPolicy={communityModelPolicy}
              section={effectiveHubSection}
              sectionToggle={
                <PillTabs
                  ariaLabel={t("picker.hubSectionAriaLabel")}
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
          </>
        )}
      </TooltipProvider>
    </PopoverContent>
  );
}

export function ModelSelector({
  models,
  additionalOnDeviceModels = [],
  loadedModelIdOverride,
  loraModels = [],
  externalModels = [],
  externalConnections = [],
  value,
  defaultValue,
  activeGgufVariant,
  activeModelConfig,
  activeLoadedContextLength,
  selectedConfig,
  selectedGgufVariant,
  onValueChange,
  resolveDownloadFootprint,
  onEject,
  onFoldersChange,
  onModelsChange,
  deleteDisabled,
  variant = "outline",
  size = "default",
  className,
  triggerLabelClassName,
  contentClassName,
  open: controlledOpen,
  onOpenChange,
  triggerDataTour,
  contentDataTour,
  showCloudIndicator = false,
  task,
  catalog,
  communityModelPolicy = "none",
  placeholder,
  loaded,
}: ModelSelectorProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = onOpenChange ?? setUncontrolledOpen;
  const navigate = useNavigate();
  const t = useT();
  const [uncontrolled, setUncontrolled] = useState(defaultValue ?? "");

  const selected = value ?? uncontrolled;
  // A selection is only a load when the caller has not said otherwise: the chat
  // model can be evicted by an image or video load while the pick survives.
  const isLoaded = selected !== "" && (loaded ?? true);

  const optionById = useMemo(() => {
    const all = new Map<string, ModelOption>();
    for (const model of models) {
      all.set(model.id, model);
    }
    for (const lora of loraModels) {
      const displayName = loraOptionLabel(lora);
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
    // A pick whose connection no longer offers it takes its option away and leaves the
    // id in the checkpoint, and the generic fallback below cannot shorten an
    // `external::` id. Name the model the user picked, and say why it is unusable: it
    // cannot be loaded, so a tidy name on its own would hide the failure until the next
    // send (#8405). The connections carry the cached catalogue, which is what separates a
    // model the user unticked from one the provider withdrew.
    const missingExternal = found
      ? null
      : missingExternalModel(selected, externalModels, externalConnections);
    // No catalog entry (yet, or ever); a cached GGUF's checkpoint is a snapshot path.
    // The leaf, not the namespaced public id (#7966), matches the catalog row that
    // later replaces this one.
    const fallbackName = missingExternal?.modelName ?? modelDisplayName(selected);
    if (activeGgufVariant) {
      const desc = `GGUF · ${activeGgufVariant}`;
      return found
        ? { ...found, description: desc }
        : { id: selected, name: fallbackName, description: desc };
    }
    if (missingExternal) {
      const disabled = missingExternal.state === "disabled";
      return {
        id: selected,
        name: fallbackName,
        description: missingExternal.providerName
          ? t(
              disabled
                ? "picker.modelDisabledByProvider"
                : "picker.modelDroppedByProvider",
              { provider: missingExternal.providerName },
            )
          : t(disabled ? "picker.modelDisabled" : "picker.modelDropped"),
      };
    }
    return found ?? { id: selected, name: fallbackName };
  }, [
    selected,
    optionById,
    activeGgufVariant,
    externalModels,
    externalConnections,
    t,
  ]);

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
        triggerLabelClassName={triggerLabelClassName}
        dataTour={triggerDataTour}
        onEject={onEject ? handleEject : undefined}
        placeholder={placeholder}
      />
      <ModelSelectorContent
        open={open}
        models={models}
        additionalOnDeviceModels={additionalOnDeviceModels}
        loadedModelIdOverride={loadedModelIdOverride}
        loraModels={loraModels}
        externalModels={externalModels}
        value={selected}
        activeGgufVariant={activeGgufVariant}
        activeModelConfig={activeModelConfig}
        activeLoadedContextLength={activeLoadedContextLength}
        selectedConfig={selectedConfig}
        selectedGgufVariant={selectedGgufVariant}
        onSelect={handleSelect}
        resolveDownloadFootprint={resolveDownloadFootprint}
        onEject={onEject ? handleEject : undefined}
        onFoldersChange={onFoldersChange}
        // A curated task picker (Images / Video) is self-contained, so it omits this.
        // A community-enabled one (Audio) already lists past unsloth, so it keeps it.
        onBrowseHub={
          task && communityModelPolicy === "none" ? undefined : handleBrowseHub
        }
        onModelsChange={onModelsChange}
        deleteDisabled={deleteDisabled}
        className={contentClassName}
        dataTour={contentDataTour}
        task={task}
        catalog={catalog}
        communityModelPolicy={communityModelPolicy}
      />
    </Popover>
  );
}

ModelSelector.Trigger = ModelSelectorTrigger;
ModelSelector.Content = ModelSelectorContent;
