// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PinIcon, PinOffIcon, RemoveCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { DropdownMenu as DropdownMenuPrimitive } from "radix-ui";
import { useEffect, useRef, useState } from "react";

import {
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useChatRuntimeStore } from "@/features/chat";
import {
  ejectEmbeddingModel,
  embeddingModelName,
  embeddingModelOwner,
  switchEmbeddingModel,
  useEmbeddingModelStore,
  useEmbeddingPinsStore,
  useSettingsDialogStore,
} from "@/features/settings";
import { useWheelScrollRef } from "@/hooks";
import { useT } from "@/i18n";
import { ChevronLeftStandardIcon, ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { MenuTickIcon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { embeddingMenuModels } from "../lib/embedding-menu-models";

const HEADING_LINK_CLASS =
  "my-0! shrink-0 cursor-pointer rounded-sm font-normal text-muted-foreground underline decoration-muted-foreground/50 underline-offset-[3px] outline-hidden transition-colors hover:text-foreground hover:decoration-foreground/60 data-[highlighted]:text-foreground data-[highlighted]:decoration-foreground/60";

// Fixed width, so the pin sits in the same place on every row, ticked or not.
const TRAILING_SLOT_CLASS =
  "flex w-[calc(var(--ui-icon-size)+2px)] shrink-0 items-center justify-center";

/** The model item of the row after (else before) the one holding `el`. */
function neighbourRowItem(el: HTMLElement): HTMLElement | null {
  const row = el.closest(".menu-row-with-action");
  for (const step of ["nextElementSibling", "previousElementSibling"] as const) {
    for (let next = row?.[step]; next; next = next[step]) {
      if (next.classList.contains("menu-row-with-action"))
        return next.querySelector<HTMLElement>('[role^="menuitem"]');
    }
  }
  return null;
}

/** "Model <name> ›" chip in the RAG menu heading. Swaps the menu to the model list. */
export function EmbeddingModelMenuChip({ onOpen }: { onOpen: () => void }) {
  const t = useT();
  const settings = useEmbeddingModelStore((s) => s.settings);

  useEffect(() => {
    void useEmbeddingModelStore.getState().load();
  }, []);

  if (!settings) return null;
  const current = settings.embeddingModel;
  return (
    <DropdownMenuPrimitive.Item
      onSelect={(event) => {
        // Keep the menu open; it shows the list instead.
        event.preventDefault();
        onOpen();
      }}
      // Pill edge stays put; slim right padding sets the text close to it.
      className="menu-heading-chip -my-1 -mr-2 flex min-w-0 shrink cursor-pointer items-center gap-1 rounded-full border-0 py-1 pr-1 pl-2.5 text-ui-12 font-medium outline-none transition-colors"
    >
      <span className="shrink-0 text-foreground">{t("settings.general.rag.menuChip")}</span>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <span className="min-w-0 truncate text-primary">{embeddingModelName(current)}</span>
        </TooltipTrigger>
        <TooltipContent side="top">{current}</TooltipContent>
      </Tooltip>
      <HugeiconsIcon
        icon={ChevronRightStandardIcon}
        strokeWidth={1.75}
        className="-ml-0.5 size-[calc(13px*var(--ui-space-scale,1))] shrink-0 text-foreground"
      />
    </DropdownMenuPrimitive.Item>
  );
}

/** Model list shown in place of the RAG menu: current, default and pinned models. */
export function EmbeddingModelMenuList({ onBack }: { onBack: () => void }) {
  const t = useT();
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const settings = useEmbeddingModelStore((s) => s.settings);
  const pinned = useEmbeddingPinsStore((s) => s.pinned);
  const togglePin = useEmbeddingPinsStore((s) => s.togglePin);
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  const [switching, setSwitching] = useState<string | null>(null);
  const [ejecting, setEjecting] = useState(false);
  // A slow switch can land after the menu closed and reopened; only the live list may navigate.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);
  const listRef = useWheelScrollRef<HTMLDivElement>();

  // Fresh residency, so Eject reflects a model indexing just loaded.
  useEffect(() => {
    void useEmbeddingModelStore.getState().load();
  }, []);

  if (!settings) return null;
  const current = settings.embeddingModel;
  const models = embeddingMenuModels(current, settings.defaultEmbeddingModel, pinned);

  const pick = async (model: string) => {
    if (model === current) {
      onBack();
      return;
    }
    if (switching) return;
    setSwitching(model);
    const name = embeddingModelName(model);
    try {
      const result = await switchEmbeddingModel(model, hfToken || undefined);
      if (result.status === "failed") {
        toast.error(t("settings.general.rag.switchFailed"), {
          description: result.message || undefined,
        });
        return;
      }
      if (result.status === "saved" && result.needsDownload) {
        toast.info(t("settings.general.rag.switchedNeedsDownload", { model: name }), {
          description: t("settings.general.rag.switchedNeedsDownloadDescription"),
          action: {
            label: t("settings.general.rag.openSettings"),
            onClick: () => openSettings("general", { scrollTarget: "general-rag-embedding" }),
          },
        });
      } else if (result.status === "saved" && result.needsDownload === null) {
        // Saved, but the resolve failed, so whether the files are on disk is unknown.
        toast.info(t("settings.general.rag.switched", { model: name }), {
          description: t("settings.general.rag.switchedUncheckedDescription"),
          action: {
            label: t("settings.general.rag.openSettings"),
            onClick: () => openSettings("general", { scrollTarget: "general-rag-embedding" }),
          },
        });
      } else if (result.status === "saved") {
        toast.success(t("settings.general.rag.switched", { model: name }), {
          description: t("settings.general.rag.reindexWarning"),
        });
      }
      if (mountedRef.current) onBack();
    } finally {
      if (mountedRef.current) setSwitching(null);
    }
  };

  const eject = async (menu: Element | null) => {
    if (ejecting) return;
    setEjecting(true);
    try {
      await ejectEmbeddingModel();
      toast.success(t("settings.general.rag.ejected"));
      // The Eject row goes with the model and took focus with it; land on the first model row.
      requestAnimationFrame(() =>
        menu?.querySelector<HTMLElement>('.menu-row-with-action [role^="menuitem"]')?.focus(),
      );
    } catch (error) {
      toast.error(t("settings.general.rag.unloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setEjecting(false);
    }
  };

  return (
    // Capped to the menu's room, so the rows scroll inside it instead of the whole menu.
    <div className="flex max-h-[calc(var(--radix-dropdown-menu-content-available-height)-1rem*var(--ui-space-scale,1))] flex-col">
      {/* Header and Eject stay put; only the model rows scroll. */}
      {/* A page header, not a label: the back button returns to the source list, and the title
          names this page, so the arrow never reads as "go back to <title>". */}
      <div className="flex shrink-0 items-start gap-2 px-2 pt-2 pb-2">
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <DropdownMenuPrimitive.Item
              aria-label={t("settings.general.rag.back")}
              onSelect={(event) => {
                event.preventDefault();
                onBack();
              }}
              className="-ml-1 flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground outline-hidden transition-colors hover:bg-[rgb(0_0_0_/_calc(0.1*var(--contrast-wash-gain,1)))] hover:text-foreground data-[highlighted]:bg-[rgb(0_0_0_/_calc(0.1*var(--contrast-wash-gain,1)))] data-[highlighted]:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.14*var(--contrast-wash-gain,1)))] dark:data-[highlighted]:bg-[rgb(255_255_255_/_calc(0.14*var(--contrast-wash-gain,1)))]"
            >
              <HugeiconsIcon
                icon={ChevronLeftStandardIcon}
                strokeWidth={2}
                className="size-[calc(14px*var(--ui-space-scale,1))]"
              />
            </DropdownMenuPrimitive.Item>
          </TooltipTrigger>
          <TooltipContent side="top">{t("settings.general.rag.back")}</TooltipContent>
        </Tooltip>
        <div className="flex min-w-0 flex-1 flex-col">
          <span className="truncate text-ui-12 font-medium leading-tight text-foreground">
            {t("settings.general.rag.embeddingModel")}
          </span>
          <span className="truncate text-xs leading-snug text-muted-foreground">
            {t("settings.general.rag.menuSubtitle")}
          </span>
        </div>
        <DropdownMenuPrimitive.Item
          className={cn(HEADING_LINK_CLASS, "mt-px text-ui-12")}
          // Deferred past the menu's focus restore.
          onSelect={() =>
            setTimeout(() => openSettings("general", { scrollTarget: "general-rag-embedding" }), 0)
          }
        >
          {t("settings.general.rag.moreModels")}
        </DropdownMenuPrimitive.Item>
      </div>
      <div
        ref={listRef}
        className="min-h-0 flex-1 overflow-y-auto max-h-[calc(236px*var(--ui-space-scale,1))]"
      >
        {models.map((model) => {
          const isPinned = pinned.includes(model);
          const owner = embeddingModelOwner(model);
          const details = [
            owner || t("settings.general.rag.localModel"),
            ...(model === settings.defaultEmbeddingModel ? [t("settings.general.rag.defaultTag")] : []),
            ...(model === current
              ? [t(settings.loaded ? "settings.general.rag.loaded" : "settings.general.rag.notLoaded")]
              : []),
          ].join(" · ");
          const pinLabel = t(isPinned ? "settings.general.rag.unpin" : "settings.general.rag.pin");
          return (
            // The pin is its own menu item, so arrow keys reach it; it is laid over the row's spacer.
            <div key={model} className="menu-row-with-action group/row relative">
              <DropdownMenuItem
                disabled={switching !== null && switching !== model}
                onSelect={(event) => {
                  // Stay open while it switches; the list goes back once it lands.
                  event.preventDefault();
                  void pick(model);
                }}
                className={cn("items-start gap-2 py-2", model === current && "font-medium")}
              >
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="truncate text-ui-13 leading-tight">{embeddingModelName(model)}</span>
                  <span className="truncate text-xs font-normal leading-snug text-muted-foreground">
                    {details}
                  </span>
                </span>
                <span className="flex shrink-0 items-center gap-2 self-center">
                  <span className="size-6" />
                  <span className={TRAILING_SLOT_CLASS}>
                    {switching === model ? (
                      <Spinner className="size-4" />
                    ) : model === current ? (
                      <HugeiconsIcon
                        icon={MenuTickIcon}
                        strokeWidth={2}
                        className="permission-mode-tick"
                      />
                    ) : null}
                  </span>
                </span>
              </DropdownMenuItem>
              <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild={true}>
                    <DropdownMenuPrimitive.CheckboxItem
                      checked={isPinned}
                      aria-label={pinLabel}
                      data-row-action={true}
                      onSelect={(event) => {
                        event.preventDefault();
                        // Unpinning an extra row removes the focused item; hand focus to a neighbour.
                        const leaving =
                          isPinned && model !== current && model !== settings.defaultEmbeddingModel;
                        const neighbour = leaving
                          ? neighbourRowItem(event.currentTarget as HTMLElement)
                          : null;
                        togglePin(model);
                        if (neighbour) requestAnimationFrame(() => neighbour.focus());
                      }}
                      // As on Recents: grey, unpin glyph once pinned. Shown on row hover or focus, always on touch.
                      className="pointer-events-auto flex size-6 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground opacity-0 outline-hidden transition-colors group-hover/row:opacity-100 group-has-[[data-highlighted]]/row:opacity-100 [@media(hover:none)]:opacity-100 hover:bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] data-[highlighted]:bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:data-[highlighted]:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
                    >
                      <HugeiconsIcon
                        icon={isPinned ? PinOffIcon : PinIcon}
                        strokeWidth={1.75}
                        className="size-3.5!"
                      />
                    </DropdownMenuPrimitive.CheckboxItem>
                  </TooltipTrigger>
                  <TooltipContent side="top">{pinLabel}</TooltipContent>
                </Tooltip>
                <span className={TRAILING_SLOT_CLASS} />
              </span>
            </div>
          );
        })}
        {pinned.length === 0 ? (
          <p className="px-3 pt-1 pb-2 text-xs text-muted-foreground">
            {t("settings.general.rag.pinHint")}
          </p>
        ) : null}
      </div>
      {settings.backendLoaded ? (
        <div className="shrink-0">
          <DropdownMenuSeparator />
          <DropdownMenuItem
            disabled={switching !== null}
            onSelect={(event) => {
              // Stay open; the row goes away once the model is out.
              event.preventDefault();
              void eject((event.currentTarget as HTMLElement).closest('[role="menu"]'));
            }}
          >
            {ejecting ? (
              <Spinner className="size-4 shrink-0" />
            ) : (
              <HugeiconsIcon icon={RemoveCircleIcon} strokeWidth={1.75} />
            )}
            {t("settings.general.rag.ejectModel")}
          </DropdownMenuItem>
        </div>
      ) : null}
    </div>
  );
}
