// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PinIcon, PinOffIcon, RemoveCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { DropdownMenu as DropdownMenuPrimitive } from "radix-ui";
import { useEffect, useState } from "react";

import {
  DropdownMenuItem,
  DropdownMenuLabel,
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
import { useT } from "@/i18n";
import { ChevronLeftStandardIcon, ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { MenuTickIcon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { embeddingMenuModels } from "../lib/embedding-menu-models";

const HEADING_LINK_CLASS =
  "my-0! shrink-0 cursor-pointer rounded-sm font-normal text-muted-foreground underline decoration-muted-foreground/50 underline-offset-[3px] outline-hidden transition-colors hover:text-foreground hover:decoration-foreground/60 data-[highlighted]:text-foreground data-[highlighted]:decoration-foreground/60";

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
      onBack();
    } finally {
      setSwitching(null);
    }
  };

  const eject = async () => {
    if (ejecting) return;
    setEjecting(true);
    try {
      await ejectEmbeddingModel();
      toast.success(t("settings.general.rag.ejected"));
    } catch (error) {
      toast.error(t("settings.general.rag.unloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setEjecting(false);
    }
  };

  return (
    <>
      <DropdownMenuLabel className="flex items-center justify-between gap-3">
        <DropdownMenuPrimitive.Item
          aria-label={t("settings.general.rag.back")}
          onSelect={(event) => {
            event.preventDefault();
            onBack();
          }}
          className="-my-1 -ml-1.5 flex min-w-0 cursor-pointer items-center gap-1 rounded-full py-1 pr-2 pl-1 outline-hidden transition-colors hover:text-foreground data-[highlighted]:text-foreground"
        >
          <HugeiconsIcon
            icon={ChevronLeftStandardIcon}
            strokeWidth={1.75}
            className="size-[calc(13px*var(--ui-space-scale,1))] shrink-0"
          />
          <span className="truncate">{t("settings.general.rag.menuTitle")}</span>
        </DropdownMenuPrimitive.Item>
        <DropdownMenuPrimitive.Item
          className={HEADING_LINK_CLASS}
          // Deferred past the menu's focus restore.
          onSelect={() =>
            setTimeout(() => openSettings("general", { scrollTarget: "general-rag-embedding" }), 0)
          }
        >
          {t("settings.general.rag.moreModels")}
        </DropdownMenuPrimitive.Item>
      </DropdownMenuLabel>
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
        return (
          <DropdownMenuItem
            key={model}
            disabled={switching !== null && switching !== model}
            onSelect={(event) => {
              // Stay open while it switches; the list goes back once it lands.
              event.preventDefault();
              void pick(model);
            }}
            className={cn("group/row items-start gap-2 py-2", model === current && "font-medium")}
          >
            <span className="flex min-w-0 flex-1 flex-col gap-0.5">
              <span className="truncate text-ui-13 leading-tight">{embeddingModelName(model)}</span>
              <span className="truncate text-xs font-normal leading-snug text-muted-foreground">
                {details}
              </span>
            </span>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  tabIndex={-1}
                  aria-pressed={isPinned}
                  aria-label={t(isPinned ? "settings.general.rag.unpin" : "settings.general.rag.pin")}
                  // Own the whole press, or the row selects on pointerup.
                  onPointerDown={(event) => event.stopPropagation()}
                  onPointerUp={(event) => event.stopPropagation()}
                  onClick={(event) => {
                    event.stopPropagation();
                    togglePin(model);
                  }}
                  // As on Recents: grey, unpin glyph once pinned, shown on row hover only.
                  className="flex size-6 shrink-0 cursor-pointer items-center justify-center self-center rounded-full text-muted-foreground opacity-0 transition-colors group-hover/row:opacity-100 group-data-[highlighted]/row:opacity-100 hover:bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]"
                >
                  <HugeiconsIcon
                    icon={isPinned ? PinOffIcon : PinIcon}
                    strokeWidth={1.75}
                    className="size-3.5!"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent side="top">
                {t(isPinned ? "settings.general.rag.unpin" : "settings.general.rag.pin")}
              </TooltipContent>
            </Tooltip>
            {switching === model ? (
              <Spinner className="size-4 shrink-0 self-center" />
            ) : model === current ? (
              <HugeiconsIcon
                icon={MenuTickIcon}
                strokeWidth={2}
                className="permission-mode-tick size-4 shrink-0 self-center"
              />
            ) : (
              <span className="size-4 shrink-0" />
            )}
          </DropdownMenuItem>
        );
      })}
      {pinned.length === 0 ? (
        <p className="px-3 pt-1 pb-2 text-xs text-muted-foreground">
          {t("settings.general.rag.pinHint")}
        </p>
      ) : null}
      {settings.backendLoaded ? (
        <>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            disabled={switching !== null}
            onSelect={(event) => {
              // Stay open; the row goes away once the model is out.
              event.preventDefault();
              void eject();
            }}
          >
            {ejecting ? (
              <Spinner className="size-4 shrink-0" />
            ) : (
              <HugeiconsIcon icon={RemoveCircleIcon} strokeWidth={1.75} />
            )}
            {t("settings.general.rag.ejectModel")}
          </DropdownMenuItem>
        </>
      ) : null}
    </>
  );
}
