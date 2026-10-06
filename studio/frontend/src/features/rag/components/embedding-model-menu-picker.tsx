// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PinIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { DropdownMenu as DropdownMenuPrimitive } from "radix-ui";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

import {
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSubContent,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { useChatRuntimeStore } from "@/features/chat";
import {
  embeddingModelName,
  embeddingModelOwner,
  switchEmbeddingModel,
  useEmbeddingModelStore,
  useEmbeddingPinsStore,
  useSettingsDialogStore,
} from "@/features/settings";
import { useT } from "@/i18n";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { MenuTickIcon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { embeddingMenuModels } from "../lib/embedding-menu-models";

/** Space between the RAG menu and the picker beside it. */
const PICKER_GAP = 8;

/** "Embedding <model> ›" chip in the RAG menu heading. Switches between the current,
 *  default and pinned models. */
export function EmbeddingModelMenuPicker() {
  const t = useT();
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const settings = useEmbeddingModelStore((s) => s.settings);
  const pinned = useEmbeddingPinsStore((s) => s.pinned);
  const togglePin = useEmbeddingPinsStore((s) => s.togglePin);
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  const [switching, setSwitching] = useState<string | null>(null);
  const chipRef = useRef<HTMLDivElement | null>(null);
  const [offsets, setOffsets] = useState({ side: PICKER_GAP, align: 0 });
  // State, not a ref: the portal mounts the picker a render after `open` flips.
  const [picker, setPicker] = useState<HTMLDivElement | null>(null);
  // Ours, not Radix's: it closes a submenu once the pointer or focus leaves it.
  const [open, setOpen] = useState(false);

  useEffect(() => {
    void useEmbeddingModelStore.getState().load();
  }, []);

  // Stays open until a click outside the picker. The chip toggles it itself.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (picker?.contains(target) || chipRef.current?.contains(target)) return;
      setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => document.removeEventListener("pointerdown", onPointerDown, true);
  }, [open, picker]);

  // A gap clear of the menu, tops aligned: right if it fits, else left.
  useLayoutEffect(() => {
    const chip = chipRef.current;
    const menu = chip?.closest<HTMLElement>('[role="menu"]');
    if (!open || !chip || !menu || !picker) return;
    const chipBox = chip.getBoundingClientRect();
    const menuBox = menu.getBoundingClientRect();
    const width = picker.offsetWidth;
    const fitsRight = menuBox.right + PICKER_GAP + width <= window.innerWidth;
    const fitsLeft = menuBox.left - PICKER_GAP - width >= 0;
    setOffsets({
      side:
        !fitsRight && fitsLeft
          ? chipBox.left - menuBox.left + PICKER_GAP
          : menuBox.right - chipBox.right + PICKER_GAP,
      align: menuBox.top - chipBox.top,
    });
  }, [open, picker]);

  if (!settings) return null;
  const current = settings.embeddingModel;
  const models = embeddingMenuModels(current, settings.defaultEmbeddingModel, pinned);

  const pick = async (model: string) => {
    if (model === current || switching) return;
    setSwitching(model);
    const name = embeddingModelName(model);
    try {
      const result = await switchEmbeddingModel(model, hfToken || undefined);
      if (result.status === "failed") {
        toast.error(t("settings.general.rag.switchFailed"), {
          description: result.message || undefined,
        });
      } else if (result.status === "saved" && result.needsDownload) {
        toast.info(t("settings.general.rag.switchedNeedsDownload", { model: name }), {
          description: t("settings.general.rag.switchedNeedsDownloadDescription"),
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
    } finally {
      setSwitching(null);
    }
  };

  return (
    <DropdownMenuPrimitive.Sub
      open={open}
      // Close requests are ignored; see the pointerdown effect.
      onOpenChange={(next) => {
        if (next) setOpen(true);
      }}
    >
      <DropdownMenuPrimitive.SubTrigger
        ref={chipRef}
        // Opens on click or arrow key, not hover.
        onPointerMove={(event) => event.preventDefault()}
        onClick={(event) => {
          if (!open) return;
          event.preventDefault();
          setOpen(false);
        }}
        // Negative margins cancel the hover pill's padding so nothing shifts.
        className="menu-heading-chip -my-1 -mr-1.5 flex min-w-0 shrink cursor-pointer items-center gap-1 rounded-full border-0 py-1 pr-1.5 pl-2.5 text-ui-12 font-medium outline-none transition-colors"
      >
        <span className="shrink-0 text-foreground">{t("settings.general.rag.menuChip")}</span>
        <span className="min-w-0 truncate text-primary" title={current}>
          {embeddingModelName(switching ?? current)}
        </span>
        {switching ? (
          <Spinner className="size-3 shrink-0" />
        ) : (
          <HugeiconsIcon
            icon={ChevronRightStandardIcon}
            strokeWidth={1.75}
            className="-ml-0.5 size-[calc(13px*var(--ui-space-scale,1))] shrink-0 text-foreground"
          />
        )}
      </DropdownMenuPrimitive.SubTrigger>
      <DropdownMenuSubContent
        ref={setPicker}
        sideOffset={offsets.side}
        alignOffset={offsets.align}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") setOpen(false);
        }}
        className="unsloth-plus-menu w-[calc(312px*var(--ui-space-scale,1))]"
      >
        <DropdownMenuLabel className="flex items-start justify-between gap-3">
          <span className="min-w-0">{t("settings.general.rag.menuTitle")}</span>
          <DropdownMenuPrimitive.Item
            // my-0!: drops the menu item margin so it lines up with the title.
            className="my-0! shrink-0 cursor-pointer rounded-sm font-normal text-muted-foreground underline decoration-muted-foreground/50 underline-offset-[3px] outline-hidden transition-colors hover:text-foreground hover:decoration-foreground/60 data-[highlighted]:text-foreground data-[highlighted]:decoration-foreground/60"
            // Deferred past the menu's focus restore.
            onSelect={() =>
              setTimeout(
                () => openSettings("general", { scrollTarget: "general-rag-embedding" }),
                0,
              )
            }
          >
            {t("settings.general.rag.changeModel")}
          </DropdownMenuPrimitive.Item>
        </DropdownMenuLabel>
        {models.map((model) => {
          const isPinned = pinned.includes(model);
          const owner = embeddingModelOwner(model);
          const details = [
            owner || t("settings.general.rag.localModel"),
            ...(model === settings.defaultEmbeddingModel
              ? [t("settings.general.rag.defaultTag")]
              : []),
          ].join(" · ");
          return (
            <DropdownMenuItem
              key={model}
              disabled={switching !== null}
              onSelect={() => void pick(model)}
              className={cn("group/row items-start gap-2 py-2", model === current && "font-medium")}
            >
              <span className="flex min-w-0 flex-1 flex-col gap-0.5" title={model}>
                <span className="truncate text-ui-13 leading-tight">{embeddingModelName(model)}</span>
                <span className="truncate text-xs font-normal leading-snug text-muted-foreground">
                  {details}
                </span>
              </span>
              <button
                type="button"
                tabIndex={-1}
                aria-pressed={isPinned}
                aria-label={t(isPinned ? "settings.general.rag.unpin" : "settings.general.rag.pin")}
                title={t(isPinned ? "settings.general.rag.unpin" : "settings.general.rag.pin")}
                // Own the whole press, or the row selects on pointerup.
                onPointerDown={(event) => event.stopPropagation()}
                onPointerUp={(event) => event.stopPropagation()}
                onClick={(event) => {
                  event.stopPropagation();
                  togglePin(model);
                }}
                className={cn(
                  "flex size-6 shrink-0 cursor-pointer items-center justify-center self-center rounded-full transition-colors hover:bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]",
                  isPinned
                    ? "text-primary"
                    : "text-muted-foreground opacity-0 group-hover/row:opacity-100 group-data-[highlighted]/row:opacity-100",
                )}
              >
                <HugeiconsIcon icon={PinIcon} strokeWidth={1.75} className="size-3.5!" />
              </button>
              {model === current ? (
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
      </DropdownMenuSubContent>
    </DropdownMenuPrimitive.Sub>
  );
}
