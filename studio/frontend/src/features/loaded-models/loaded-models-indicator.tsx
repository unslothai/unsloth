// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A loaded model only shows on the page that loaded it, so memory stays held with
// nothing on screen saying so. This card lists what is resident from anywhere and
// ejects it in place. It joins the shared bottom-right stack instead of pinning
// itself, so the update banners and download panel never overlap it.

import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { hasAuthToken, mustChangePassword } from "@/features/auth";
import { useSettingsDialogStore } from "@/features/settings";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import {
  AiBrain01Icon,
  ArrowExpandDiagonal01Icon,
  Cancel01Icon,
  DragDropVerticalIcon,
  Image01Icon,
  Message01Icon,
  Mic01Icon,
  Video01Icon,
  VolumeHighIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useCallback } from "react";
import {
  LOADED_MODEL_KIND_LABELS,
  type LoadedModelEntry,
  type LoadedModelKind,
  loadedModelTarget,
  shortModelLabel,
} from "./loaded-models-sources";
import {
  LOADED_MODELS_PREFERENCE_KEYS,
  useShowLoadedModels,
} from "./show-loaded-models-pref";
import { useDragPosition } from "./use-drag-position";
import { useLoadedModels } from "./use-loaded-models";

// Collapsing to the pill is deliberate, so it survives reloads. Expanded by
// default: a card you have to open first answers nothing.
const COLLAPSED_KEY = LOADED_MODELS_PREFERENCE_KEYS.collapsed;

const KIND_ICONS: Record<LoadedModelKind, typeof AiBrain01Icon> = {
  text: Message01Icon,
  tts: VolumeHighIcon,
  image: Image01Icon,
  video: Video01Icon,
  stt: Mic01Icon,
};

// Nothing to report on the auth and onboarding screens. Desktop auto-authenticates,
// so only the browser needs the token check: polling before one exists is all 401s.
const HIDDEN_ROUTES = new Set([
  "/login",
  "/signup",
  "/change-password",
  "/onboarding",
]);

function canShowIndicator(pathname: string): boolean {
  if (HIDDEN_ROUTES.has(pathname)) return false;
  if (isTauri) return true;
  return hasAuthToken() && !mustChangePassword();
}

function rowSubtitle(entry: LoadedModelEntry): string {
  const kind = LOADED_MODEL_KIND_LABELS[entry.kind];
  return entry.detail ? `${kind} · ${entry.detail}` : kind;
}

function LoadedModelRow({
  entry,
  ejecting,
  onEject,
  onOpen,
}: {
  entry: LoadedModelEntry;
  ejecting: boolean;
  onEject: () => void;
  onOpen: () => void;
}) {
  const label = shortModelLabel(entry.name);
  const target = loadedModelTarget(entry.source);
  return (
    <div className="flex items-center gap-2 rounded-[14px] px-1.5 py-1 transition-colors hover:bg-foreground/[0.04]">
      {/* Only the label half is the link: the eject button cannot nest inside it. */}
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label={`${label}. Open ${target.label}`}
            onClick={onOpen}
            className="flex min-w-0 flex-1 items-center gap-2 text-left"
          >
            <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground/[0.05] text-muted-foreground">
              <HugeiconsIcon
                icon={KIND_ICONS[entry.kind]}
                strokeWidth={1.75}
                className="size-[15px]"
              />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-ui-12p5 font-medium text-foreground">
                {label}
              </span>
              <span className="block truncate text-ui-11 text-muted-foreground">
                {rowSubtitle(entry)}
              </span>
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="left" sideOffset={6}>
          <span className="block">{entry.name}</span>
          <span className="block text-muted-foreground">
            Open {target.label}
          </span>
        </TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label={`Eject ${label}`}
            disabled={ejecting}
            onClick={onEject}
            className="flex size-6 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-foreground/[0.07] hover:text-foreground disabled:pointer-events-none disabled:opacity-60"
          >
            {ejecting ? (
              <Spinner className="size-3.5" label="Ejecting" />
            ) : (
              <HugeiconsIcon
                icon={Cancel01Icon}
                strokeWidth={2}
                className="size-3.5"
              />
            )}
          </button>
        </TooltipTrigger>
        <TooltipContent side="left" sideOffset={6}>
          Eject to free memory
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

export function LoadedModelsIndicator({
  positioned = true,
}: { positioned?: boolean } = {}) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const showIndicator = useShowLoadedModels();
  const enabled = showIndicator && canShowIndicator(pathname);
  const { entries, ejecting, eject } = useLoadedModels(enabled);
  const [collapsed, setCollapsed] = usePersistedToggle(COLLAPSED_KEY);
  const navigate = useNavigate();
  const openEntry = useCallback(
    (entry: LoadedModelEntry) => {
      const target = loadedModelTarget(entry.source);
      if (target.open === "settings") {
        // Read on click, not at render: the settings barrel reaches back here
        // through the General tab, so the binding is only safe once both
        // modules have finished evaluating. The /settings route is no use, it
        // redirects home and would take the user off the page they are on.
        useSettingsDialogStore.getState().openDialog(target.tab);
        return;
      }
      // No search params: this only takes the user to the page, it does not
      // start a new thread or reload anything.
      void navigate({ to: target.to });
    },
    [navigate],
  );
  const {
    position,
    size,
    panelRef,
    startDrag,
    startResize,
    dragging,
    resizing,
    reset,
    justDragged,
  } = useDragPosition(LOADED_MODELS_PREFERENCE_KEYS);

  if (!enabled || entries.length === 0) return null;

  const countLabel = `${entries.length} ${entries.length === 1 ? "model" : "models"} loaded`;
  // The pill keeps its own shape, so a resized card does not stretch it.
  const sized = size && !collapsed ? size : null;

  return (
    <div
      ref={panelRef}
      className={cn(
        // Dragged: pinned where the user left it, out of the stack's flow.
        // Otherwise anchored bottom-right, or flowing as a right-aligned row
        // in the shared stack so overlays stack instead of overlapping.
        "pointer-events-none",
        position && "fixed z-[9999]",
        position && !sized && "w-fit",
        !position &&
          (positioned
            ? "fixed bottom-4 right-4 z-50"
            : "flex min-h-0 justify-end"),
        (dragging || resizing) && "select-none",
      )}
      style={{
        ...(position ? { left: position.left, top: position.top } : null),
        ...sized,
      }}
    >
      {collapsed ? (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={`${countLabel}. Show details, or drag to move`}
              onPointerDown={startDrag}
              // The pill is its own drag handle, so a press that moved is a
              // drag and must not also expand the card.
              onClick={() => {
                if (!justDragged()) setCollapsed(false);
              }}
              className="menu-soft-surface pointer-events-auto flex h-9 cursor-grab touch-none items-center gap-1.5 rounded-full pl-2.5 pr-3 font-heading text-muted-foreground transition-colors hover:text-foreground active:cursor-grabbing"
            >
              <HugeiconsIcon
                icon={AiBrain01Icon}
                strokeWidth={1.75}
                className="size-[15px]"
              />
              <span className="text-ui-12p5 font-medium tabular-nums">
                {entries.length}
              </span>
            </button>
          </TooltipTrigger>
          <TooltipContent side="left" sideOffset={6}>
            {countLabel}
          </TooltipContent>
        </Tooltip>
      ) : (
        <div
          className={cn(
            "menu-soft-surface group pointer-events-auto flex min-h-0 flex-col overflow-hidden rounded-[20px] p-1.5 font-heading",
            // Sized: the wrapper already holds the clamped box, so filling it
            // keeps the held corner exactly where the resize put it.
            sized ? "size-full" : "w-[268px] max-w-[calc(100vw-2rem)]",
          )}
        >
          <div className="flex items-center gap-1.5 px-1.5 pb-1 pt-0.5">
            {/* The card is anchored bottom-right, where there is no room, so it
                grows up and left and the grip belongs at the leading corner.
                It shares the title icon's slot rather than adding another
                control to an already small header. */}
            <div className="relative flex size-[15px] shrink-0 items-center justify-center">
              <HugeiconsIcon
                icon={AiBrain01Icon}
                strokeWidth={1.75}
                className="size-[15px] text-muted-foreground transition-opacity group-hover:opacity-0"
              />
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <div
                    aria-label="Drag to resize"
                    onPointerDown={startResize}
                    onDoubleClick={reset}
                    className="absolute -inset-1.5 flex cursor-nwse-resize touch-none items-center justify-center rounded-full text-muted-foreground opacity-0 transition-opacity hover:bg-foreground/[0.07] hover:text-foreground group-hover:opacity-100"
                  >
                    <HugeiconsIcon
                      icon={ArrowExpandDiagonal01Icon}
                      strokeWidth={2}
                      className="size-3.5"
                    />
                  </div>
                </TooltipTrigger>
                <TooltipContent side="left" sideOffset={6}>
                  Drag to resize, double-click to reset
                </TooltipContent>
              </Tooltip>
            </div>
            <span className="min-w-0 flex-1 truncate text-ui-12p5 font-semibold text-foreground">
              Loaded models
            </span>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <div
                  aria-label="Drag to move"
                  onPointerDown={startDrag}
                  className="flex size-6 shrink-0 cursor-grab touch-none items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:bg-foreground/[0.07] hover:text-foreground active:cursor-grabbing"
                >
                  <HugeiconsIcon
                    icon={DragDropVerticalIcon}
                    strokeWidth={1.75}
                    className="size-3.5"
                  />
                </div>
              </TooltipTrigger>
              <TooltipContent side="left" sideOffset={6}>
                Drag to move
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  aria-label="Collapse loaded models"
                  onClick={() => setCollapsed(true)}
                  className="flex size-6 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-foreground/[0.07] hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={ChevronDownStandardIcon}
                    strokeWidth={1.75}
                    className="size-3.5"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent side="left" sideOffset={6}>
                Collapse
              </TooltipContent>
            </Tooltip>
          </div>
          {/* Capped so four resident runtimes still leave the banners on screen,
              unless the user has given the card a height of their own. */}
          <div
            className={cn(
              "flex min-h-0 flex-col gap-0.5 overflow-y-auto",
              sized ? "flex-1" : "max-h-[min(272px,42dvh)]",
            )}
          >
            {entries.map((entry) => (
              <LoadedModelRow
                key={entry.id}
                entry={entry}
                ejecting={ejecting.has(entry.id)}
                onEject={() => void eject(entry)}
                onOpen={() => openEntry(entry)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
