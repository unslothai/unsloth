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
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import {
  AiBrain01Icon,
  Cancel01Icon,
  DragDropVerticalIcon,
  Image01Icon,
  Message01Icon,
  Mic01Icon,
  Video01Icon,
  VolumeHighIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRouterState } from "@tanstack/react-router";
import {
  LOADED_MODEL_KIND_LABELS,
  type LoadedModelEntry,
  type LoadedModelKind,
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
}: {
  entry: LoadedModelEntry;
  ejecting: boolean;
  onEject: () => void;
}) {
  const label = shortModelLabel(entry.name);
  return (
    <div className="flex items-center gap-2 rounded-[14px] px-1.5 py-1 transition-colors hover:bg-foreground/[0.04]">
      <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground/[0.05] text-muted-foreground">
        <HugeiconsIcon
          icon={KIND_ICONS[entry.kind]}
          strokeWidth={1.75}
          className="size-[15px]"
        />
      </span>
      <div className="min-w-0 flex-1">
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <div className="truncate text-ui-12p5 font-medium text-foreground">
              {label}
            </div>
          </TooltipTrigger>
          <TooltipContent side="left" sideOffset={6}>
            {entry.name}
          </TooltipContent>
        </Tooltip>
        <div className="truncate text-ui-11 text-muted-foreground">
          {rowSubtitle(entry)}
        </div>
      </div>
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
  const { position, panelRef, startDrag, dragging } = useDragPosition(
    LOADED_MODELS_PREFERENCE_KEYS.position,
  );

  if (!enabled || entries.length === 0) return null;

  const countLabel = `${entries.length} ${entries.length === 1 ? "model" : "models"} loaded`;

  return (
    <div
      ref={panelRef}
      className={cn(
        // Dragged: pinned where the user left it, out of the stack's flow.
        // Otherwise anchored bottom-right, or flowing as a right-aligned row
        // in the shared stack so overlays stack instead of overlapping.
        "pointer-events-none",
        position && "fixed z-[9999] w-fit",
        !position &&
          (positioned ? "fixed bottom-4 right-4 z-50" : "flex min-h-0 justify-end"),
        dragging && "select-none",
      )}
      style={position ? { left: position.left, top: position.top } : undefined}
    >
      {collapsed ? (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={`${countLabel}. Show details`}
              onClick={() => setCollapsed(false)}
              className="menu-soft-surface pointer-events-auto flex h-9 cursor-pointer items-center gap-1.5 rounded-full pl-2.5 pr-3 font-heading text-muted-foreground transition-colors hover:text-foreground"
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
        <div className="menu-soft-surface pointer-events-auto flex min-h-0 w-[268px] max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-[20px] p-1.5 font-heading">
          <div className="flex items-center gap-1.5 px-1.5 pb-1 pt-0.5">
            <HugeiconsIcon
              icon={AiBrain01Icon}
              strokeWidth={1.75}
              className="size-[15px] shrink-0 text-muted-foreground"
            />
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
          {/* Capped so four resident runtimes still leave the banners on screen. */}
          <div className="flex max-h-[min(272px,42dvh)] min-h-0 flex-col gap-0.5 overflow-y-auto">
            {entries.map((entry) => (
              <LoadedModelRow
                key={entry.id}
                entry={entry}
                ejecting={ejecting.has(entry.id)}
                onEject={() => void eject(entry)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
