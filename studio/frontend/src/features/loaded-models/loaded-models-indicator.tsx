// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Joins the shared bottom-right stack so update banners and the download panel never overlap it.

import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAudioWorkspaceStore } from "@/features/audio/stores/audio-workspace-store";
import { hasAuthToken, mustChangePassword } from "@/features/auth";
import { useSettingsDialogStore } from "@/features/settings";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { isTauri } from "@/lib/api-base";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { subscribeModelLifecycle } from "@/lib/model-lifecycle-events";
import { SparkleIcon } from "@/lib/sparkle-icon";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  DragDropVerticalIcon,
  Image01Icon,
  Message01Icon,
  Mic01Icon,
  RemoveCircleIcon,
  Video01Icon,
} from "@hugeicons/core-free-icons";
import { Volume02Icon } from "@/lib/volume-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useCallback, useEffect, useRef } from "react";
import {
  type LoadedModelEntry,
  type LoadedModelKind,
  loadedModelKindLabel,
  loadedModelTarget,
  shortModelLabel,
} from "./loaded-models-sources";
import {
  LOADED_MODELS_PREFERENCE_KEYS,
  setLoadedModelsDismissed,
  useLoadedModelsDismissed,
  useShowLoadedModels,
} from "./show-loaded-models-pref";
import { useDragPosition } from "./use-drag-position";
import { useLoadedModels } from "./use-loaded-models";

const COLLAPSED_KEY = LOADED_MODELS_PREFERENCE_KEYS.collapsed;

const KIND_ICONS: Record<LoadedModelKind, typeof SparkleIcon> = {
  text: Message01Icon,
  tts: Volume02Icon,
  image: Image01Icon,
  video: Video01Icon,
  stt: Mic01Icon,
};

// Desktop auto-authenticates; in the browser, polling before a token exists is all 401s.
const HIDDEN_ROUTES = new Set(["/login", "/signup", "/change-password"]);

function canShowIndicator(pathname: string): boolean {
  if (HIDDEN_ROUTES.has(pathname)) return false;
  if (isTauri) return true;
  return hasAuthToken() && !mustChangePassword();
}

function rowSubtitle(entry: LoadedModelEntry): string {
  const kind = loadedModelKindLabel(entry);
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
  const target = loadedModelTarget(entry.source, entry.workflows);
  return (
    <div className="flex items-center gap-2 rounded-[14px] px-1.5 py-1 transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)]">
      {/* Only the label half is the link: the eject button cannot nest inside it. */}
      <Tooltip>
        <TooltipTrigger asChild={true}>
          <button
            type="button"
            aria-label={`${label}. Open ${target.label}`}
            onClick={onOpen}
            className="flex min-w-0 flex-1 items-center gap-2 text-left"
          >
            <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] text-muted-foreground">
              <HugeiconsIcon
                icon={KIND_ICONS[entry.kind]}
                strokeWidth={1.75}
                className="size-[calc(15px*var(--ui-space-scale,1))]"
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
      {entry.loading ? (
        <span className="flex size-6 shrink-0 items-center justify-center">
          <Spinner className="size-3.5" label="Loading" />
        </span>
      ) : (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={`Eject ${label}`}
              disabled={ejecting}
              onClick={onEject}
              className="flex size-6 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground disabled:pointer-events-none disabled:opacity-60"
            >
              {ejecting ? (
                <Spinner className="size-3.5" label="Ejecting" />
              ) : (
                // Eject releases weights, unlike the header's X.
                <HugeiconsIcon
                  icon={RemoveCircleIcon}
                  strokeWidth={1.75}
                  className="size-3.5"
                />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="left" sideOffset={6}>
            Eject to free memory
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}

export function LoadedModelsIndicator({
  positioned = true,
}: { positioned?: boolean } = {}) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const showIndicator = useShowLoadedModels();
  const dismissed = useLoadedModelsDismissed();
  const reachable = canShowIndicator(pathname);
  const enabled = showIndicator && !dismissed && reachable;
  // Reachability carries the auth gate: polling on /login would trigger authFetch's refresh-redirect
  // ladder. Dismissal is excluded so a closed card still hears the next load.
  const { entries, polledEntries, ejecting, eject } = useLoadedModels(
    enabled,
    showIndicator && reachable,
  );
  const [collapsed, setCollapsed] = usePersistedToggle(COLLAPSED_KEY);
  const navigate = useNavigate();
  const openEntry = useCallback(
    (entry: LoadedModelEntry) => {
      const target = loadedModelTarget(
        entry.source,
        entry.workflows,
        useAudioWorkspaceStore.getState().workflow,
      );
      if (target.open === "settings") {
        // Read on click, not at render: the settings barrel imports back here (circular).
        useSettingsDialogStore.getState().openDialog(target.tab);
        return;
      }
      // Navigation only (Audio carries its workflow): no new thread, no reload.
      void navigate({ to: target.to, search: target.search });
    },
    [navigate],
  );
  const { position, panelRef, startDrag, dragging, justDragged } =
    useDragPosition(LOADED_MODELS_PREFERENCE_KEYS.position);

  // Subscribed above the early return so a dismissed card still hears the load that reopens it.
  useEffect(
    () =>
      subscribeModelLifecycle(({ loading }) => {
        if (loading) {
          setLoadedModelsDismissed(false);
        }
      }),
    [],
  );

  // API loads raise no frontend event, so a new polled row reopens a closed card. The first
  // poll after closing is the baseline, or the card could never be closed.
  const idsWhileClosedRef = useRef<Set<string> | null>(null);
  useEffect(() => {
    if (!dismissed) {
      idsWhileClosedRef.current = null;
      return;
    }
    const ids = new Set(polledEntries.map((entry) => entry.id));
    const before = idsWhileClosedRef.current;
    idsWhileClosedRef.current = ids;
    if (!before) return;
    for (const id of ids) {
      if (!before.has(id)) {
        setLoadedModelsDismissed(false);
        return;
      }
    }
  }, [dismissed, polledEntries]);

  if (!enabled || entries.length === 0) return null;

  const countLabel = `${entries.length} ${entries.length === 1 ? "model" : "models"} loaded`;

  return (
    <div
      ref={panelRef}
      className={cn(
        "pointer-events-none",
        position && "fixed z-[9999] w-fit",
        !position &&
          (positioned
            ? "fixed bottom-4 right-4 z-50"
            : "flex min-h-0 justify-end"),
        dragging && "select-none",
      )}
      style={position ? { left: position.left, top: position.top } : undefined}
    >
      {collapsed ? (
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={`${countLabel}. Show details, or drag to move`}
              onPointerDown={startDrag}
              // The pill is its own drag handle, so a press that moved must not also expand the card.
              onClick={() => {
                if (!justDragged()) setCollapsed(false);
              }}
              className="menu-soft-surface menu-soft-edgeless pointer-events-auto flex h-9 cursor-grab touch-none items-center gap-1.5 rounded-full pl-2.5 pr-3 font-heading text-muted-foreground transition-colors hover:text-foreground active:cursor-grabbing"
            >
              <HugeiconsIcon
                icon={SparkleIcon}
                strokeWidth={1.75}
                className="size-[calc(15px*var(--ui-space-scale,1))]"
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
        <div className="menu-soft-surface menu-soft-edgeless pointer-events-auto flex min-h-0 w-[calc(268px*var(--ui-space-scale,1))] max-w-[calc(100vw-2rem)] flex-col overflow-hidden rounded-[20px] p-1.5 font-heading">
          <div className="flex items-center gap-1.5 px-1.5 pb-2.5 pt-0.5">
            <HugeiconsIcon
              icon={SparkleIcon}
              strokeWidth={1.75}
              className="size-[calc(15px*var(--ui-space-scale,1))] shrink-0 text-muted-foreground"
            />
            <span className="min-w-0 flex-1 truncate text-ui-12p5 font-semibold text-foreground">
              Loaded models
            </span>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <div
                  aria-label="Drag to move"
                  // Not a button, so no click consumes the drag sentinel; without this the next click is refused.
                  onPointerDown={startDrag}
                  className="flex size-6 shrink-0 cursor-grab touch-none items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground active:cursor-grabbing"
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
                  className="flex size-6 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground"
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
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  aria-label="Close loaded models"
                  onClick={() => setLoadedModelsDismissed(true)}
                  className="flex size-6 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(7%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={Cancel01Icon}
                    strokeWidth={2}
                    className="size-3.5"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent side="left" sideOffset={6}>
                <span className="block">Close</span>
                <span className="block text-muted-foreground">
                  Back on the next model load
                </span>
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="flex max-h-[min(272px,42dvh)] min-h-0 flex-col gap-0.5 overflow-y-auto">
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
