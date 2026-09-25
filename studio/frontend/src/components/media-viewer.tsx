// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ArrowTurnBackwardIcon,
  Cancel01Icon,
  Delete02Icon,
  Download01Icon,
  Folder01Icon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type KeyboardEventHandler, type ReactNode, useRef, useState } from "react";

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { StarPointedIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import { type MediaZoom, MediaZoomStage } from "./media-zoom";
import { type MediaNoun, useProjectSubmenu } from "./project-submenu";

export interface MediaViewerActions {
  /** The white pill, e.g. Chat about this. */
  primary?: { label: string; icon: IconSvgElement; onClick: () => void; disabled?: boolean };
  onDownload?: () => void;
  /** Where the file came from: its chat, or the page that generated it. */
  viewOriginal?: { label: string; onClick: () => void };
  /** Shown only where it can work; see the Library's useRevealLabel. */
  reveal?: { label: string; onClick: () => void };
  favorite?: boolean;
  onToggleFavorite?: () => void;
  onAddToProject?: (projectId: string) => Promise<{ already: boolean }>;
  onDelete?: () => void;
}

const MEDIA_ZOOMS = [0.25, 0.5, 0.75, 1, 1.25, 1.5] as const;

const MORE_ACTIONS: Record<MediaNoun, TranslationKey> = {
  image: "library.viewer.moreActionsImage",
  video: "library.viewer.moreActionsVideo",
  clip: "library.viewer.moreActionsClip",
  file: "library.viewer.moreActionsFile",
};

function percent(scale: number, locale: string): string {
  return new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 0 }).format(
    scale,
  );
}

/** The header's scale pill: the current scale, and a menu of `scales`, after Fit when the frame's
 *  `fitScale` is given. */
export function ScaleMenu({
  value,
  scales,
  fitScale,
  onChange,
}: {
  value: MediaZoom;
  scales: readonly number[];
  fitScale?: number;
  onChange: (value: MediaZoom) => void;
}) {
  const t = useT();
  const locale = useLocale();
  const options = scales.map((scale) => ({ value: String(scale), label: percent(scale, locale) }));
  if (fitScale !== undefined) options.unshift({ value: "fit", label: t("library.viewer.fit") });
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          aria-label={t("library.viewer.scale")}
          className="mr-1 flex h-9 shrink-0 items-center gap-1 rounded-full bg-muted px-3.5 text-sm tabular-nums outline-none transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring"
        >
          {percent(value === "fit" ? (fitScale ?? 1) : value, locale)}
          <HugeiconsIcon icon={ChevronDownStandardIcon} strokeWidth={1.75} className="size-4" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-40">
        <DropdownMenuRadioGroup
          value={String(value)}
          onValueChange={(next) => onChange(next === "fit" ? "fit" : Number(next))}
        >
          {options.map((option) => (
            <DropdownMenuRadioItem key={option.value} value={option.value}>
              {option.label}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/**
 * One file, nearly as tall as the window. Images and videos can be scaled and, once larger than the
 * frame, dragged. Page-agnostic, so any page that shows files can open them the same way.
 */
export function MediaViewer({
  open,
  onOpenChange,
  title,
  meta,
  media,
  noun,
  actions,
  extra,
  onKeyDown,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  meta?: ReactNode;
  /** Images and videos: adds the scale menu. */
  media: boolean;
  /** Used in labels and messages, e.g. "image". */
  noun: MediaNoun;
  actions: MediaViewerActions;
  /** Anything else for the header, before the actions (e.g. Save). */
  extra?: ReactNode;
  onKeyDown?: KeyboardEventHandler<HTMLDivElement>;
  children: ReactNode;
}) {
  const t = useT();
  const [menuOpen, setMenuOpen] = useState(false);
  const [zoom, setZoom] = useState<MediaZoom>("fit");
  const [fitScale, setFitScale] = useState<number | null>(null);
  // Every opening starts at Fit.
  const [wasOpen, setWasOpen] = useState(open);
  if (open !== wasOpen) {
    setWasOpen(open);
    if (open) setZoom("fit");
  }
  // Opened from a card, a row or a link rather than a DialogTrigger, so Radix has nothing to return
  // focus to on close and would drop it on <body>. Whatever had focus when it opened gets it back.
  const returnFocus = useRef<HTMLElement | null>(null);
  const project = useProjectSubmenu({ noun, onAddToProject: actions.onAddToProject });
  const iconButton =
    "flex size-9 shrink-0 items-center justify-center rounded-full outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring aria-expanded:bg-muted";
  const hasMenu = Boolean(
    actions.viewOriginal ||
      actions.reveal ||
      actions.onToggleFavorite ||
      actions.onAddToProject ||
      actions.onDelete,
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton={false}
        onKeyDown={onKeyDown}
        onOpenAutoFocus={() => {
          const active = document.activeElement;
          returnFocus.current = active instanceof HTMLElement && active !== document.body ? active : null;
        }}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          const target = returnFocus.current;
          returnFocus.current = null;
          if (target?.isConnected) target.focus({ preventScroll: true });
        }}
        className="flex h-[calc(100dvh-var(--studio-window-chrome-top,0px)-2rem)] w-[min(92vw,1200px)] max-w-none flex-col gap-0 overflow-hidden p-0 sm:max-w-none"
      >
        <div className="flex items-center gap-2 py-3 pl-6 pr-4">
          <div className="min-w-0 flex-1">
            <DialogTitle className="truncate text-[15px] font-medium">{title}</DialogTitle>
            <DialogDescription className="mt-0.5 truncate text-[13px]">
              {meta}
            </DialogDescription>
          </div>
          {extra}
          {media && fitScale !== null && (
            <ScaleMenu value={zoom} scales={MEDIA_ZOOMS} fitScale={fitScale} onChange={setZoom} />
          )}
          {actions.primary && (
            <button
              type="button"
              disabled={actions.primary.disabled}
              onClick={actions.primary.onClick}
              className="mr-1 flex h-9 shrink-0 items-center gap-2 rounded-full bg-foreground px-4 text-sm font-medium text-background outline-none transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
            >
              <HugeiconsIcon icon={actions.primary.icon} strokeWidth={1.75} className="size-4" />
              {actions.primary.label}
            </button>
          )}
          {actions.onDownload && (
            <button
              type="button"
              aria-label={t("library.viewer.download")}
              title={t("library.viewer.download")}
              onClick={actions.onDownload}
              className={iconButton}
            >
              <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-5" />
            </button>
          )}
          {hasMenu && (
            <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
              <DropdownMenuTrigger asChild={true}>
                <button type="button" aria-label={t(MORE_ACTIONS[noun])} className={iconButton}>
                  <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                align="end"
                className="unsloth-plus-menu sidebar-row-menu menu-flat-destructive w-56"
              >
                {actions.viewOriginal && (
                  <DropdownMenuItem onClick={actions.viewOriginal.onClick}>
                    <HugeiconsIcon icon={ArrowTurnBackwardIcon} strokeWidth={1.75} className="size-icon" />
                    {actions.viewOriginal.label}
                  </DropdownMenuItem>
                )}
                {actions.reveal && (
                  <DropdownMenuItem onClick={actions.reveal.onClick}>
                    <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                    {actions.reveal.label}
                  </DropdownMenuItem>
                )}
                {(actions.viewOriginal || actions.reveal) && <DropdownMenuSeparator />}
                {actions.onToggleFavorite && (
                  <DropdownMenuItem onClick={actions.onToggleFavorite}>
                    <HugeiconsIcon
                      icon={StarPointedIcon}
                      strokeWidth={1.75}
                      className={cn("size-icon", actions.favorite && "[&_path]:fill-current")}
                    />
                    {t(
                      actions.favorite
                        ? "library.menu.removeFromFavorites"
                        : "library.menu.addToFavorites",
                    )}
                  </DropdownMenuItem>
                )}
                {project.submenu}
                {actions.onDelete && (
                  <DropdownMenuItem variant="destructive" onClick={actions.onDelete}>
                    <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-icon" />
                    {t("common.delete")}
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          <DialogClose asChild={true}>
            <button type="button" aria-label={t("library.viewer.close")} className={iconButton}>
              <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-5" />
            </button>
          </DialogClose>
        </div>
        <div className={cn("flex min-h-0 flex-1", media ? "px-4 pb-4" : "px-6 pb-6")}>
          {media ? (
            <MediaZoomStage zoom={zoom} onFitScale={setFitScale}>
              {children}
            </MediaZoomStage>
          ) : (
            children
          )}
        </div>
        {project.dialog}
      </DialogContent>
    </Dialog>
  );
}
