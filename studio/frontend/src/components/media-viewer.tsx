// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ArrowLeft02Icon,
  ArrowRight02Icon,
  ArrowTurnBackwardIcon,
  Cancel01Icon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  Folder01Icon,
  Image02Icon,
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
import { type MediaInset, type MediaZoom, MediaZoomStage } from "./media-zoom";
import { type MediaNoun, useProjectSubmenu } from "./project-submenu";

export interface MediaViewerActions {
  primary?: { label: string; icon: IconSvgElement; onClick: () => void; disabled?: boolean };
  onDownload?: () => void;
  viewOriginal?: { label: string; onClick: () => void };
  reveal?: { label: string; onClick: () => void };
  favorite?: boolean;
  onToggleFavorite?: () => void;
  onAddToProject?: (projectId: string) => Promise<{ already: boolean }>;
  onDelete?: () => void;
  /** Delete's label when it only takes the file out of something, e.g. an unsent message. */
  deleteLabel?: string;
  copy?: { label: string; onClick: () => void };
}

export interface MediaViewerGallery {
  onPrevious?: () => void;
  onNext?: () => void;
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

const FLOATING =
  "bg-card shadow-[0_2px_10px_-2px_rgba(0,0,0,0.14)] hover:bg-[color-mix(in_oklab,var(--card),var(--foreground)_5%)] aria-expanded:bg-[color-mix(in_oklab,var(--card),var(--foreground)_5%)] dark:bg-[color-mix(in_oklab,var(--card),var(--foreground)_6%)] dark:shadow-none dark:hover:bg-[color-mix(in_oklab,var(--card),var(--foreground)_12%)] dark:aria-expanded:bg-[color-mix(in_oklab,var(--card),var(--foreground)_12%)]";

export function ScaleMenu({
  value,
  scales,
  fitScale,
  onChange,
  className,
  floating = false,
}: {
  value: MediaZoom;
  scales: readonly number[];
  fitScale?: number;
  onChange: (value: MediaZoom) => void;
  className?: string;
  floating?: boolean;
}) {
  const t = useT();
  const locale = useLocale();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          aria-label={t("library.viewer.scale")}
          className={cn(
            "mr-1 flex h-9 shrink-0 items-center gap-1 rounded-full bg-muted px-3.5 text-sm tabular-nums outline-none transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring",
            floating && cn("mr-0 h-10 pr-2.5 pl-4", FLOATING),
            className,
          )}
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
          {scales.map((scale) => (
            <DropdownMenuRadioItem key={scale} value={String(scale)}>
              {percent(scale, locale)}
            </DropdownMenuRadioItem>
          ))}
          {/* Fit, the one a preview opens at, sits apart below the fixed sizes, as ChatGPT has it. */}
          {fitScale !== undefined && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuRadioItem value="fit">{t("library.viewer.fit")}</DropdownMenuRadioItem>
            </>
          )}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function lightboxInset(): MediaInset {
  if (typeof window === "undefined") return { top: 0, right: 0, bottom: 0, left: 0 };
  const root = getComputedStyle(document.documentElement);
  const rem = (Number.parseFloat(root.fontSize) || 16) * (Number.parseFloat(root.getPropertyValue("--ui-space-scale")) || 1);
  const side = window.innerWidth < 640 ? 12 : 7 * rem;
  return { top: 6.5 * rem, right: side, bottom: 4.5 * rem, left: side };
}

function GalleryArrow({
  label,
  icon,
  onClick,
  className,
}: {
  label: string;
  icon: IconSvgElement;
  onClick?: () => void;
  className: string;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={!onClick}
      onClick={onClick}
      className={cn(
        "absolute top-1/2 z-10 flex size-11 -translate-y-1/2 cursor-pointer items-center justify-center rounded-full outline-none transition-[background-color,opacity] focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default disabled:opacity-35",
        FLOATING,
        className,
      )}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-5" />
    </button>
  );
}

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
  flush = false,
  redactFromReload = false,
  variant = "card",
  gallery,
  itemKey,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  meta?: ReactNode;
  media: boolean;
  noun: MediaNoun;
  actions: MediaViewerActions;
  extra?: ReactNode;
  onKeyDown?: KeyboardEventHandler<HTMLDivElement>;
  flush?: boolean;
  redactFromReload?: boolean;
  /** "lightbox": the picture alone over the blurred page, its controls floating above it. */
  variant?: "card" | "lightbox";
  gallery?: MediaViewerGallery;
  itemKey?: string;
  children: ReactNode;
}) {
  const t = useT();
  const [menuOpen, setMenuOpen] = useState(false);
  const [zoom, setZoom] = useState<MediaZoom>("fit");
  const [fitScale, setFitScale] = useState<number | null>(null);
  const [wasOpen, setWasOpen] = useState(open);
  if (open !== wasOpen) {
    setWasOpen(open);
    if (open) setZoom("fit");
  }
  const [shownKey, setShownKey] = useState(itemKey);
  if (itemKey !== shownKey) {
    setShownKey(itemKey);
    setZoom("fit");
  }
  const returnFocus = useRef<HTMLElement | null>(null);
  const project = useProjectSubmenu({ noun, onAddToProject: actions.onAddToProject });
  const lightbox = variant === "lightbox";
  const iconButton = lightbox
    ? cn(
        "flex size-10 shrink-0 cursor-pointer items-center justify-center rounded-full outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring",
        FLOATING,
      )
    : "flex size-9 shrink-0 items-center justify-center rounded-full outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring aria-expanded:bg-muted";
  const hasMenu = Boolean(
    actions.copy ||
      actions.viewOriginal ||
      actions.reveal ||
      actions.onToggleFavorite ||
      actions.onAddToProject ||
      actions.onDelete,
  );

  const zoomMenu = media && fitScale !== null && (
    <ScaleMenu
      value={zoom}
      scales={MEDIA_ZOOMS}
      fitScale={fitScale}
      onChange={setZoom}
      floating={lightbox}
    />
  );
  const controls = (
    <>
      {actions.primary && (
        <button
          type="button"
          disabled={actions.primary.disabled}
          onClick={actions.primary.onClick}
          className={
            lightbox
              ? cn(
                  "flex h-10 shrink-0 cursor-pointer items-center gap-2 rounded-full px-4 text-sm font-medium text-foreground outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default disabled:opacity-50",
                  FLOATING,
                )
              : "mr-1 flex h-9 shrink-0 items-center gap-2 rounded-full bg-foreground px-4 text-sm font-medium text-background outline-none transition-opacity hover:opacity-90 focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
          }
        >
          <HugeiconsIcon icon={actions.primary.icon} strokeWidth={1.75} className="size-4" />
          {actions.primary.label}
        </button>
      )}
      {actions.onDownload && (
        <button
          type="button"
          aria-label={t("library.menu.download")}
          title={t("library.menu.download")}
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
            sideOffset={lightbox ? 6 : undefined}
            className="unsloth-plus-menu sidebar-row-menu menu-flat-destructive w-56"
          >
            {actions.copy && (
              <>
                <DropdownMenuItem onClick={actions.copy.onClick}>
                  <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} className="size-icon" />
                  {actions.copy.label}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
              </>
            )}
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
                {actions.deleteLabel ?? t("common.delete")}
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      )}
      <DialogClose asChild={true}>
        <button type="button" aria-label={t("common.close")} className={iconButton}>
          <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-5" />
        </button>
      </DialogClose>
    </>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton={false}
        data-reload-snapshot-sensitive={redactFromReload ? "" : undefined}
        onKeyDown={(event) => {
          onKeyDown?.(event);
          if (event.defaultPrevented || !gallery) return;
          // Not while typing, or inside a menu, which uses the arrows itself.
          const target = event.target as HTMLElement;
          if (target.closest("input, textarea, [role=menu]")) return;
          const step =
            event.key === "ArrowLeft"
              ? gallery.onPrevious
              : event.key === "ArrowRight"
                ? gallery.onNext
                : undefined;
          if (!step) return;
          event.preventDefault();
          step();
        }}
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
        overlayClassName={
          lightbox
            ? "bg-[color-mix(in_oklab,var(--background)_55%,transparent)] supports-backdrop-filter:backdrop-blur-[1px] duration-150"
            : undefined
        }
        className={
          lightbox
            ? // The whole window below the desktop titlebar; the picture is the only surface.
              "top-[var(--studio-window-chrome-top,0px)] right-0 bottom-0 left-0 flex h-auto max-h-[calc(100dvh-var(--studio-window-chrome-top,0px))] w-auto max-w-none translate-none flex-col gap-0 overflow-hidden rounded-none bg-transparent p-0 ring-0 duration-150 sm:max-w-none max-sm:h-auto"
            : "flex h-[calc(100dvh-var(--studio-window-chrome-top,0px)-2rem)] w-[min(92vw,1200px)] max-w-none flex-col gap-0 overflow-hidden rounded-[1.5rem] p-0 sm:max-w-none"
        }
      >
        {lightbox ? (
          <>
            {/* Clicking around the picture closes, as it does in ChatGPT; a drag to pan never does. */}
            <div
              className="absolute inset-0 flex"
              onClick={(event) => {
                const target = event.target;
                if (!(target instanceof HTMLImageElement || target instanceof HTMLVideoElement)) {
                  onOpenChange(false);
                }
              }}
            >
              {media ? (
                <MediaZoomStage zoom={zoom} onFitScale={setFitScale} inset={lightboxInset()}>
                  {children}
                </MediaZoomStage>
              ) : (
                children
              )}
            </div>
            {gallery && (gallery.onPrevious || gallery.onNext) ? (
              <>
                {gallery.onPrevious ? (
                  <GalleryArrow
                    label={t("imageViewer.previous")}
                    icon={ArrowLeft02Icon}
                    onClick={gallery.onPrevious}
                    className="left-3"
                  />
                ) : null}
                {gallery.onNext ? (
                  <GalleryArrow
                    label={t("imageViewer.next")}
                    icon={ArrowRight02Icon}
                    onClick={gallery.onNext}
                    className="right-3"
                  />
                ) : null}
              </>
            ) : null}
            <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start gap-2 p-3">
              <div
                className={cn(
                  "pointer-events-auto flex h-10 min-w-0 max-w-[min(28rem,40vw)] items-center gap-2 rounded-full pl-3.5 pr-4",
                  FLOATING,
                  "hover:bg-card dark:hover:bg-[color-mix(in_oklab,var(--card),var(--foreground)_6%)]",
                )}
                title={typeof meta === "string" && meta ? `${title}\n${meta}` : title}
              >
                <HugeiconsIcon icon={Image02Icon} strokeWidth={1.75} className="size-4.5 shrink-0" />
                <DialogTitle className="truncate text-sm font-medium">{title}</DialogTitle>
                <DialogDescription className="sr-only">{meta}</DialogDescription>
              </div>
              <div className="pointer-events-auto ml-auto flex shrink-0 items-center gap-2">
                {extra}
                {zoomMenu}
                {controls}
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="flex items-center gap-2 py-3 pl-6 pr-4">
              <div className="min-w-0 flex-1">
                <DialogTitle className="truncate text-ui-15 font-medium">{title}</DialogTitle>
                <DialogDescription className="mt-0.5 truncate text-ui-13">
                  {meta}
                </DialogDescription>
              </div>
              {extra}
              {zoomMenu}
              {controls}
            </div>
            <div className={cn("flex min-h-0 flex-1", !flush && (media ? "px-4 pb-4" : "px-6 pb-6"))}>
              {media ? (
                <MediaZoomStage zoom={zoom} onFitScale={setFitScale}>
                  {children}
                </MediaZoomStage>
              ) : (
                children
              )}
            </div>
          </>
        )}
        {project.dialog}
      </DialogContent>
    </Dialog>
  );
}
