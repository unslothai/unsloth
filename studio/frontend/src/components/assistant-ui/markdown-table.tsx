// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import { Cancel01Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ComponentProps,
  type ReactNode,
  useContext,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  type ExtraProps,
  StreamdownContext,
  type StreamdownProps,
  TableCopyDropdown,
  TableDownloadDropdown,
  defaultTranslations,
} from "streamdown";
import { ExpandArrowsIcon } from "@/lib/expand-icons";

// Streamdown ships its own glyphs for the table controls; swap in ours so copy,
// download and expand match the icons used everywhere else.
// `strokeWidth` is dropped, not forwarded: SVG types it `string | number`, HugeiconsIcon wants a number.
function streamdownIcon(icon: typeof Copy01Icon) {
  return function StreamdownIcon({
    size,
    strokeWidth: _strokeWidth,
    ...props
  }: ComponentProps<"svg"> & { size?: number }) {
    return <HugeiconsIcon icon={icon} size={size} {...props} />;
  };
}
export const MARKDOWN_TABLE_ICONS = {
  CopyIcon: streamdownIcon(Copy01Icon),
  DownloadIcon: streamdownIcon(Download01Icon),
  // Streamdown's key for the table's enlarge control.
  Maximize2Icon: streamdownIcon(ExpandArrowsIcon),
} satisfies NonNullable<StreamdownProps["icons"]>;

// Streamdown's copy and download find the table through this attribute, so the wrapper keeps it.
const TABLE_WRAPPER = "table-wrapper";

// Flags the edges hiding columns (CSS fades them) and the scrollbar height the fade stops above.
function TableScroll({
  children,
  className,
  props,
}: {
  children: ReactNode;
  className?: string;
  props: ComponentProps<"table">;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = scroller.current;
    if (!el) {
      return;
    }
    const update = () => {
      const max = el.scrollWidth - el.clientWidth;
      el.toggleAttribute("data-fade-start", el.scrollLeft > 1);
      el.toggleAttribute("data-fade-end", el.scrollLeft < max - 1);
      el.style.setProperty(
        "--md-table-scrollbar",
        `${el.offsetHeight - el.clientHeight}px`,
      );
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    // Streaming rows widen the table without resizing the scroller.
    if (el.firstElementChild) {
      observer.observe(el.firstElementChild);
    }
    el.addEventListener("scroll", update, { passive: true });
    return () => {
      observer.disconnect();
      el.removeEventListener("scroll", update);
    };
  }, []);
  return (
    <div ref={scroller} className="aui-md-table-scroll">
      <table className={className} data-streamdown="table" {...props}>
        {children}
      </table>
    </div>
  );
}

// rgb()/rgba() and the modern `/ alpha` syntax; anything else is taken as opaque.
const SLASH_ALPHA = /\/\s*([\d.]+%?)\s*\)$/;
const RGBA_ALPHA = /^rgba\([^,]+,[^,]+,[^,]+,\s*([\d.]+)\s*\)$/;
function isOpaque(color: string): boolean {
  const alpha = SLASH_ALPHA.exec(color)?.[1] ?? RGBA_ALPHA.exec(color)?.[1];
  if (alpha === undefined) {
    return true;
  }
  return alpha.endsWith("%")
    ? Number.parseFloat(alpha) >= 100
    : Number(alpha) >= 1;
}

// The controls sit over the header, so they paint whatever is behind the table: every background
// up to the first opaque one, stacked. Chat, the Hub, the browser panel and dialogs all differ.
function paintSurface(wrapper: HTMLElement): void {
  const layers: string[] = [];
  for (let el = wrapper.parentElement; el; el = el.parentElement) {
    const color = getComputedStyle(el).backgroundColor;
    if (color === "transparent" || color === "rgba(0, 0, 0, 0)") {
      continue;
    }
    layers.push(`linear-gradient(${color}, ${color})`);
    if (isOpaque(color)) {
      break;
    }
  }
  layers.push("linear-gradient(var(--background), var(--background))");
  wrapper.style.setProperty("--md-table-surface", layers.join(", "));
}

export function MarkdownTable({
  children,
  className,
  node: _node,
  ...props
}: ComponentProps<"table"> & ExtraProps) {
  const t = useT();
  const { isAnimating } = useContext(StreamdownContext);
  const [expanded, setExpanded] = useState(false);
  const label = defaultTranslations.viewFullscreen;
  const wrapper = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (wrapper.current) {
      paintSurface(wrapper.current);
    }
  }, []);
  // Repainted on each approach, so a theme switch since mount is picked up.
  const repaint = (event: { currentTarget: HTMLDivElement }) =>
    paintSurface(event.currentTarget);
  return (
    <div
      ref={wrapper}
      className="aui-md-table"
      data-streamdown={TABLE_WRAPPER}
      onPointerEnter={repaint}
      onFocus={repaint}
    >
      <TableScroll className={className} props={props}>
        {children}
      </TableScroll>
      <div className="aui-md-table-actions">
        <TableCopyDropdown />
        <TableDownloadDropdown />
        <button
          type="button"
          title={label}
          aria-label={label}
          disabled={isAnimating}
          onClick={() => setExpanded(true)}
        >
          <HugeiconsIcon icon={ExpandArrowsIcon} size={14} />
        </button>
      </div>
      <Dialog open={expanded} onOpenChange={setExpanded}>
        <DialogContent
          showCloseButton={false}
          aria-describedby={undefined}
          overlayClassName="bg-black/70"
          overlayTone="dense"
          // The whole window below the desktop titlebar, so a click beside the panel closes it.
          className="top-[var(--studio-window-chrome-top,0px)] right-0 bottom-0 left-0 flex h-auto max-h-none w-auto max-w-none translate-none items-center justify-center gap-0 overflow-hidden rounded-none bg-transparent p-4 ring-0 sm:max-w-none sm:p-16 max-sm:h-auto"
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              setExpanded(false);
            }
          }}
        >
          <DialogTitle className="sr-only">{label}</DialogTitle>
          <div className="aui-md-table aui-md-table-dialog max-h-full max-w-[min(72rem,100%)] overflow-y-auto rounded-3xl bg-popover px-6 py-8 ring-1 ring-border/60 sm:px-14 sm:py-12">
            <TableScroll className={className} props={props}>
              {children}
            </TableScroll>
          </div>
          <DialogClose asChild={true}>
            <button
              type="button"
              aria-label={t("common.close")}
              className="absolute top-4 right-4 flex size-10 cursor-pointer items-center justify-center rounded-full bg-popover text-foreground shadow-lg transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <HugeiconsIcon
                icon={Cancel01Icon}
                strokeWidth={1.75}
                className="size-5"
              />
            </button>
          </DialogClose>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// For markdown shown inside another control: the same table, without controls.
export function StaticMarkdownTable({
  children,
  className,
  node: _node,
  ...props
}: ComponentProps<"table"> & ExtraProps) {
  return (
    <div className="aui-md-table">
      <TableScroll className={className} props={props}>
        {children}
      </TableScroll>
    </div>
  );
}
