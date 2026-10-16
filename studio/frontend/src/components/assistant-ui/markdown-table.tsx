// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
import { Cancel01Icon, ExpandIcon } from "@hugeicons/core-free-icons";
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
  TableCopyDropdown,
  TableDownloadDropdown,
  defaultTranslations,
} from "streamdown";

// Streamdown's copy and download find the table through this attribute, so the wrapper keeps it.
const TABLE_WRAPPER = "table-wrapper";

/**
 * The table's horizontal scroller. Flags which edges hide more columns so CSS fades them, and
 * publishes the scrollbar's height so the fade stops above it.
 */
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

/**
 * Markdown tables drawn inline, without Streamdown's card and toolbar row: copy, download and
 * expand float over the header's right end on hover, and expand opens the table in a dialog.
 */
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
  return (
    <div className="aui-md-table" data-streamdown={TABLE_WRAPPER}>
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
          <HugeiconsIcon icon={ExpandIcon} size={14} />
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
              className="absolute top-4 right-4 flex size-10 cursor-pointer items-center justify-center rounded-full bg-popover text-foreground shadow-lg ring-1 ring-border/60 transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
