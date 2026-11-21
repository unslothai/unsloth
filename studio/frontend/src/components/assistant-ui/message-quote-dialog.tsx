// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Compact, non-modal quote comment panel. It floats above the selected passage
// without dimming the page. Saving only appends to the session quote
// collection; composer text, focus, and submission are untouched.

import { useT } from "@/i18n";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type FC,
  type FormEvent,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import type { QuoteAnchor } from "./message-quote-model";
import {
  type SelectionRect,
  type ToolbarSize,
  type ViewportRect,
  placeToolbar,
  shortMessageId,
} from "./message-selection-model";

type MessageQuoteDialogProps = {
  readonly anchor: QuoteAnchor | null;
  readonly anchorRect: SelectionRect | null;
  readonly viewportEl: HTMLElement | null;
  readonly open: boolean;
  readonly onCancel: (reason: QuoteDialogCancelReason) => void;
  readonly onSave: (comment: string) => void;
};

export type QuoteDialogCancelReason = "escape" | "outside";

const PANEL_MARGIN = 8;
const PANEL_MAX_WIDTH = 420;
// Only used while the real panel is measured; keeps any hidden frame close to
// the quote instead of at the viewport origin.
const PANEL_ESTIMATED_HEIGHT = 104;

function fallbackPlacement(
  anchor: SelectionRect,
  size: ToolbarSize,
  viewport: ViewportRect,
): { left: number; top: number } {
  const minLeft = viewport.left + PANEL_MARGIN;
  const maxLeft = Math.max(
    minLeft,
    viewport.left + viewport.width - PANEL_MARGIN - size.width,
  );
  const minTop = viewport.top + PANEL_MARGIN;
  const maxTop = Math.max(
    minTop,
    viewport.top + viewport.height - PANEL_MARGIN - size.height,
  );
  return {
    left: Math.min(
      Math.max(minLeft, anchor.left + anchor.width / 2 - size.width / 2),
      maxLeft,
    ),
    top: Math.min(
      Math.max(minTop, anchor.top - PANEL_MARGIN - size.height),
      maxTop,
    ),
  };
}

function viewportFor(viewportEl: HTMLElement): ViewportRect {
  const rect = viewportEl.getBoundingClientRect();
  // Keep the panel above the composer dock even though the viewport rect
  // includes the area painted behind it.
  const composerTop =
    viewportEl.parentElement
      ?.querySelector<HTMLElement>(".aui-thread-composer-dock")
      ?.getBoundingClientRect().top ?? null;
  return {
    left: rect.left,
    top: rect.top,
    width: rect.width,
    height:
      composerTop === null
        ? rect.height
        : Math.max(0, Math.min(rect.height, composerTop - rect.top)),
  };
}

export const MessageQuoteDialog: FC<MessageQuoteDialogProps> = ({
  anchor,
  anchorRect,
  viewportEl,
  open,
  onCancel,
  onSave,
}) => {
  if (
    !open ||
    anchor === null ||
    anchorRect === null ||
    viewportEl === null ||
    typeof document === "undefined"
  ) {
    return null;
  }

  return createPortal(
    <QuoteCommentPanel
      anchor={anchor}
      anchorRect={anchorRect}
      viewportEl={viewportEl}
      onCancel={onCancel}
      onSave={onSave}
    />,
    document.body,
  );
};

const QuoteCommentPanel: FC<{
  readonly anchor: QuoteAnchor;
  readonly anchorRect: SelectionRect;
  readonly viewportEl: HTMLElement;
  readonly onCancel: (reason: QuoteDialogCancelReason) => void;
  readonly onSave: (comment: string) => void;
}> = ({ anchor, anchorRect, viewportEl, onCancel, onSave }) => {
  const t = useT();
  const descriptionId = useId();
  const [comment, setComment] = useState("");
  const [size, setSize] = useState<ToolbarSize | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const shortId = shortMessageId(anchor.messageId);
  const source =
    anchor.turn > 0
      ? `${t("turns.label", { number: anchor.turn })} · ${shortId}`
      : shortId;

  const measure = useCallback(() => {
    const node = panelRef.current;
    if (node === null) {
      return;
    }
    const rect = node.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return;
    }
    setSize((current) =>
      current !== null &&
      current.width === rect.width &&
      current.height === rect.height
        ? current
        : { width: rect.width, height: rect.height },
    );
  }, []);

  useLayoutEffect(() => {
    measure();
    const node = panelRef.current;
    if (node === null) {
      return;
    }
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    window.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [measure]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      inputRef.current?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    const onPointerDown = (event: PointerEvent) => {
      const node = panelRef.current;
      if (
        node !== null &&
        event.target instanceof Node &&
        node.contains(event.target)
      ) {
        return;
      }
      onCancel("outside");
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") {
        return;
      }
      event.preventDefault();
      onCancel("escape");
    };
    // Defer listener attachment so the click that opened the panel cannot
    // immediately dismiss it.
    const frame = window.requestAnimationFrame(() => {
      document.addEventListener("pointerdown", onPointerDown, true);
      document.addEventListener("keydown", onKeyDown, true);
    });
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeyDown, true);
    };
  }, [onCancel]);

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSave(comment);
  };

  const viewport = viewportFor(viewportEl);
  const panelWidth = Math.max(
    0,
    Math.min(PANEL_MAX_WIDTH, viewport.width - PANEL_MARGIN * 2),
  );
  const placement =
    size === null
      ? null
      : (placeToolbar(anchorRect, size, viewport, PANEL_MARGIN) ??
        fallbackPlacement(anchorRect, size, viewport));
  const hiddenPlacement = fallbackPlacement(
    anchorRect,
    {
      width: Math.max(panelWidth, 1),
      height: PANEL_ESTIMATED_HEIGHT,
    },
    viewport,
  );
  // Stay hidden for the first measurement, then appear instantly at the
  // measured spot. No entrance animation: the click should pop the panel in.
  const style =
    panelWidth <= 0
      ? {
          left: hiddenPlacement.left,
          top: hiddenPlacement.top,
          width: panelWidth,
          visibility: "hidden" as const,
        }
      : placement === null
        ? {
            left: hiddenPlacement.left,
            top: hiddenPlacement.top,
            width: panelWidth,
            visibility: "hidden" as const,
          }
        : {
            left: placement.left,
            top: placement.top,
            width: panelWidth,
          };

  return (
    <div
      ref={panelRef}
      data-quote-comment-panel=""
      // biome-ignore lint/a11y/useSemanticElements: a non-modal anchored comment panel; a native <dialog> would add modal semantics.
      role="dialog"
      aria-modal={false}
      aria-label={t("messageSelection.quoteDialogTitle")}
      aria-describedby={descriptionId}
      style={style}
      className="menu-soft-surface fixed z-50 flex flex-col gap-2 rounded-xl p-3 font-sans"
    >
      <p id={descriptionId} className="sr-only">
        {t("messageSelection.quoteDialogDescription")}
      </p>
      <span className="sr-only">
        {t("messageSelection.quoteDialogPassage")}: {source}
      </span>

      <div className="flex min-w-0 items-stretch gap-2.5">
        <span
          aria-hidden={true}
          className="w-0.5 shrink-0 self-stretch rounded-full bg-muted-foreground/35"
        />
        <p
          data-quote-dialog-passage=""
          className="line-clamp-2 min-w-0 flex-1 text-ui-12 leading-snug text-foreground/85"
        >
          {anchor.text}
        </p>
      </div>

      <form className="flex items-center gap-1.5" onSubmit={submit}>
        <input
          ref={inputRef}
          type="text"
          value={comment}
          onChange={(event) => setComment(event.target.value)}
          placeholder={t("messageSelection.quoteDialogCommentPlaceholder")}
          aria-label={t("messageSelection.quoteDialogCommentLabel")}
          className="h-9 min-w-0 flex-1 bg-transparent px-1.5 text-ui-13 text-foreground outline-none placeholder:text-muted-foreground"
        />
        <button
          type="submit"
          aria-label={t("messageSelection.quoteDialogSave")}
          title={t("messageSelection.quoteDialogSave")}
          className="flex size-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <HugeiconsIcon
            icon={Tick02Icon}
            strokeWidth={2}
            aria-hidden={true}
            className="size-[calc(16px*var(--ui-space-scale,1))]"
          />
        </button>
      </form>
    </div>
  );
};
