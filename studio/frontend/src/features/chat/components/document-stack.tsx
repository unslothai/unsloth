// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { cn } from "@/lib/utils";
import { ChevronDownIcon, ChevronUpIcon, FileText } from "lucide-react";
import { motion, useAnimation, useReducedMotion } from "motion/react";
import type { KeyboardEvent } from "react";
import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { PendingDocumentAttachment } from "../types";
import {
  AttachmentChipBody,
  AttachmentChipTitle,
} from "./attachment-chip-primitives";
import {
  DocAttachmentChip,
  documentAttachmentSummary,
} from "./doc-attachment-chip";
import type { DocumentSheetNavigation } from "./document-preview-panel";

interface DocumentStackProps {
  items: PendingDocumentAttachment[];
  onRemove?: (id: string) => void;
  className?: string;
}

type NavigationDirection = -1 | 0 | 1;
type AnimationState = "idle" | "navigating";
type StackCard = {
  item: PendingDocumentAttachment;
  index: number;
  relativePosition: number;
  isFront: boolean;
};

const CARD_WIDTH_CSS = "min(20rem, calc(100vw - 3rem))";
const CARD_HEIGHT_PX = 56;
const MAX_VISIBLE_BACKGROUND_CARDS = 3;
const STACK_TOP_PADDING_PX = 10;
const STACK_EDGE_OFFSET_Y_PX = 20;
const STACK_EDGE_INSET_X_PX = 8;
const HOVER_PREVIEW_LIFT_PX = 42;
// Depth illusion is conveyed by Y-offset + z-index; cards do not scale down.
const NAVIGATION_SETTLE_MS = 260;
const FRONT_CARD_Z_INDEX = 50;
const BACKGROUND_CARD_Z_INDEX_BASE = 40;
const BACKGROUND_CARD_Z_INDEX_STEP = 3;
const DOCUMENT_CARD_SURFACE =
  "relative flex max-w-full items-center gap-2 rounded-md border px-2.5 py-2 text-sm";

const CARD_TRANSITION = {
  type: "spring",
  stiffness: 360,
  damping: 32,
  mass: 0.68,
} as const;

type DocumentStackTransition = typeof CARD_TRANSITION | { duration: number };

function clampActiveIndex(index: number, count: number): number {
  if (count <= 0) {
    return 0;
  }
  return Math.min(Math.max(index, 0), count - 1);
}

function wrapDocumentIndex(index: number, count: number): number {
  if (count <= 0) {
    return 0;
  }
  return ((index % count) + count) % count;
}

function DocumentTypeBadge({
  fileType,
  className,
}: {
  fileType: string;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "shrink-0 rounded-md border border-border/70 bg-muted/35 px-1.5 py-0.5 text-[10px] font-semibold text-muted-foreground dark:bg-muted/45",
        className,
      )}
    >
      {fileType}
    </span>
  );
}

function DocumentStackPreviewCard({
  attachment,
  isHovered,
  maxVisualPayloads,
}: {
  attachment: PendingDocumentAttachment;
  isHovered: boolean;
  maxVisualPayloads: number;
}) {
  const { fileType, subtitle } = documentAttachmentSummary(
    attachment,
    maxVisualPayloads,
  );

  return (
    <div
      className={cn(
        DOCUMENT_CARD_SURFACE,
        "pointer-events-none h-14 w-full max-w-none overflow-hidden border-border/70 bg-card pr-3 text-left text-card-foreground shadow-sm backdrop-blur-none transition-[border-color,background-color,box-shadow] duration-200 dark:bg-card",
        "motion-reduce:transition-none",
        isHovered
          ? "border-primary/30 shadow-md"
          : "border-border/55 text-muted-foreground/85 shadow-none",
      )}
      style={{
        height: isHovered ? CARD_HEIGHT_PX : STACK_EDGE_OFFSET_Y_PX,
        paddingBottom: isHovered ? undefined : 0,
        paddingTop: isHovered ? undefined : 0,
      }}
      aria-hidden="true"
    >
      {isHovered ? (
        <span className="flex min-w-0 flex-1 items-center gap-2">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-amber-500/10 text-amber-600 dark:text-amber-400">
            <FileText className="size-4" aria-hidden="true" />
          </span>
          <AttachmentChipBody className="gap-0">
            <span className="flex min-w-0 items-center gap-1.5">
              <AttachmentChipTitle
                className="text-xs"
                title={attachment.filename}
              >
                {attachment.filename}
              </AttachmentChipTitle>
              <DocumentTypeBadge fileType={fileType} />
            </span>
            <span
              className="truncate text-[11px] text-muted-foreground"
              title={subtitle}
            >
              {subtitle}
            </span>
          </AttachmentChipBody>
        </span>
      ) : (
        <span className="flex min-w-0 flex-1 items-center gap-1.5">
          <FileText
            className="size-3 shrink-0 text-amber-600 dark:text-amber-400"
            aria-hidden="true"
          />
          <span
            className="min-w-0 flex-1 truncate text-[11px] font-medium leading-none"
            title={attachment.filename}
          >
            {attachment.filename}
          </span>
          <DocumentTypeBadge
            fileType={fileType}
            className="px-1 py-0 text-[9px]"
          />
        </span>
      )}
    </div>
  );
}

function FrontCardSwipe({
  item,
  navigationDirection,
  reducedMotion,
  navigation,
  previewOpen,
  onPreviewOpenChange,
  onRemove,
}: {
  item: PendingDocumentAttachment;
  navigationDirection: NavigationDirection;
  reducedMotion: boolean;
  navigation?: DocumentSheetNavigation;
  previewOpen?: boolean;
  onPreviewOpenChange?: (open: boolean) => void;
  onRemove?: () => void;
}) {
  const controls = useAnimation();
  const prevItemIdRef = useRef<string | null>(null);

  useEffect(() => {
    const prevId = prevItemIdRef.current;
    prevItemIdRef.current = item.id;
    if (prevId === null || prevId === item.id || reducedMotion) {
      return;
    }
    const exitX = 0;
    const exitY =
      navigationDirection !== 0 ? -navigationDirection * 32 : 28;
    const enterX = 0;
    const enterY =
      navigationDirection !== 0 ? navigationDirection * 32 : -28;
    let cancelled = false;
    void (async () => {
      await controls.start({
        x: exitX,
        y: exitY,
        opacity: 0,
        scale: 0.92,
        transition: { duration: 0.12, ease: [0.4, 0, 1, 1] },
      });
      if (cancelled) return;
      controls.set({ x: enterX, y: enterY, opacity: 0, scale: 0.92 });
      await controls.start({
        x: 0,
        y: 0,
        opacity: 1,
        scale: 1,
        transition: { duration: 0.18, ease: [0, 0, 0.2, 1] },
      });
    })();
    return () => {
      cancelled = true;
    };
  }, [item.id, navigationDirection, reducedMotion, controls]);

  return (
    <motion.div className="w-full" initial={false} animate={controls}>
      <DocAttachmentChip
        attachment={item}
        onRemove={onRemove}
        wrapperClassName="w-full"
        className="h-14 w-full max-w-none items-center border-border/70 bg-card shadow-sm backdrop-blur-none dark:bg-card"
        navigation={navigation}
        previewOpen={previewOpen}
        onPreviewOpenChange={onPreviewOpenChange}
      />
    </motion.div>
  );
}

function getStackCardLayout({
  isFront,
  depth,
  isHoveredBackground,
  visibleBackgroundCount,
  topReserve,
}: {
  isFront: boolean;
  depth: number;
  isHoveredBackground: boolean;
  visibleBackgroundCount: number;
  topReserve: number;
}) {
  const edgeInset = isFront
    ? 0
    : Math.min(depth, visibleBackgroundCount) * STACK_EDGE_INSET_X_PX;
  const x = edgeInset;
  const baseY = isFront
    ? topReserve
    : topReserve - depth * STACK_EDGE_OFFSET_Y_PX;
  const y =
    isFront || !isHoveredBackground ? baseY : baseY - HOVER_PREVIEW_LIFT_PX;
  const scale = 1;
  const zIndex = isFront
    ? FRONT_CARD_Z_INDEX
    : BACKGROUND_CARD_Z_INDEX_BASE -
      depth * BACKGROUND_CARD_Z_INDEX_STEP +
      (isHoveredBackground ? 1 : 0);
  const opacity =
    isFront || isHoveredBackground ? 1 : Math.max(0.32, 0.7 - depth * 0.16);

  const width = edgeInset
    ? `calc(${CARD_WIDTH_CSS} - ${edgeInset * 2}px)`
    : CARD_WIDTH_CSS;

  return { opacity, scale, width, x, y, zIndex };
}

function DocumentStackCardLayer({
  card,
  visibleBackgroundCount,
  topReserve,
  transition,
  safeHoveredDocumentIndex,
  animationState,
  maxVisualPayloads,
  navigation,
  previewOpen,
  onPreviewOpenChange,
  navigationDirection,
  reducedMotion,
  onRemove,
  onActivateDocument,
  onHoverDocument,
  onNavigateDocument,
}: {
  card: StackCard;
  visibleBackgroundCount: number;
  topReserve: number;
  transition: DocumentStackTransition;
  safeHoveredDocumentIndex: number | null;
  animationState: AnimationState;
  maxVisualPayloads: number;
  navigation?: DocumentSheetNavigation;
  previewOpen?: boolean;
  onPreviewOpenChange?: (open: boolean) => void;
  navigationDirection: NavigationDirection;
  reducedMotion: boolean;
  onRemove?: (id: string) => void;
  onActivateDocument: (index: number) => void;
  onHoverDocument: (index: number, hovered: boolean) => void;
  onNavigateDocument: (direction: Exclude<NavigationDirection, 0>) => void;
}) {
  const { item, index, relativePosition, isFront } = card;
  const isHoveredBackground =
    safeHoveredDocumentIndex === index && !isFront && animationState === "idle";
  const { opacity, scale, width, x, y, zIndex } = getStackCardLayout({
    isFront,
    depth: relativePosition,
    isHoveredBackground,
    visibleBackgroundCount,
    topReserve,
  });
  const hitLayout = getStackCardLayout({
    isFront,
    depth: relativePosition,
    isHoveredBackground: false,
    visibleBackgroundCount,
    topReserve,
  });
  const activateBackgroundCard = (): void => onActivateDocument(index);

  return (
    <Fragment>
      <motion.div
        className="absolute top-0 left-0 rounded-md"
        initial={false}
        animate={{ x, y, scale, opacity }}
        transition={transition}
        style={{
          width,
          zIndex,
          transformOrigin: "top center",
        }}
      >
        {isFront ? (
          <FrontCardSwipe
            item={item}
            navigationDirection={navigationDirection}
            reducedMotion={reducedMotion}
            navigation={navigation}
            previewOpen={previewOpen}
            onPreviewOpenChange={onPreviewOpenChange}
            onRemove={onRemove ? () => onRemove(item.id) : undefined}
          />
        ) : (
          <DocumentStackPreviewCard
            attachment={item}
            isHovered={isHoveredBackground}
            maxVisualPayloads={maxVisualPayloads}
          />
        )}
      </motion.div>
      {isFront ? null : (
        <motion.button
          type="button"
          className="absolute top-0 left-0 cursor-pointer rounded-md bg-transparent focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          initial={false}
          animate={{ x: hitLayout.x, y: hitLayout.y, scale: hitLayout.scale }}
          transition={transition}
          style={{
            width: hitLayout.width,
            height: STACK_EDGE_OFFSET_Y_PX,
            zIndex: 60 - relativePosition,
            transformOrigin: "top center",
          }}
          onPointerEnter={() => onHoverDocument(index, true)}
          onPointerLeave={() => onHoverDocument(index, false)}
          onPointerCancel={() => onHoverDocument(index, false)}
          onFocus={() => onHoverDocument(index, true)}
          onBlur={() => onHoverDocument(index, false)}
          onClick={(event) => {
            event.stopPropagation();
            activateBackgroundCard();
          }}
          onKeyDown={(event) => {
            if (
              event.defaultPrevented ||
              event.altKey ||
              event.ctrlKey ||
              event.metaKey
            ) {
              return;
            }

            if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
              event.preventDefault();
              onNavigateDocument(-1);
            } else if (
              event.key === "ArrowDown" ||
              event.key === "ArrowRight"
            ) {
              event.preventDefault();
              onNavigateDocument(1);
            }
          }}
          aria-label={`Make ${item.filename} the front document`}
          title={item.filename}
        />
      )}
    </Fragment>
  );
}

/**
 * State model for the document stack:
 * - activeDocumentIndex controls the front card only.
 * - hoveredDocumentIndex controls temporary background-card preview only.
 * - orderedDocuments is the stable render order from props.
 * - navigationDirection and animationState are arrow/click transition hints.
 */
export function DocumentStack({
  items,
  onRemove,
  className,
}: DocumentStackProps) {
  const reducedMotion = useReducedMotion();
  const maxVisualPayloads = useChatRuntimeStore(
    (s) => s.docExtract.maxVisualPayloads,
  );
  const orderedDocuments = useMemo(() => items, [items]);
  const count = orderedDocuments.length;
  const [activeDocumentIndex, setActiveDocumentIndex] = useState(0);
  const [hoveredDocumentIndex, setHoveredDocumentIndex] = useState<
    number | null
  >(null);
  const [navigationDirection, setNavigationDirection] =
    useState<NavigationDirection>(0);
  const [animationState, setAnimationState] = useState<AnimationState>("idle");
  const [previewOpen, setPreviewOpen] = useState(false);
  const fieldsetRef = useRef<HTMLFieldSetElement>(null);

  const safeActiveDocumentIndex = clampActiveIndex(activeDocumentIndex, count);
  const safeHoveredDocumentIndex =
    hoveredDocumentIndex !== null &&
    hoveredDocumentIndex >= 0 &&
    hoveredDocumentIndex < count &&
    hoveredDocumentIndex !== safeActiveDocumentIndex
      ? hoveredDocumentIndex
      : null;
  const activeDocument = orderedDocuments[safeActiveDocumentIndex];

  useEffect(() => {
    if (animationState !== "navigating") {
      return;
    }
    const timeout = window.setTimeout(
      () => {
        setAnimationState("idle");
        setNavigationDirection(0);
      },
      reducedMotion ? 0 : NAVIGATION_SETTLE_MS,
    );
    return () => window.clearTimeout(timeout);
  }, [animationState, reducedMotion]);

  const activateDocument = useCallback(
    (index: number) => {
      const targetIndex = clampActiveIndex(index, count);
      const targetDocument = orderedDocuments[targetIndex];
      if (targetIndex === safeActiveDocumentIndex) {
        return;
      }
      if (!targetDocument) {
        return;
      }
      setHoveredDocumentIndex(null);
      setNavigationDirection(0);
      setAnimationState("navigating");
      setActiveDocumentIndex(targetIndex);
    },
    [count, orderedDocuments, safeActiveDocumentIndex],
  );

  const cycleActiveDocument = useCallback(
    (direction: Exclude<NavigationDirection, 0>) => {
      if (count <= 1) {
        return;
      }
      const nextActiveDocumentIndex = wrapDocumentIndex(
        safeActiveDocumentIndex + direction,
        count,
      );
      setHoveredDocumentIndex(null);
      setNavigationDirection(direction);
      setAnimationState("navigating");
      setActiveDocumentIndex(nextActiveDocumentIndex);
    },
    [count, safeActiveDocumentIndex],
  );

  useEffect(() => {
    const el = fieldsetRef.current;
    if (!el || count <= 1) return;
    let lastWheelAt = 0;
    const handleWheel = (event: globalThis.WheelEvent) => {
      const dy = event.deltaY;
      if (Math.abs(dy) < 4 || Math.abs(dy) <= Math.abs(event.deltaX)) {
        return;
      }
      const now = performance.now();
      if (now - lastWheelAt < 260) {
        event.preventDefault();
        return;
      }
      lastWheelAt = now;
      event.preventDefault();
      cycleActiveDocument(dy > 0 ? 1 : -1);
    };
    let touchStartY: number | null = null;
    let touchSwiped = false;
    const handleTouchStart = (event: globalThis.TouchEvent) => {
      touchStartY = event.touches[0]?.clientY ?? null;
      touchSwiped = false;
    };
    const handleTouchMove = (event: globalThis.TouchEvent) => {
      if (touchStartY === null || touchSwiped) return;
      const currentY = event.touches[0]?.clientY;
      if (currentY === undefined) return;
      const dy = currentY - touchStartY;
      if (Math.abs(dy) >= 32) {
        touchSwiped = true;
        cycleActiveDocument(dy < 0 ? 1 : -1);
      }
    };
    const handleTouchEnd = () => {
      touchStartY = null;
      touchSwiped = false;
    };
    el.addEventListener("wheel", handleWheel, { passive: false });
    el.addEventListener("touchstart", handleTouchStart, { passive: true });
    el.addEventListener("touchmove", handleTouchMove, { passive: true });
    el.addEventListener("touchend", handleTouchEnd, { passive: true });
    el.addEventListener("touchcancel", handleTouchEnd, { passive: true });
    return () => {
      el.removeEventListener("wheel", handleWheel);
      el.removeEventListener("touchstart", handleTouchStart);
      el.removeEventListener("touchmove", handleTouchMove);
      el.removeEventListener("touchend", handleTouchEnd);
      el.removeEventListener("touchcancel", handleTouchEnd);
    };
  }, [count, cycleActiveDocument]);

  const updateHoveredDocument = useCallback(
    (index: number, hovered: boolean) => {
      if (animationState !== "idle") {
        return;
      }
      setHoveredDocumentIndex((current) => {
        if (hovered) {
          return index === safeActiveDocumentIndex ? null : index;
        }
        return current === index ? null : current;
      });
    },
    [animationState, safeActiveDocumentIndex],
  );

  const handleStackKeyDown = useCallback(
    (event: KeyboardEvent<HTMLFieldSetElement>) => {
      if (
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey
      ) {
        return;
      }

      if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        event.preventDefault();
        cycleActiveDocument(-1);
      } else if (
        event.key === "ArrowDown" ||
        event.key === "ArrowRight"
      ) {
        event.preventDefault();
        cycleActiveDocument(1);
      }
    },
    [cycleActiveDocument],
  );

  const navigation = useMemo<DocumentSheetNavigation>(
    () => ({
      currentIndex: safeActiveDocumentIndex,
      totalCount: count,
      onNavigate: (direction) => cycleActiveDocument(direction),
    }),
    [count, cycleActiveDocument, safeActiveDocumentIndex],
  );

  if (count === 0) {
    return null;
  }

  if (count === 1) {
    const singleDocument = orderedDocuments[0];
    if (!singleDocument) {
      return null;
    }
    return (
      <DocAttachmentChip
        attachment={singleDocument}
        onRemove={onRemove ? () => onRemove(singleDocument.id) : undefined}
        className={cn("max-w-80", className)}
      />
    );
  }

  if (!activeDocument) {
    return null;
  }

  const visibleBackgroundCount = Math.min(
    count - 1,
    MAX_VISIBLE_BACKGROUND_CARDS,
  );
  const topReserve =
    STACK_TOP_PADDING_PX + visibleBackgroundCount * STACK_EDGE_OFFSET_Y_PX;
  const transition = reducedMotion ? { duration: 0 } : CARD_TRANSITION;
  const stackCards: StackCard[] = Array.from(
    { length: visibleBackgroundCount + 1 },
    (_, relativePosition) => {
      const index = wrapDocumentIndex(
        safeActiveDocumentIndex + relativePosition,
        count,
      );
      const item = orderedDocuments[index];
      if (!item) {
        return null;
      }
      return {
        item,
        index,
        relativePosition,
        isFront: relativePosition === 0,
      };
    },
  ).filter((card): card is StackCard => card !== null);
  const hiddenBackgroundCount = Math.max(0, count - 1 - visibleBackgroundCount);
  const navAnnouncement =
    navigationDirection === 1
      ? "Next document"
      : navigationDirection === -1
        ? "Previous document"
        : "Selected document";

  return (
    <div className={cn("inline-flex max-w-full items-end gap-1.5", className)}>
      <fieldset
        ref={fieldsetRef}
        className="relative m-0 min-w-0 shrink-0 touch-pan-x overflow-visible border-0 p-0 [min-inline-size:0]"
        style={{
          width: CARD_WIDTH_CSS,
          height: CARD_HEIGHT_PX + topReserve,
        }}
        onKeyDown={handleStackKeyDown}
      >
        <legend className="sr-only">{`${count} attached documents`}</legend>
        <span className="sr-only" aria-live="polite">
          {`${navAnnouncement}: ${activeDocument.filename}. ${count} documents attached.`}
        </span>

        {stackCards.map((card) => (
          <DocumentStackCardLayer
            key={card.relativePosition}
            card={card}
            visibleBackgroundCount={visibleBackgroundCount}
            topReserve={topReserve}
            transition={transition}
            safeHoveredDocumentIndex={safeHoveredDocumentIndex}
            animationState={animationState}
            maxVisualPayloads={maxVisualPayloads}
            navigation={navigation}
            previewOpen={card.isFront ? previewOpen : undefined}
            onPreviewOpenChange={
              card.isFront ? setPreviewOpen : undefined
            }
            navigationDirection={navigationDirection}
            reducedMotion={reducedMotion ?? false}
            onRemove={onRemove}
            onActivateDocument={activateDocument}
            onHoverDocument={updateHoveredDocument}
            onNavigateDocument={cycleActiveDocument}
          />
        ))}

      </fieldset>

      <div
        className="relative flex h-14 shrink-0 flex-col items-center justify-center gap-1"
        aria-label="Document navigation"
      >
        <button
          type="button"
          className="inline-flex size-7 items-center justify-center rounded-full border border-border/70 bg-background p-1 text-muted-foreground shadow-sm transition-colors hover:bg-accent hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:bg-card"
          onClick={() => cycleActiveDocument(-1)}
          aria-label="Previous attached document"
          title="Previous document"
        >
          <ChevronUpIcon className="size-3.5" aria-hidden="true" />
        </button>
        <button
          type="button"
          className="inline-flex size-7 items-center justify-center rounded-full border border-border/70 bg-background p-1 text-muted-foreground shadow-sm transition-colors hover:bg-accent hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:bg-card"
          onClick={() => cycleActiveDocument(1)}
          aria-label="Next attached document"
          title="Next document"
        >
          <ChevronDownIcon className="size-3.5" aria-hidden="true" />
        </button>
        {hiddenBackgroundCount > 0 ? (
          <span
            className="pointer-events-none absolute left-1/2 inline-flex h-4 min-w-4 -translate-x-1/2 items-center justify-center rounded-full border border-border/70 bg-background px-1 text-[9px] font-semibold tabular-nums text-muted-foreground shadow-sm dark:bg-card"
            style={{
              top: -(
                (visibleBackgroundCount - 0.5) * STACK_EDGE_OFFSET_Y_PX +
                8
              ),
            }}
            aria-label={`${hiddenBackgroundCount} more attached`}
          >
            +{hiddenBackgroundCount}
          </span>
        ) : null}
      </div>
    </div>
  );
}
