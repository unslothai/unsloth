// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tooltip as TooltipPrimitive } from "radix-ui";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  useSyncExternalStore,
} from "react";
import type * as React from "react";

import { isBlockedByActiveModal } from "@/components/ui/tooltip-modal-layer";
import { resolveTooltipOpen } from "@/components/ui/tooltip-open-state";
import { cn } from "@/lib/utils";

type ToggleFn = () => void;
const TooltipToggleCtx = createContext<ToggleFn | null>(null);
const TooltipTriggerElementCtx = createContext<(element: HTMLElement | null) => void>(
  () => undefined,
);

// Radix sets body pointer-events to none while a modal layer is up. That is also
// when a hovered trigger stops receiving pointerleave, so a tooltip already on
// screen hangs over the dialog with nothing able to close it. One observer
// serves every tooltip.
let modalLayerUp = false;
const modalLayerListeners = new Set<() => void>();
let modalLayerObserver: MutationObserver | null = null;

function readModalLayer(): void {
  modalLayerUp = document.body.style.pointerEvents === "none";
  for (const listener of modalLayerListeners) listener();
}

function readPointerEvents(style: string | null): string {
  return (
    /(?:^|;)\s*pointer-events\s*:\s*([^;]+)/.exec(style ?? "")?.[1]?.trim() ??
    ""
  );
}

function readRelevantLayerMutations(records: MutationRecord[]): void {
  const pointerEventsChanged = records.some(
    (record) =>
      readPointerEvents(record.oldValue) !==
      readPointerEvents(
        (record.target as Element).getAttribute?.("style") ?? null,
      ),
  );
  if (pointerEventsChanged) readModalLayer();
}

function subscribeModalLayer(listener: () => void): () => void {
  modalLayerListeners.add(listener);
  if (!modalLayerObserver && typeof MutationObserver !== "undefined") {
    modalLayerObserver = new MutationObserver(readRelevantLayerMutations);
    modalLayerObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ["style"],
      attributeOldValue: true,
      subtree: true,
    });
    readModalLayer();
  }
  return () => {
    modalLayerListeners.delete(listener);
  };
}

function getModalLayer(): boolean {
  return modalLayerUp;
}

function assignRef<T>(ref: React.Ref<T> | undefined, value: T | null): void {
  if (typeof ref === "function") {
    ref(value);
  } else if (ref) {
    ref.current = value;
  }
}

type ModalBlockStore = {
  getSnapshot: () => boolean;
  getTriggerElement: () => HTMLElement | null;
  setTriggerElement: (element: HTMLElement | null) => void;
  subscribe: (listener: () => void) => () => void;
};

function createModalBlockStore(): ModalBlockStore {
  let triggerElement: HTMLElement | null = null;
  let blocked = false;
  let unsubscribeModalLayer: (() => void) | null = null;
  const listeners = new Set<() => void>();

  const update = () => {
    // No trigger to ask (a child that drops the ref) falls back to the whole
    // modal: leaving it shut beats leaving it stranded over the dialog.
    const next =
      getModalLayer() &&
      (triggerElement === null || isBlockedByActiveModal(triggerElement));
    if (next === blocked) return;
    blocked = next;
    for (const listener of listeners) listener();
  };

  return {
    getSnapshot: () => blocked,
    getTriggerElement: () => triggerElement,
    setTriggerElement: (element) => {
      triggerElement = element;
      update();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      if (listeners.size === 1) {
        unsubscribeModalLayer = subscribeModalLayer(update);
      }
      update();
      return () => {
        listeners.delete(listener);
        if (listeners.size === 0) {
          unsubscribeModalLayer?.();
          unsubscribeModalLayer = null;
        }
      };
    },
  };
}

function getServerModalBlock(): boolean {
  return false;
}

/** Tap-to-pin is for touch, which has no hover. The click's own pointerType is
 * the only thing that answers for the pointer actually used: the media query
 * reports the primary device, so on a hybrid it mislabels every event from the
 * other one. Keyboard activation reports "", which correctly does not pin. */
function isTouchClick(event: React.MouseEvent): boolean {
  const pointerType = (event.nativeEvent as Partial<PointerEvent>).pointerType;
  if (typeof pointerType === "string") return pointerType === "touch";
  // No PointerEvent (older WebViews): fall back to the device class.
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches
  );
}

// Default to instant open (no hover delay). Most tooltips in the app —
// chat-area icon labels, sidebar nav labels, the context/token
// calculators — should feel snappy. Consumers that want a delay still
// pass an explicit `delayDuration` prop.
function TooltipProvider({
  delayDuration = 0,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
  return (
    <TooltipPrimitive.Provider
      data-slot="tooltip-provider"
      delayDuration={delayDuration}
      {...props}
    />
  );
}

function Tooltip({
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Root>) {
  const isControlled = controlledOpen !== undefined;
  // Radix's own open state is never handed back once a modal forces the
  // tooltip shut, so hover is tracked here and `open` is always supplied.
  // Falling back to `undefined` would re-expose a `true` from before the
  // dialog, with the pointer somewhere else entirely.
  const [hoverOpen, setHoverOpen] = useState(false);
  const [clickOpen, setClickOpen] = useState(false);
  // A controlled tooltip's owner cannot be reset from here, and it never saw
  // the pointerleave a modal swallowed, so its `open` is still true when the
  // modal closes. Stay shut until the owner says false at least once.
  const [dismissedUntilOwnerResets, setDismissedUntilOwnerResets] =
    useState(false);
  const [modalBlockStore] = useState(createModalBlockStore);
  const blocked = useSyncExternalStore(
    modalBlockStore.subscribe,
    modalBlockStore.getSnapshot,
    getServerModalBlock,
  );

  const onOpenChange = useCallback(
    (nextOpen: boolean) => {
      setHoverOpen(nextOpen);
      if (!nextOpen) setClickOpen(false);
      controlledOnOpenChange?.(nextOpen);
    },
    [controlledOnOpenChange],
  );

  const toggle = useCallback(() => {
    setClickOpen((prev) => !prev);
  }, []);

  // Drop what is open when a modal takes over, so closing the dialog does not
  // bring the tooltip back on its own. Owners of a controlled tooltip are told,
  // since their state (a `hovered` flag) never sees the pointerleave either.
  useEffect(() => {
    if (!blocked) return;
    setHoverOpen(false);
    setClickOpen(false);
    setDismissedUntilOwnerResets(true);
    controlledOnOpenChange?.(false);
  }, [blocked, controlledOnOpenChange]);

  // `panel-resize-handle.tsx` passes `open` with no onOpenChange, so telling it
  // does nothing and its `hovered` stays true. Releasing on its say-so alone
  // would put the tooltip back with the pointer somewhere else entirely.
  useEffect(() => {
    if (controlledOpen === false) setDismissedUntilOwnerResets(false);
  }, [controlledOpen]);

  // A pin must not outlive its interaction: once a dialog covers the trigger,
  // pointerleave never fires and Radix can no longer close it. Presses on a
  // trigger are skipped so tapping the same one still toggles it shut.
  useEffect(() => {
    if (!clickOpen) return;
    const release = (event: Event) => {
      const target = event.target as Node | null;
      // The trigger is matched by element, not by data-slot: an `asChild` child
      // can drop the attribute (a component that does not spread props), and
      // then a press on this very trigger read as outside, cleared the pin, and
      // the click handler toggled it straight back on.
      if (target && modalBlockStore.getTriggerElement()?.contains(target)) {
        return;
      }
      if (
        (target as Element | null)?.closest?.('[data-slot="tooltip-content"]')
      ) {
        return;
      }
      setClickOpen(false);
    };
    const releaseOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setClickOpen(false);
    };
    const releaseNow = () => setClickOpen(false);
    document.addEventListener("pointerdown", release, true);
    document.addEventListener("keydown", releaseOnEscape, true);
    window.addEventListener("blur", releaseNow);
    return () => {
      document.removeEventListener("pointerdown", release, true);
      document.removeEventListener("keydown", releaseOnEscape, true);
      window.removeEventListener("blur", releaseNow);
    };
  }, [clickOpen, modalBlockStore]);

  return (
    <TooltipTriggerElementCtx.Provider value={modalBlockStore.setTriggerElement}>
      <TooltipToggleCtx.Provider value={isControlled ? null : toggle}>
        <TooltipPrimitive.Root
          data-slot="tooltip"
          // Always a boolean, so Radix keeps no separate open state that could
          // survive a modal and resurface after it closes.
          open={resolveTooltipOpen({
            blocked,
            controlledOpen,
            dismissedUntilOwnerResets,
            hoverOpen,
            clickOpen,
          })}
          onOpenChange={onOpenChange}
          {...props}
        />
      </TooltipToggleCtx.Provider>
    </TooltipTriggerElementCtx.Provider>
  );
}

function TooltipTrigger({
  onClick,
  ref,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Trigger>) {
  const toggle = useContext(TooltipToggleCtx);
  const setTriggerElement = useContext(TooltipTriggerElementCtx);

  // The trigger says which layer this tooltip belongs to, and unlike the
  // content it is mounted the whole time, so the answer never goes missing.
  const triggerRef = useCallback(
    (el: HTMLElement | null) => {
      assignRef(ref, el);
      setTriggerElement(el);
    },
    [ref, setTriggerElement],
  );

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLButtonElement>) => {
      // Run the composed handler first: when this trigger wraps another Radix
      // trigger (e.g. DialogTrigger around an attachment tile), that trigger's
      // action is skipped if the event is already default-prevented.
      onClick?.(e);
      // With a mouse, hover already shows it and pinning only strands it. Let
      // Radix's own close-on-click run instead. `toggle` is absent when the
      // consumer controls `open`, where a pin would be dead state anyway.
      if (!toggle || !isTouchClick(e)) return;
      // preventDefault keeps Radix Tooltip's internal close-on-click from
      // undoing the tap-toggle below (its composed handler checks it).
      e.preventDefault();
      toggle();
    },
    [toggle, onClick],
  );

  return (
    <TooltipPrimitive.Trigger
      ref={triggerRef}
      data-slot="tooltip-trigger"
      onClick={handleClick}
      {...props}
    />
  );
}

type TooltipVariant = "default" | "rich" | "none";

// `default` applies the compact black-pill styling shared with the
// sidebar/chat icon labels. `rich` opts into the larger multi-row
// popover surface used for timing/context breakdowns. `none` is an
// escape hatch for tooltips that need to bring their own surface.
function TooltipContent({
  variant = "default",
  className,
  sideOffset = 0,
  children,
  ref,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content> & {
  variant?: TooltipVariant;
}) {
  // Single-line compact tooltips render as a full pill; wrapped ones keep
  // the squarer corners so tall pills do not look like capsules. A ref
  // callback measures on mount: Radix mounts the portal content without
  // re-rendering this wrapper, so an effect here would never see the node.
  const contentRef = useCallback(
    (el: React.ComponentRef<typeof TooltipPrimitive.Content> | null) => {
      assignRef(ref, el);
      if (!el || variant !== "default") return;
      const cs = getComputedStyle(el);
      const lineHeight = Number.parseFloat(cs.lineHeight) || 16;
      const innerHeight =
        el.clientHeight -
        Number.parseFloat(cs.paddingTop) -
        Number.parseFloat(cs.paddingBottom);
      el.classList.toggle("rounded-full!", innerHeight < lineHeight * 1.5);
    },
    [ref, variant],
  );
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        ref={contentRef}
        data-slot="tooltip-content"
        sideOffset={sideOffset}
        className={cn(
          "z-[999999] w-fit max-w-xs",
          variant === "default" && "tooltip-compact",
          variant === "rich" && "tooltip-rich",
          className,
        )}
        {...props}
      >
        {children}
      </TooltipPrimitive.Content>
    </TooltipPrimitive.Portal>
  );
}

export { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger };
