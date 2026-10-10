// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A collapsible that never measures its content. Radix's CollapsibleContentImpl calls
// getBoundingClientRect on every open change regardless of whether anything consumes the result,
// forcing a full-document layout. With grid 0fr/1fr nothing needs measuring. The wrapper must be
// min-height: 0 and overflow: hidden or the row never collapses.

import { cn } from "@/lib/utils";
import * as React from "react";

type UnmeasuredCollapsibleContextValue = {
  open: boolean;
  disabled?: boolean;
  contentId: string;
  onOpenToggle: () => void;
};

const UnmeasuredCollapsibleContext =
  React.createContext<UnmeasuredCollapsibleContextValue | null>(null);

function useUnmeasuredCollapsibleContext(consumer: string) {
  const context = React.useContext(UnmeasuredCollapsibleContext);
  if (!context) {
    throw new Error(`\`${consumer}\` must be used within \`UnmeasuredCollapsible\``);
  }
  return context;
}

function getState(open: boolean) {
  return open ? "open" : "closed";
}

type UnmeasuredCollapsibleProps = Omit<
  React.ComponentPropsWithoutRef<"div">,
  "onToggle"
> & {
  open?: boolean;
  defaultOpen?: boolean;
  disabled?: boolean;
  onOpenChange?: (open: boolean) => void;
};

const UnmeasuredCollapsible = React.forwardRef<
  HTMLDivElement,
  UnmeasuredCollapsibleProps
>(
  (
    {
      open: openProp,
      defaultOpen = false,
      disabled,
      onOpenChange,
      children,
      ...props
    },
    forwardedRef,
  ) => {
    const [uncontrolledOpen, setUncontrolledOpen] = React.useState(defaultOpen);
    const isControlled = openProp !== undefined;
    const open = isControlled ? openProp : uncontrolledOpen;
    const contentId = React.useId();

    // Closes over the committed `open`, not a render-time ref: an abandoned render can leave a ref
    // holding a value never committed.
    const onOpenToggle = React.useCallback(() => {
      const next = !open;
      if (!isControlled) {
        setUncontrolledOpen(next);
      }
      onOpenChange?.(next);
    }, [open, isControlled, onOpenChange]);

    const context = React.useMemo<UnmeasuredCollapsibleContextValue>(
      () => ({ open, disabled, contentId, onOpenToggle }),
      [open, disabled, contentId, onOpenToggle],
    );

    return (
      <UnmeasuredCollapsibleContext.Provider value={context}>
        <div
          data-slot="collapsible"
          data-state={getState(open)}
          data-disabled={disabled ? "" : undefined}
          {...props}
          ref={forwardedRef}
        >
          {children}
        </div>
      </UnmeasuredCollapsibleContext.Provider>
    );
  },
);
UnmeasuredCollapsible.displayName = "UnmeasuredCollapsible";

const UnmeasuredCollapsibleTrigger = React.forwardRef<
  HTMLButtonElement,
  React.ComponentPropsWithoutRef<"button">
>(({ onClick, ...props }, forwardedRef) => {
  const context = useUnmeasuredCollapsibleContext("UnmeasuredCollapsibleTrigger");
  return (
    <button
      type="button"
      aria-controls={context.contentId}
      aria-expanded={context.open || false}
      data-slot="collapsible-trigger"
      data-state={getState(context.open)}
      data-disabled={context.disabled ? "" : undefined}
      disabled={context.disabled}
      {...props}
      ref={forwardedRef}
      onClick={(event) => {
        onClick?.(event);
        if (!event.defaultPrevented) {
          context.onOpenToggle();
        }
      }}
    />
  );
});
UnmeasuredCollapsibleTrigger.displayName = "UnmeasuredCollapsibleTrigger";

type UnmeasuredCollapsibleContentProps = React.ComponentPropsWithoutRef<"div"> & {
  // Fallback for when `transitionend` never arrives (display: none, background tab, reduced
  // motion); armed with CLOSE_FALLBACK_MARGIN_MS so it normally loses.
  closeDurationMs?: number;
  // Rendered while closed, like Radix `forceMount`, so callers drive presence.
  forceMount?: boolean;
};

const DEFAULT_CLOSE_DURATION_MS = 200;

// The backstop is armed before the `0fr` class commits, so without a margin it would always
// win and unmount early.
export const CLOSE_FALLBACK_MARGIN_MS = 50;

const UnmeasuredCollapsibleContent = React.forwardRef<
  HTMLDivElement,
  UnmeasuredCollapsibleContentProps
>(
  (
    {
      className,
      children,
      closeDurationMs = DEFAULT_CLOSE_DURATION_MS,
      forceMount,
      ...props
    },
    forwardedRef,
  ) => {
    const context = useUnmeasuredCollapsibleContext("UnmeasuredCollapsibleContent");
    const open = context.open;

    // `mounted` = presence through the close transition; `expanded` = `0fr` vs `1fr` row size.
    const [mounted, setMounted] = React.useState(open);
    const [expanded, setExpanded] = React.useState(open);
    const nodeRef = React.useRef<HTMLDivElement>(null);
    const composedRef = React.useCallback(
      (node: HTMLDivElement | null) => {
        nodeRef.current = node;
        if (typeof forwardedRef === "function") {
          forwardedRef(node);
        } else if (forwardedRef) {
          forwardedRef.current = node;
        }
      },
      [forwardedRef],
    );

    // Mount at `0fr`, then flip to `1fr` a frame later: a same-frame class change would snap open.
    React.useEffect(() => {
      if (open) {
        setMounted(true);
        let inner = 0;
        const outer = requestAnimationFrame(() => {
          inner = requestAnimationFrame(() => setExpanded(true));
        });
        return () => {
          cancelAnimationFrame(outer);
          cancelAnimationFrame(inner);
        };
      }
      setExpanded(false);
      return undefined;
    }, [open]);

    // `transitionend` bubbles and fires per property, so filter it; the timeout is the backstop.
    React.useEffect(() => {
      if (open || !mounted) {
        return undefined;
      }
      const node = nodeRef.current;
      const finish = () => setMounted(false);
      const onTransitionEnd = (event: TransitionEvent) => {
        if (event.target === node && event.propertyName === "grid-template-rows") {
          finish();
        }
      };
      node?.addEventListener("transitionend", onTransitionEnd);
      const timeout = window.setTimeout(finish, closeDurationMs + CLOSE_FALLBACK_MARGIN_MS);
      return () => {
        node?.removeEventListener("transitionend", onTransitionEnd);
        window.clearTimeout(timeout);
      };
    }, [open, mounted, closeDurationMs]);

    const present = forceMount || mounted;

    return (
      <div
        data-slot="collapsible-content"
        data-state={getState(open)}
        data-disabled={context.disabled ? "" : undefined}
        id={context.contentId}
        hidden={!present}
        {...props}
        ref={composedRef}
        className={cn(
          // `[hidden]` loses to any author `display`, so switch the display utility itself.
          present ? "grid" : "hidden",
          "transition-[grid-template-rows]",
          expanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
          className,
        )}
      >
        <div className="min-h-0 overflow-hidden">{present && children}</div>
      </div>
    );
  },
);
UnmeasuredCollapsibleContent.displayName = "UnmeasuredCollapsibleContent";

export {
  UnmeasuredCollapsible,
  UnmeasuredCollapsibleContent,
  UnmeasuredCollapsibleTrigger,
};
