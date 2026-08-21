// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A collapsible that never measures its content.
//
// WHY THIS EXISTS, and why swapping the keyframes alone would not have worked.
//
// Radix's `CollapsibleContentImpl` runs this on every `open` change (react-collapsible 1.1.12,
// dist/index.mjs, verbatim shape):
//
//     useLayoutEffect(() => {
//       const node = ref.current;
//       if (node) {
//         node.style.transitionDuration = "0s";     // write
//         node.style.animationName = "none";        // write
//         const rect = node.getBoundingClientRect();// READ -> synchronous layout
//         heightRef.current = rect.height;
//         widthRef.current = rect.width;
//         ...                                       // write back
//       }
//     }, [context.open, present]);
//
// It publishes the result as `--radix-collapsible-content-height`, which the
// `animate-collapsible-down` / `animate-collapsible-up` keyframes consume. The read is
// UNCONDITIONAL: it does not check whether any stylesheet references the variable. So replacing the
// height keyframes with a `grid-template-rows: 0fr -> 1fr` transition removes the CONSUMER of the
// measurement and leaves the measurement itself untouched. The forced layout stays, and with it the
// full-document relayout that Blink charges for it, because Blink's layout is O(total layout
// objects) and not O(dirty objects).
//
// Hence a local primitive. It keeps Radix's public shape -- `data-state`, `data-disabled`,
// `aria-expanded`, `aria-controls`, the generated content id, `hidden` when closed, and children
// unmounted while closed -- and drops only the measurement, because with `0fr -> 1fr` there is
// nothing left to measure: `1fr` resolves against the content on its own, every frame, including
// while the content is still streaming in.
//
// Two things the grid technique requires, and it silently does not collapse without them:
//   * the animating child must be `min-height: 0`, or its automatic minimum size floors the row at
//     the content's height and `0fr` never reaches zero;
//   * it must be `overflow: hidden`, or the content paints outside the zero-height row.
// Both live on the wrapper this component renders, not on the caller's node, so a caller cannot
// forget them. That wrapper is one element more than Radix renders, which is a real DOM difference
// and is why the flag that selects this path also needs screenshots, not just the parity digest.

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

    // Closes over the committed `open`, and deliberately not over a ref written during
    // render. A render can be abandoned or suspended, and React does not roll a ref back when
    // that happens, so a ref assigned here can hold a value that was never committed while the
    // trigger the user is looking at still shows the old one; the click would then toggle away
    // from the visible state. The ref bought nothing anyway: `context` below is memoized on
    // `open`, so it is rebuilt on every open change whatever this callback's identity is.
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
  // Upper bound on how long the caller's `grid-template-rows` transition can run. Children unmount
  // when the transition ends; this is the fallback for the cases where `transitionend` never
  // arrives -- a closed pane inside a `display: none` ancestor, a tab in the background, or
  // reduced motion collapsing the duration to 0.01ms in a browser that then skips the event.
  // The timer is armed at this plus `CLOSE_FALLBACK_MARGIN_MS`, so it stays a fallback rather
  // than the path that normally wins.
  closeDurationMs?: number;
  // Rendered even while closed. Matches Radix's `forceMount`, and like Radix's it exists so a
  // caller can drive its own presence.
  forceMount?: boolean;
};

const DEFAULT_CLOSE_DURATION_MS = 200;

// The backstop is armed in the same passive-effect flush that queues `setExpanded(false)`, so its
// countdown starts before React commits the `0fr` class and before the browser starts the
// transition. Armed at exactly `closeDurationMs` it would therefore always win the race it is
// supposed to lose, unmounting the children a few milliseconds early -- and more than that when a
// busy main thread delays the commit. The margin puts it back behind `transitionend`.
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

    // `mounted` is presence: true from the moment the pane starts opening until the close
    // transition has finished, so the content is still there to animate out.
    // `expanded` is the row size: it drives `0fr` vs `1fr`.
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

    // Opening: mount at `0fr` first, then flip to `1fr` once the browser has computed a style for
    // the mounted node. Two frames, because a class change in the same frame as the insertion has
    // no before-change style to transition FROM and would snap open. This costs one frame of
    // animation start, never a layout read.
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

    // Closing: unmount the children when the row has finished shrinking. `transitionend` bubbles
    // from descendants and fires once per property, so both are filtered; the timeout is the
    // backstop described on `closeDurationMs`.
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
          // `hidden` alone would not hide this: the UA sheet's `[hidden] { display: none }` loses
          // to any author `display`, and `grid` is an author declaration. So the closed state
          // switches the display utility itself rather than relying on the attribute.
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
