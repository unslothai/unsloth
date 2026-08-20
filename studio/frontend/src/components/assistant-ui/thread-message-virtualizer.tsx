// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

/**
 * Windowed replacement for `<ThreadPrimitive.Messages>`.
 *
 * `ThreadPrimitive.Messages` builds one element per message and keeps every one of them mounted for
 * the life of the thread, so a long thread pays for every message it has ever rendered on every
 * layout, restyle and hit test. This mounts only the messages near the viewport.
 *
 * Gated by THREAD_MESSAGE_VIRTUALIZATION_ENABLED, which is off. See thread-feature-flags.ts for what
 * has to land before it goes on.
 */

import { ThreadPrimitive, useAui, useAuiState } from "@assistant-ui/react";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  type ComponentProps,
  type FC,
  useCallback,
  useLayoutEffect,
  useState,
} from "react";

import {
  THREAD_MESSAGE_ANCHORING,
  THREAD_MESSAGE_ESTIMATE_SIZE_PX,
  THREAD_MESSAGE_OVERSCAN,
  messageKeyAt,
  scrollMarginFor,
} from "./thread-message-virtualizer-policy";

type MessageComponents = ComponentProps<
  typeof ThreadPrimitive.MessageByIndex
>["components"];

export const VirtualizedThreadMessages: FC<{
  /** The scroll element, owned by useIntentAwareAutoScroll and mirrored by Thread. */
  scrollElement: HTMLElement | null;
  /**
   * Must be hoisted by the caller. ThreadPrimitive.MessageByIndex is memoized on its index and on
   * each component in this object, so a fresh object per render defeats the memo and re-renders
   * every mounted message body.
   */
  components: MessageComponents;
}> = ({ scrollElement, components }) => {
  const aui = useAui();
  // Length only, which is what ThreadPrimitive.Messages subscribes to. Subscribing to the array
  // itself would re-render this container on every streamed token; ids are read on demand in
  // getItemKey instead.
  const count = useAuiState(({ thread }) => thread.messages.length);

  // State, not a ref: the container is not rendered at all while the thread is empty, so the effect
  // below has to re-run when it appears. A ref would still be null from the empty render and, with
  // nothing in its dependencies having changed, would never be measured.
  const [container, setContainer] = useState<HTMLDivElement | null>(null);
  const [scrollMargin, setScrollMargin] = useState(0);

  // The list is not the first thing in the viewport (header padding, welcome block), so the
  // virtualizer needs the gap between the two. Layout effect, not effect: this has to be right in
  // the frame the list first paints, or every item is positioned too high for a frame.
  useLayoutEffect(() => {
    if (!container || !scrollElement) {
      return;
    }
    const measure = (): void => {
      setScrollMargin((previous) => {
        const next = scrollMarginFor(
          container.getBoundingClientRect().top,
          scrollElement.getBoundingClientRect().top,
          scrollElement.scrollTop,
        );
        // Sub-pixel churn here would re-render the list on every scroll frame.
        return Math.abs(next - previous) < 1 ? previous : next;
      });
    };
    measure();
    // Anything above the list changing height moves it: the welcome block unmounting on the first
    // message, the chat-model notice showing, a breakpoint crossing.
    const observer = new ResizeObserver(measure);
    observer.observe(scrollElement);
    observer.observe(container);
    return () => observer.disconnect();
  }, [scrollElement, container]);

  const getItemKey = useCallback(
    (index: number) => messageKeyAt(aui.thread().getState().messages, index),
    [aui],
  );

  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count,
    getScrollElement: () => scrollElement,
    estimateSize: () => THREAD_MESSAGE_ESTIMATE_SIZE_PX,
    overscan: THREAD_MESSAGE_OVERSCAN,
    scrollMargin,
    getItemKey,
    ...THREAD_MESSAGE_ANCHORING,
  });

  // Nothing to window, and no container either: an empty list must not put a 0px positioned box
  // between the welcome block and the spacer, because that is a DOM node the unvirtualized path
  // does not have.
  if (count === 0) {
    return null;
  }

  return (
    <div
      ref={setContainer}
      className="relative w-full shrink-0"
      style={{ height: virtualizer.getTotalSize() }}
    >
      {virtualizer.getVirtualItems().map((item) => (
        <div
          // The message id. See messageKeyAt: an index key remounts every message below a prepend.
          key={item.key}
          data-index={item.index}
          // Measured, not fixed: messages run from about 20px to 2000px, so a fixed height would be
          // wrong for nearly all of them.
          ref={virtualizer.measureElement}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            transform: `translateY(${item.start - virtualizer.options.scrollMargin}px)`,
          }}
        >
          <ThreadPrimitive.MessageByIndex
            index={item.index}
            components={components}
          />
        </div>
      ))}
    </div>
  );
};
