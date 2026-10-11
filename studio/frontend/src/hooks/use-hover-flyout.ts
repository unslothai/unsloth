// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { isBlockedByActiveModal } from "@/components/ui/tooltip-modal-layer";
import { type HoverFlyoutIntent, isPointerHeadingInto } from "@/lib/hover-intent";

type PointerHandler = (event: ReactPointerEvent<HTMLElement>) => void;

export function useHoverFlyout(
  intent: HoverFlyoutIntent,
  contentRef: RefObject<HTMLElement | null>,
) {
  const [open, setOpen] = useState(false);
  const timerRef = useRef<number | undefined>(undefined);
  const stopAimRef = useRef<(() => void) | null>(null);

  const cancel = useCallback(() => {
    window.clearTimeout(timerRef.current);
    timerRef.current = undefined;
    stopAimRef.current?.();
    stopAimRef.current = null;
  }, []);

  const dismiss = useCallback(() => {
    cancel();
    setOpen(false);
  }, [cancel]);

  const closeAfterGrace = useCallback(() => {
    window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(dismiss, intent.closeDelay);
  }, [dismiss, intent.closeDelay]);

  const onTriggerEnter = useCallback<PointerHandler>(
    (event) => {
      if (event.pointerType !== "mouse") return;
      cancel();
      if (open) return;
      const trigger = event.currentTarget;
      // A modal opened inside the delay (a shortcut while the pointer rests here) sends WebKit no
      // pointerleave, so the timer would open the flyout over the dialog.
      timerRef.current = window.setTimeout(() => {
        if (!isBlockedByActiveModal(trigger)) setOpen(true);
      }, intent.openDelay);
    },
    [cancel, intent.openDelay, open],
  );

  const onTriggerLeave = useCallback<PointerHandler>(
    (event) => {
      if (event.pointerType !== "mouse") return;
      cancel();
      if (!open) return;
      const exit = { x: event.clientX, y: event.clientY };
      const followAim = (move: PointerEvent) => {
        const target = contentRef.current?.getBoundingClientRect();
        const pointer = { x: move.clientX, y: move.clientY };
        if (target && isPointerHeadingInto(exit, pointer, target)) {
          closeAfterGrace();
        } else {
          dismiss();
        }
      };
      window.addEventListener("pointermove", followAim);
      stopAimRef.current = () =>
        window.removeEventListener("pointermove", followAim);
      closeAfterGrace();
    },
    [cancel, closeAfterGrace, contentRef, dismiss, open],
  );

  const onContentEnter = useCallback<PointerHandler>(
    (event) => {
      if (event.pointerType === "mouse") cancel();
    },
    [cancel],
  );

  const onContentLeave = useCallback<PointerHandler>(
    (event) => {
      if (event.pointerType !== "mouse") return;
      cancel();
      if (open) closeAfterGrace();
    },
    [cancel, closeAfterGrace, open],
  );

  useEffect(() => cancel, [cancel]);

  const trigger = useMemo(
    () => ({ onPointerEnter: onTriggerEnter, onPointerLeave: onTriggerLeave }),
    [onTriggerEnter, onTriggerLeave],
  );
  const content = useMemo(
    () => ({ onPointerEnter: onContentEnter, onPointerLeave: onContentLeave }),
    [onContentEnter, onContentLeave],
  );

  return { open, dismiss, trigger, content };
}
