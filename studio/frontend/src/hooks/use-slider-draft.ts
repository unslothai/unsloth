// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffectEvent, useLayoutEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

export function useSliderDraft(
  value: number,
  onCommit: (value: number) => void,
  onDraft?: (value: number | null) => void,
) {
  const [draft, setDraft] = useState<number | null>(null);
  const pointers = useRef(new Set<number>());
  const dragged = useRef<{ value: number; base: number } | null>(null);
  const owner = useRef(value);
  useLayoutEffect(() => {
    owner.current = value;
  }, [value]);
  const show = (next: number | null) => {
    setDraft(next);
    onDraft?.(next);
  };
  const end = () => {
    const drag = dragged.current;
    pointers.current.clear();
    dragged.current = null;
    show(null);
    if (
      drag !== null &&
      drag.base === owner.current &&
      drag.value !== owner.current
    )
      onCommit(drag.value);
  };
  const release = ({ pointerId }: { pointerId: number }) => {
    if (!pointers.current.delete(pointerId)) return;
    if (pointers.current.size === 0) end();
  };
  const endOnUnmount = useEffectEvent(() => {
    if (pointers.current.size > 0) end();
  });
  const releaseRemoved = useEffectEvent((event: PointerEvent) => {
    if (event.target === document) release(event);
  });
  useLayoutEffect(() => {
    const onLostCapture = (event: PointerEvent) => releaseRemoved(event);
    document.addEventListener("lostpointercapture", onLostCapture);
    return () => {
      document.removeEventListener("lostpointercapture", onLostCapture);
      endOnUnmount();
    };
  }, []);
  return {
    draft,
    sliderProps: {
      value: [draft ?? value],
      onPointerDown: ({ pointerId }: { pointerId: number }) => {
        pointers.current.add(pointerId);
      },
      onValueChange: ([next]: number[]) => {
        if (pointers.current.size === 0) return;
        const drag = { value: next, base: owner.current };
        dragged.current = drag;
        flushSync(() => show(next));
        drag.base = owner.current;
      },
      onValueCommit: ([next]: number[]) => {
        if (pointers.current.size === 0) onCommit(next);
      },
      onLostPointerCapture: release,
    },
  };
}
