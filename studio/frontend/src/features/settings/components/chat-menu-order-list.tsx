// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { type PlusMenuItemId, usePlusMenuPrefsStore } from "@/features/chat";
import {
  flipShifts,
  insertionIndex,
  ownsDrag,
} from "@/features/chat/prompt-storage/reorder";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { GripVerticalIcon } from "lucide-react";
import {
  type ReactNode,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { SettingsRow } from "./settings-row";

function move<T>(items: readonly T[], from: number, to: number): T[] {
  const next = [...items];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item);
  return next;
}

export function ChatMenuOrderList({
  items,
}: {
  items: Record<PlusMenuItemId, { label: string; icon: ReactNode }>;
}) {
  const t = useT();
  const order = usePlusMenuPrefsStore((s) => s.order);
  const pins = usePlusMenuPrefsStore((s) => s.pins);
  const setOrder = usePlusMenuPrefsStore((s) => s.setOrder);
  const togglePin = usePlusMenuPrefsStore((s) => s.togglePin);
  const [dragging, setDragging] = useState<PlusMenuItemId | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const rowRefs = useRef(new Map<PlusMenuItemId, HTMLDivElement>());
  const pointerIdRef = useRef<number | null>(null);
  const pointerYRef = useRef(0);
  const orderRef = useRef(order);
  const firstOffsets = useRef(new Map<string, number>());
  const reorderTick = useRef(0);
  const animatedTick = useRef(0);

  useLayoutEffect(() => {
    orderRef.current = order;
  });

  const measureOffsets = useCallback(() => {
    const offsets = new Map<string, number>();
    rowRefs.current.forEach((el, id) => offsets.set(id, el.offsetTop));
    return offsets;
  }, []);

  // FLIP, as in the saved-prompt list: each moved row starts where it was and slides into place.
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs after each reorder commits.
  useLayoutEffect(() => {
    if (reorderTick.current === animatedTick.current) return;
    animatedTick.current = reorderTick.current;
    flipShifts(firstOffsets.current, measureOffsets()).forEach((dy, id) => {
      const el = rowRefs.current.get(id as PlusMenuItemId);
      if (!el) return;
      el.style.transition = "none";
      el.style.transform = `translateY(${dy}px)`;
      requestAnimationFrame(() => {
        el.style.transition = "transform 180ms cubic-bezier(0.2, 0, 0, 1)";
        el.style.transform = "";
      });
    });
  }, [order, measureOffsets]);

  const applyOrder = useCallback(
    (from: number, to: number) => {
      if (from === to) return;
      firstOffsets.current = measureOffsets();
      reorderTick.current += 1;
      const next = move(orderRef.current, from, to);
      // Ahead of the commit, so the next pointer move hit-tests the new order.
      orderRef.current = next;
      setOrder(next);
    },
    [measureOffsets, setOrder],
  );

  const onGripPointerDown = (
    id: PlusMenuItemId,
    event: ReactPointerEvent<HTMLButtonElement>,
  ) => {
    if (event.button !== 0 || !event.isPrimary || pointerIdRef.current !== null)
      return;
    // No pointer capture: the reorder moves the row's node, which would release it mid-drag.
    event.preventDefault();
    pointerIdRef.current = event.pointerId;
    pointerYRef.current = event.clientY;
    setDragging(id);
  };

  useEffect(() => {
    if (!dragging) return;
    const container = containerRef.current;
    if (!container) return;
    const evaluate = () => {
      const current = orderRef.current;
      const from = current.indexOf(dragging);
      if (from < 0) return;
      // Layout offsets ignore the FLIP transforms still running on neighbouring rows.
      const localY =
        pointerYRef.current - container.getBoundingClientRect().top;
      const boxes = current.map((id) => {
        const el = rowRefs.current.get(id);
        return el ? { top: el.offsetTop, height: el.offsetHeight } : undefined;
      });
      applyOrder(from, insertionIndex(boxes, from, localY));
    };
    const end = () => {
      pointerIdRef.current = null;
      setDragging(null);
    };
    const onMove = (event: PointerEvent) => {
      if (!ownsDrag(pointerIdRef.current, event.pointerId)) return;
      // A release outside the window sends no pointerup.
      if (event.buttons === 0) {
        end();
        return;
      }
      pointerYRef.current = event.clientY;
      evaluate();
    };
    const onUp = (event: PointerEvent) => {
      if (ownsDrag(pointerIdRef.current, event.pointerId)) end();
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    window.addEventListener("blur", end);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      window.removeEventListener("blur", end);
    };
  }, [dragging, applyOrder]);

  return (
    // `relative`: the offsetParent the hit-test measures against.
    <div
      ref={containerRef}
      data-testid="chat-menu-order"
      className={cn("relative flex flex-col", dragging && "select-none")}
    >
      {order.map((id, index) => {
        const item = items[id];
        return (
          <div
            key={id}
            data-menu-item={id}
            ref={(el) => {
              if (el) rowRefs.current.set(id, el);
              else rowRefs.current.delete(id);
            }}
            className={cn(
              "relative rounded-[12px] bg-transparent",
              dragging === id && "z-10 bg-muted shadow-sm",
            )}
          >
            <SettingsRow
              label={item.label}
              icon={
                <span className="flex items-center gap-1.5">
                  <button
                    type="button"
                    aria-label={t("settings.chat.menu.reorder", {
                      item: item.label,
                    })}
                    onPointerDown={(event) => onGripPointerDown(id, event)}
                    onKeyDown={(event) => {
                      const delta =
                        event.key === "ArrowUp"
                          ? -1
                          : event.key === "ArrowDown"
                            ? 1
                            : 0;
                      if (delta === 0) return;
                      event.preventDefault();
                      const to = index + delta;
                      if (to >= 0 && to < order.length) applyOrder(index, to);
                    }}
                    className={cn(
                      "-ml-1 flex touch-none items-center rounded text-muted-foreground outline-none hover:text-foreground focus-visible:ring-1 focus-visible:ring-ring",
                      dragging === id ? "cursor-grabbing" : "cursor-grab",
                    )}
                  >
                    <GripVerticalIcon className="size-4" />
                  </button>
                  {item.icon}
                </span>
              }
            >
              <Switch
                aria-label={item.label}
                checked={pins[id]}
                onCheckedChange={() => togglePin(id)}
              />
            </SettingsRow>
          </div>
        );
      })}
    </div>
  );
}
