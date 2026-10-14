// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { type PlusMenuItemId, usePlusMenuPrefsStore } from "@/features/chat";
import { useT } from "@/i18n";
import {
  AttachmentIcon,
  CodeIcon,
  DragDropVerticalIcon,
  InternetIcon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { Reorder, useDragControls } from "motion/react";
import { type ReactNode, type RefObject, useRef, useState } from "react";

type ItemMeta = { label: string; icon: ReactNode };

// Same row as the Profile menu customizer (sidebar-menu-customizer.tsx).
function FixedRow({ icon, label }: { icon: IconSvgElement; label: string }) {
  return (
    <div className="flex h-9 items-center gap-2.5 rounded-lg px-2 text-muted-foreground/70">
      <span className="size-4" aria-hidden="true" />
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />
      <span className="text-ui-13">{label}</span>
    </div>
  );
}

function MovableRow({
  id,
  meta,
  index,
  order,
  bounds,
  onDrop,
}: {
  id: PlusMenuItemId;
  meta: ItemMeta;
  index: number;
  order: PlusMenuItemId[];
  bounds: RefObject<HTMLUListElement | null>;
  onDrop: () => void;
}) {
  const t = useT();
  const controls = useDragControls();
  const pinned = usePlusMenuPrefsStore((s) => s.pins[id]);
  const togglePin = usePlusMenuPrefsStore((s) => s.togglePin);
  const setOrder = usePlusMenuPrefsStore((s) => s.setOrder);
  return (
    <Reorder.Item
      value={id}
      data-menu-item={id}
      dragListener={false}
      dragControls={controls}
      layout="position"
      // The row stays inside the list instead of following the pointer over the rest of the tab.
      dragConstraints={bounds}
      dragElastic={0}
      onDragEnd={onDrop}
      whileDrag={{
        backgroundColor: "var(--popover)",
        boxShadow: "0 4px 16px rgb(0 0 0 / 0.18)",
        zIndex: 10,
      }}
      className="relative flex h-9 items-center gap-2.5 rounded-lg px-2"
    >
      <button
        type="button"
        aria-label={t("settings.appearance.sidebarMenu.dragToReorder")}
        onPointerDown={(e) => {
          e.preventDefault();
          controls.start(e);
        }}
        // Arrow keys move the row too, since a pointer drag has no keyboard path.
        onKeyDown={(e) => {
          const to =
            index + (e.key === "ArrowUp" ? -1 : e.key === "ArrowDown" ? 1 : 0);
          if (to === index || to < 0 || to >= order.length) {
            return;
          }
          e.preventDefault();
          const next = [...order];
          next.splice(to, 0, ...next.splice(index, 1));
          setOrder(next);
        }}
        className="flex size-4 shrink-0 cursor-grab touch-none items-center justify-center text-muted-foreground active:cursor-grabbing"
      >
        <HugeiconsIcon
          icon={DragDropVerticalIcon}
          strokeWidth={1.75}
          className="size-4"
        />
      </button>
      <span className="flex size-4 items-center justify-center text-foreground/80 [&_svg]:size-4">
        {meta.icon}
      </span>
      <span className="text-ui-13 text-foreground">{meta.label}</span>
      <Switch
        className="ml-auto"
        aria-label={meta.label}
        checked={pinned}
        onCheckedChange={() => togglePin(id)}
      />
    </Reorder.Item>
  );
}

/** The "+" menu's movable items between its fixed ones, laid out as the menu shows them. */
export function ChatMenuOrderList({
  items,
}: { items: Record<PlusMenuItemId, ItemMeta> }) {
  const t = useT();
  const order = usePlusMenuPrefsStore((s) => s.order);
  const setOrder = usePlusMenuPrefsStore((s) => s.setOrder);
  const listRef = useRef<HTMLUListElement>(null);
  // Swaps stay local while dragging; the store (persisted, read by both + menus) takes the drop.
  const [draft, setDraftState] = useState<PlusMenuItemId[] | null>(null);
  // Read on drop: the last swap and the release can land before a re-render.
  const draftRef = useRef<PlusMenuItemId[] | null>(null);
  const setDraft = (ids: PlusMenuItemId[] | null) => {
    draftRef.current = ids;
    setDraftState(ids);
  };
  const shown = draft ?? order;
  const drop = () => {
    if (draftRef.current) {
      setOrder(draftRef.current);
    }
    setDraft(null);
  };
  return (
    <div
      data-testid="chat-menu-order"
      className="flex flex-col rounded-xl border border-border/70 p-1.5"
    >
      <FixedRow
        icon={AttachmentIcon}
        label={t("settings.chat.menu.addFiles")}
      />
      <FixedRow icon={InternetIcon} label={t("settings.chat.menu.webSearch")} />
      <FixedRow icon={CodeIcon} label={t("settings.chat.menu.tools")} />
      <div className="mx-2 my-1 border-t border-border/70" />
      <Reorder.Group
        ref={listRef}
        axis="y"
        values={shown}
        onReorder={(ids: PlusMenuItemId[]) => setDraft(ids)}
        className="flex flex-col"
      >
        {shown.map((id, index) => (
          <MovableRow
            key={id}
            id={id}
            meta={items[id]}
            index={index}
            order={shown}
            bounds={listRef}
            onDrop={drop}
          />
        ))}
      </Reorder.Group>
      <FixedRow
        icon={MoreHorizontalIcon}
        label={t("settings.chat.menu.more")}
      />
    </div>
  );
}
