// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CloudIcon,
  CpuIcon,
  CursorInfo02Icon,
  DragDropVerticalIcon,
  Globe02Icon,
  HelpCircleIcon,
  Logout05Icon,
  Message01Icon,
  Moon02Icon,
  PaintBrush02Icon,
  PowerIcon,
  Settings02Icon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Reorder, useDragControls } from "motion/react";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import type { IconSvgElement } from "@hugeicons/react";
import type { SidebarMenuItemPref } from "../stores/appearance-custom-store";
import { useAppearanceCustomStore } from "../stores/appearance-custom-store";

const ITEM_META: Record<
  SidebarMenuItemPref["id"],
  { icon: IconSvgElement; labelKey: TranslationKey }
> = {
  api: { icon: Globe02Icon, labelKey: "shell.navigation.api" },
  darkMode: { icon: Moon02Icon, labelKey: "settings.appearance.sidebarMenu.darkModeToggle" },
  guidedTour: { icon: CursorInfo02Icon, labelKey: "shell.navigation.guidedTour" },
  profile: { icon: UserIcon, labelKey: "settings.tabs.profile" },
  appearance: { icon: PaintBrush02Icon, labelKey: "settings.tabs.appearance" },
  resources: { icon: CpuIcon, labelKey: "settings.tabs.resources" },
  chat: { icon: Message01Icon, labelKey: "settings.tabs.chat" },
  connections: { icon: CloudIcon, labelKey: "settings.tabs.connections" },
};

function FixedRow({ icon, label }: { icon: IconSvgElement; label: string }) {
  return (
    <div className="flex h-9 items-center gap-2.5 rounded-lg px-2 text-muted-foreground/70">
      {/* Spacer where the drag handle sits on movable rows. */}
      <span className="size-4" aria-hidden="true" />
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-4" />
      <span className="text-ui-13">{label}</span>
    </div>
  );
}

function MovableRow({ item }: { item: SidebarMenuItemPref }) {
  const t = useT();
  const controls = useDragControls();
  const patch = useAppearanceCustomStore((s) => s.patch);
  const sidebarMenu = useAppearanceCustomStore(
    (s) => s.customization.sidebarMenu,
  );
  const meta = ITEM_META[item.id];
  return (
    <Reorder.Item
      value={item.id}
      dragListener={false}
      dragControls={controls}
      layout="position"
      // Rows sit flat on the dialog surface; the dragged row lifts above its
      // siblings so it stays readable while crossing them.
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
        className="flex size-4 shrink-0 cursor-grab touch-none items-center justify-center text-muted-foreground active:cursor-grabbing"
      >
        <HugeiconsIcon
          icon={DragDropVerticalIcon}
          strokeWidth={1.75}
          className="size-4"
        />
      </button>
      <HugeiconsIcon
        icon={meta.icon}
        strokeWidth={1.75}
        className="size-4 text-foreground/80"
      />
      <span className="text-ui-13 text-foreground">{t(meta.labelKey)}</span>
      <Switch
        className="ml-auto"
        checked={item.visible}
        onCheckedChange={(visible) =>
          patch({
            sidebarMenu: sidebarMenu.map((entry) =>
              entry.id === item.id ? { ...entry, visible } : entry,
            ),
          })
        }
      />
    </Reorder.Item>
  );
}

/**
 * Show/hide and reorder the optional sidebar profile menu items. The pinned
 * entries (Settings on top; Help, Log out, Shutdown below) are rendered as
 * static rows so the final menu layout is obvious.
 */
export function SidebarMenuCustomizer() {
  const t = useT();
  const sidebarMenu = useAppearanceCustomStore(
    (s) => s.customization.sidebarMenu,
  );
  const patch = useAppearanceCustomStore((s) => s.patch);
  return (
    <div className="flex flex-col rounded-xl border border-border/70 p-1.5">
      <FixedRow icon={Settings02Icon} label={t("shell.navigation.settings")} />
      <Reorder.Group
        axis="y"
        values={sidebarMenu.map((item) => item.id)}
        onReorder={(ids: SidebarMenuItemPref["id"][]) =>
          patch({
            sidebarMenu: ids.flatMap(
              (id) => sidebarMenu.find((entry) => entry.id === id) ?? [],
            ),
          })
        }
        className="flex flex-col"
      >
        {sidebarMenu.map((item) => (
          <MovableRow key={item.id} item={item} />
        ))}
      </Reorder.Group>
      <div className="mx-2 my-1 border-t border-border/70" />
      <FixedRow icon={HelpCircleIcon} label={t("common.help")} />
      <FixedRow icon={Logout05Icon} label={t("shell.navigation.logOut")} />
      <FixedRow icon={PowerIcon} label={t("common.shutdown")} />
    </div>
  );
}
