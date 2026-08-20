// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AudioWave01Icon,
  BotIcon,
  ChefHatIcon,
  CloudIcon,
  DashboardCircleIcon,
  DownloadSquare01Icon,
  DragDropVerticalIcon,
  FlimSlateIcon,
  Folder01Icon,
  Globe02Icon,
  Image03Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  Search01Icon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Reorder, useDragControls } from "motion/react";
import { Switch } from "@/components/ui/switch";
import {
  FEATURE_AGENTS_NAV,
  FEATURE_API_MONITOR,
  FEATURE_AUDIO,
  FEATURE_EXPORT,
  FEATURE_FILES_NAV,
  FEATURE_IMAGES,
  FEATURE_MANAGEMENT_NAV,
  FEATURE_MEMORY_NAV,
  FEATURE_PROJECTS,
  FEATURE_RECIPES,
  FEATURE_SEARCH_NAV,
  FEATURE_TRAIN,
  FEATURE_VIDEO,
} from "@/config/disabled-features";
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { useT } from "@/i18n";
import type { TranslationKey } from "@/i18n";
import type { IconSvgElement } from "@hugeicons/react";
import type { SidebarNavItemPref } from "../stores/appearance-custom-store";
import { useAppearanceCustomStore } from "../stores/appearance-custom-store";

const ITEM_META: Record<
  SidebarNavItemPref["id"],
  { icon: IconSvgElement; labelKey: TranslationKey }
> = {
  projects: { icon: Folder01Icon, labelKey: "shell.navigation.projects" },
  hub: { icon: DashboardCircleIcon, labelKey: "shell.navigation.hub" },
  agents: { icon: BotIcon, labelKey: "shell.navigation.agents" },
  files: { icon: Folder01Icon, labelKey: "shell.navigation.files" },
  memory: { icon: CloudIcon, labelKey: "shell.navigation.memory" },
  search: { icon: Search01Icon, labelKey: "shell.navigation.searchApps" },
  management: {
    icon: Settings02Icon,
    labelKey: "shell.navigation.management",
  },
  images: { icon: Image03Icon, labelKey: "shell.navigation.images" },
  train: { icon: TestTubeOutlineIcon, labelKey: "shell.navigation.train" },
  video: { icon: FlimSlateIcon, labelKey: "shell.navigation.video" },
  audio: { icon: AudioWave01Icon, labelKey: "shell.navigation.audio" },
  recipes: { icon: ChefHatIcon, labelKey: "shell.navigation.recipes" },
  export: { icon: DownloadSquare01Icon, labelKey: "shell.navigation.export" },
  api: { icon: Globe02Icon, labelKey: "shell.navigation.api" },
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

function MovableRow({ item }: { item: SidebarNavItemPref }) {
  const t = useT();
  const controls = useDragControls();
  const patch = useAppearanceCustomStore((s) => s.patch);
  const sidebarNav = useAppearanceCustomStore((s) => s.customization.sidebarNav);
  const meta = ITEM_META[item.id];
  return (
    <Reorder.Item
      value={item.id}
      dragListener={false}
      dragControls={controls}
      layout="position"
      // The dragged row lifts above its siblings so it stays readable.
      whileDrag={{
        backgroundColor: "var(--popover)",
        boxShadow: "0 4px 16px rgb(0 0 0 / 0.18)",
        zIndex: 10,
      }}
      className="relative flex h-9 items-center gap-2.5 rounded-lg px-2"
    >
      <button
        type="button"
        aria-label={t("settings.appearance.sidebarNav.dragToReorder")}
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
        aria-label={t("settings.appearance.sidebarNav.pinToSidebar", {
          name: t(meta.labelKey),
        })}
        checked={item.pinned}
        onCheckedChange={(pinned) =>
          patch({
            sidebarNav: sidebarNav.map((entry) =>
              entry.id === item.id ? { ...entry, pinned } : entry,
            ),
          })
        }
      />
    </Reorder.Item>
  );
}

/** Pin and reorder the sidebar nav rows. Unpinned rows collect in the "More" flyout; a single unpinned row is hidden instead of getting a menu of one. New chat is static: an action, not a destination. */
export function SidebarNavCustomizer() {
  const t = useT();
  const sidebarNav = useAppearanceCustomStore((s) => s.customization.sidebarNav);
  const patch = useAppearanceCustomStore((s) => s.patch);
  const enabledNav = sidebarNav.filter(
    (item) =>
      (item.id !== "projects" || FEATURE_PROJECTS) &&
      (item.id !== "agents" || FEATURE_AGENTS_NAV) &&
      (item.id !== "files" || FEATURE_FILES_NAV) &&
      (item.id !== "memory" || FEATURE_MEMORY_NAV) &&
      (item.id !== "search" || FEATURE_SEARCH_NAV) &&
      (item.id !== "management" || FEATURE_MANAGEMENT_NAV) &&
      (item.id !== "images" || FEATURE_IMAGES) &&
      (item.id !== "train" || FEATURE_TRAIN) &&
      (item.id !== "video" || FEATURE_VIDEO) &&
      (item.id !== "audio" || FEATURE_AUDIO) &&
      (item.id !== "recipes" || FEATURE_RECIPES) &&
      (item.id !== "export" || FEATURE_EXPORT) &&
      (item.id !== "api" || FEATURE_API_MONITOR),
  );
  const enabledIds = new Set(enabledNav.map((item) => item.id));
  const unpinnedCount = enabledNav.filter((item) => !item.pinned).length;
  return (
    <div className="flex flex-col rounded-xl border border-border/70 p-1.5">
      <FixedRow icon={PencilEdit02Icon} label={t("shell.navigation.newChat")} />
      <Reorder.Group
        axis="y"
        values={enabledNav.map((item) => item.id)}
        onReorder={(ids: SidebarNavItemPref["id"][]) =>
          patch({
            sidebarNav: [
              ...ids.flatMap(
                (id) => sidebarNav.find((entry) => entry.id === id) ?? [],
              ),
              ...sidebarNav.filter((entry) => !enabledIds.has(entry.id)),
            ],
          })
        }
        className="flex flex-col"
      >
        {enabledNav.map((item) => (
          <MovableRow key={item.id} item={item} />
        ))}
      </Reorder.Group>
      {/* Mirrors the sidebar: More only exists at two or more. */}
      {unpinnedCount > 1 && (
        <>
          <div className="mx-2 my-1 border-t border-border/70" />
          <FixedRow
            icon={MoreHorizontalIcon}
            label={t("settings.appearance.sidebarNav.moreHolds", {
              count: String(unpinnedCount),
            })}
          />
        </>
      )}
    </div>
  );
}
