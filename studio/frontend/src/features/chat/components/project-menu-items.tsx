// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { useChatFavoritesStore } from "@/features/library/chats/favorites-store";
import { useT } from "@/i18n";
import { MessageCircleIcon, StarPointedIcon } from "@/lib/hugeicons-derived";
import { isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Delete02Icon,
  Download01Icon,
  FolderExportIcon,
  LayerIcon,
  PencilEdit02Icon,
  PinIcon,
  PinOffIcon,
  PlusSignIcon,
  Settings02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { type IconSvgElement, HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import { useFileProjectInSection } from "../hooks/use-file-project-in-section";
import { usePinnedProjectsStore } from "../stores/pinned-projects-store";
import { useSidebarOrganizationStore } from "../stores/sidebar-organization-store";
import { listStoredChatThreads } from "../utils/chat-history-storage";
import { pickAndImportChats } from "../utils/import-chats";
import { BulkExportItems, exportThreads } from "./bulk-export-items";
import { OpenProjectFolderItem } from "./open-chat-folder-item";

function Item({
  icon,
  onSelect,
  destructive,
  children,
}: {
  icon: IconSvgElement;
  onSelect: () => void;
  destructive?: boolean;
  children: ReactNode;
}) {
  return (
    <DropdownMenuItem onSelect={onSelect} variant={destructive ? "destructive" : "default"}>
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-icon" />
      <span className="truncate">{children}</span>
    </DropdownMenuItem>
  );
}

/**
 * The Library's project menu, for the project's home and the Projects page. With onView and
 * onNewChat it opens with them and the folder, as in the Library; the home puts the folder
 * under Export.
 */
export function ProjectMenuItems({
  project,
  chatCount,
  onView,
  onNewChat,
  onEdit,
  onDelete,
  onNewSection,
  subClassName,
}: {
  project: { id: string; name: string };
  /** Off Export at 0; unknown (undefined) leaves it on. */
  chatCount?: number;
  onView?: () => void;
  onNewChat?: () => void;
  onEdit: () => void;
  onDelete: () => void;
  /** Opens the page's new section dialog; the page files the project there. */
  onNewSection: () => void;
  subClassName?: string;
}) {
  const t = useT();
  const pinned = usePinnedProjectsStore((s) => s.pinnedIds.includes(project.id));
  const togglePin = usePinnedProjectsStore((s) => s.togglePin);
  const favorite = useChatFavoritesStore((s) => s.projectIds.includes(project.id));
  const setFavorites = useChatFavoritesStore((s) => s.setProjects);
  const sections = useSidebarOrganizationStore((s) => s.customSections);
  const sectionId = useSidebarOrganizationStore((s) => s.sectionByProjectId[project.id] ?? null);
  const fileProjectInSection = useFileProjectInSection();
  const leaving = sections.find((section) => section.id === sectionId);
  const opening = Boolean(onView || onNewChat);

  async function exportProject(format: Parameters<typeof exportThreads>[1], merged: boolean) {
    try {
      const threads = await listStoredChatThreads({ projectId: project.id, includeArchived: false });
      const ids = [...new Set(threads.map((thread) => thread.id))];
      if (ids.length > 0) await exportThreads(ids, format, merged, `project-${project.name}`);
    } catch (error) {
      if (!isDownloadCancelled(error)) toast.error(t("settings.data.exportFailed"));
    }
  }

  return (
    <>
      {opening && (
        <>
          {onView && (
            <Item icon={MessageCircleIcon} onSelect={onView}>
              {t("library.chats.menu.viewChats")}
            </Item>
          )}
          {onNewChat && (
            <Item icon={PencilEdit02Icon} onSelect={onNewChat}>
              {t("library.chats.menu.newChatInProject")}
            </Item>
          )}
          <OpenProjectFolderItem projectId={project.id} />
          <DropdownMenuSeparator />
        </>
      )}
      <Item icon={Settings02Icon} onSelect={onEdit}>
        {t("library.chats.menu.edit")}
      </Item>
      <Item icon={pinned ? PinOffIcon : PinIcon} onSelect={() => togglePin(project.id)}>
        {t(pinned ? "settings.data.library.unpin" : "settings.data.library.pin")}
      </Item>
      <DropdownMenuItem onSelect={() => setFavorites([project.id], !favorite)}>
        <HugeiconsIcon
          icon={StarPointedIcon}
          strokeWidth={1.75}
          className={cn("size-icon", favorite && "[&_path]:fill-current")}
        />
        <span>{t(favorite ? "library.menu.removeFromFavorites" : "library.menu.addToFavorites")}</span>
      </DropdownMenuItem>
      <DropdownMenuSeparator />
      {/* Sections only: projects never nest. */}
      <DropdownMenuSub>
        <DropdownMenuSubTrigger>
          <HugeiconsIcon icon={FolderExportIcon} strokeWidth={1.75} className="size-icon" />
          <span>{t("shell.sections.moveTo")}</span>
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent
          className={cn(
            "max-h-[var(--radix-dropdown-menu-content-available-height)] w-56 overflow-y-auto",
            subClassName,
          )}
        >
          <DropdownMenuLabel className="font-normal text-muted-foreground">
            {t("shell.sections.sectionsHeading")}
          </DropdownMenuLabel>
          <Item icon={PlusSignIcon} onSelect={onNewSection}>
            {t("shell.sections.newSection")}
          </Item>
          {sections
            .filter((section) => section.id !== sectionId)
            .map((section) => (
              <Item
                key={section.id}
                icon={LayerIcon}
                onSelect={() => fileProjectInSection(project, section.id)}
              >
                {section.name}
              </Item>
            ))}
          {sectionId && (
            <Item icon={Cancel01Icon} onSelect={() => fileProjectInSection(project, null)}>
              {leaving
                ? t("shell.sections.removeFrom", { name: leaving.name })
                : t("shell.sections.removeFromSection")}
            </Item>
          )}
        </DropdownMenuSubContent>
      </DropdownMenuSub>
      <DropdownMenuSub>
        <DropdownMenuSubTrigger disabled={chatCount === 0}>
          <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-icon" />
          <span>{t("common.export")}</span>
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent className={cn("w-56", subClassName)}>
          <BulkExportItems onExport={(format, merged) => void exportProject(format, merged)} />
        </DropdownMenuSubContent>
      </DropdownMenuSub>
      <Item
        icon={Upload01Icon}
        onSelect={() => void pickAndImportChats({ projectId: project.id, name: project.name })}
      >
        {t("settings.chat.importChats")}
      </Item>
      {!opening && <OpenProjectFolderItem projectId={project.id} />}
      <DropdownMenuSeparator />
      <Item icon={Delete02Icon} destructive onSelect={onDelete}>
        {t("library.chats.menu.deleteProject")}
      </Item>
    </>
  );
}
