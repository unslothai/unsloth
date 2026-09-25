// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Folder01Icon, FolderAddIcon, FolderExportIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useState } from "react";

import {
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { NewProjectDialog, useChatProjects } from "@/features/chat";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";

export type MediaNoun = "image" | "video" | "clip" | "file";

const ADD_FAILED: Record<MediaNoun, TranslationKey> = {
  image: "library.project.addFailedImage",
  video: "library.project.addFailedVideo",
  clip: "library.project.addFailedClip",
  file: "library.project.addFailedFile",
};

const NEW_PROJECT_TITLE: Record<MediaNoun, TranslationKey> = {
  image: "library.project.newProjectImage",
  video: "library.project.newProjectVideo",
  clip: "library.project.newProjectClip",
  file: "library.project.newProjectFile",
};

export function useProjectSubmenu({
  noun,
  onAddToProject,
}: {
  noun: MediaNoun;
  onAddToProject?: (projectId: string) => Promise<{ already: boolean }>;
}) {
  const t = useT();
  const [creatingProject, setCreatingProject] = useState(false);
  const { projects } = useChatProjects();

  async function addToProject(projectId: string, projectName: string) {
    if (!onAddToProject) return;
    try {
      const { already } = await onAddToProject(projectId);
      toast.success(
        t(already ? "library.project.alreadyIn" : "library.project.addedTo", {
          project: projectName,
        }),
      );
    } catch (err) {
      toast.error(t(ADD_FAILED[noun]), {
        description: err instanceof Error ? err.message : undefined,
      });
    }
  }

  const submenu = onAddToProject ? (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>
        <HugeiconsIcon icon={FolderExportIcon} strokeWidth={1.75} className="size-icon" />
        <span>{t("library.project.label")}</span>
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent
        sideOffset={0}
        alignOffset={-4}
        className="unsloth-plus-menu sidebar-row-menu w-48"
      >
        {/* Actions above the rule, destinations below, as in a chat's Project menu. */}
        <DropdownMenuItem onClick={() => setCreatingProject(true)}>
          <HugeiconsIcon icon={FolderAddIcon} strokeWidth={1.75} className="size-icon" />
          <span>{t("library.project.newProject")}</span>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        {projects.length === 0 ? (
          <DropdownMenuItem disabled={true}>{t("library.project.noProjects")}</DropdownMenuItem>
        ) : (
          projects.map((project) => (
            <DropdownMenuItem
              key={project.id}
              onClick={() => void addToProject(project.id, project.name)}
            >
              <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
              <span className="truncate">{project.name}</span>
            </DropdownMenuItem>
          ))
        )}
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  ) : null;

  const dialog = creatingProject ? (
    <NewProjectDialog
      open={true}
      onOpenChange={setCreatingProject}
      title={t(NEW_PROJECT_TITLE[noun])}
      submitLabel={t("library.project.createAndAdd")}
      onCreated={(project) => addToProject(project.id, project.name)}
    />
  ) : null;

  const close = useCallback(() => setCreatingProject(false), []);
  return { submenu, dialog, close };
}
