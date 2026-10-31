// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback } from "react";
import { usePinnedProjectsStore } from "../stores/pinned-projects-store";
import { useSidebarOrganizationStore } from "../stores/sidebar-organization-store";

/** Files a project in a section (null unfiles). Filing unpins it. */
export function useFileProjectInSection(): (
  project: { id: string; name: string },
  sectionId: string | null,
  sectionName?: string,
) => void {
  const t = useT();
  return useCallback(
    (project, sectionId, sectionName) => {
      const organization = useSidebarOrganizationStore.getState();
      const leaving = organization.sectionByProjectId[project.id];
      const nameOf = (id: string | undefined) =>
        organization.customSections.find((section) => section.id === id)?.name ?? "";
      organization.setProjectsSection([project.id], sectionId);
      if (sectionId) {
        const pins = usePinnedProjectsStore.getState();
        if (pins.pinnedIds.includes(project.id)) pins.unpin(project.id);
        organization.setSectionHidden(sectionId, false);
      }
      toast.success(
        sectionId
          ? t("library.chats.toast.projectMoved", {
              project: project.name,
              section: sectionName ?? nameOf(sectionId),
            })
          : t("library.chats.toast.projectUnfiled", {
              project: project.name,
              section: nameOf(leaving),
            }),
      );
    },
    [t],
  );
}
