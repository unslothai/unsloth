// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ProjectRecord } from "../types";

export type ProjectLandingTab = "chats" | "sources" | "agent";

const PROJECT_LANDING_TABS: readonly ProjectLandingTab[] = [
  "chats",
  "sources",
  "agent",
];

export function nextProjectLandingTab(
  current: ProjectLandingTab,
  key: string,
): ProjectLandingTab {
  if (key === "Home") {
    return "chats";
  }
  if (key === "End") {
    return "agent";
  }
  if (key !== "ArrowLeft" && key !== "ArrowRight") {
    return current;
  }
  const currentIndex = PROJECT_LANDING_TABS.indexOf(current);
  const delta = key === "ArrowRight" ? 1 : -1;
  return (
    PROJECT_LANDING_TABS[
      (currentIndex + delta + PROJECT_LANDING_TABS.length) %
        PROJECT_LANDING_TABS.length
    ] ?? current
  );
}

export function projectFolderUnavailable(
  project: ProjectRecord | undefined,
): boolean {
  return (
    project?.workspaceKind === "folder" && project.workspaceAvailable === false
  );
}
