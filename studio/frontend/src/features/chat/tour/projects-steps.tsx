// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

export function buildProjectsTourSteps({
  hasProjects,
}: {
  hasProjects: boolean;
}): TourStep[] {
  const steps: TourStep[] = [
    {
      id: "new",
      target: "projects-new",
      title: "Make a project",
      body: (
        <>
          A project keeps related chats together with shared files and a shared
          system prompt. Sources you add reach every chat inside it.
        </>
      ),
    },
    {
      id: "io",
      target: "projects-io",
      title: "Import and export",
      body: (
        <>
          Export chats as JSONL, CSV or ShareGPT, a quick way to turn real
          conversations into training data. Import brings them back.
        </>
      ),
    },
  ];

  if (hasProjects) {
    steps.push({
      id: "list",
      target: "projects-list",
      title: "Open a project",
      body: (
        <>
          Click a row to open it in Chat. The row menu pins it to the sidebar,
          renames it, or deletes it with its chats.
        </>
      ),
    });
  }

  return steps;
}
