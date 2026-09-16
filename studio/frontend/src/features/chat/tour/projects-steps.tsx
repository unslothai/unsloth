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
          A project keeps related chats together and gives them shared files and
          a shared system prompt. Anything you add as a source is available to
          every chat inside it.
        </>
      ),
    },
    {
      id: "search",
      target: "projects-search",
      title: "Find one fast",
      body: (
        <>
          Searches every project by name, not just the ones on screen. Sort by
          activity to keep what you touched last on top.
        </>
      ),
    },
    {
      id: "io",
      target: "projects-io",
      title: "Import and export",
      body: (
        <>
          Export chats as JSONL, CSV or ShareGPT, which is a quick way to turn
          real conversations into training data. Import brings exported chats
          back into a project.
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
          Click a row to open it in Chat. The menu on each row pins it to the
          sidebar, renames it, or deletes it with its chats.
        </>
      ),
    });
  }

  return steps;
}
