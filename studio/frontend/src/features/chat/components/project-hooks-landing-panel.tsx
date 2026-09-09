// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatProjects } from "../hooks/use-chat-projects";
import { ProjectHooksPanel } from "./project-hooks-panel";

export function ProjectHooksLandingPanel({ projectId }: { projectId: string }) {
  const { projects, hasLoaded } = useChatProjects();
  const project = projects.find((candidate) => candidate.id === projectId);
  if (!project) {
    return <p className="mt-8 text-sm text-muted-foreground">{hasLoaded ? "Project unavailable." : "Loading project…"}</p>;
  }
  return (
    <div className="mt-8 flex flex-col gap-6">
      <ProjectHooksPanel key={`hooks:${project.id}`} project={project} />
    </div>
  );
}
