// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type ProjectsHomeSettings = {
  projectsHome: string;
  source: "default" | "studio" | "environment";
  editable: boolean;
  isCustom: boolean;
  environmentVariable: string | null;
};

type ApiProjectsHomeSettings = {
  // biome-ignore lint/style/useNamingConvention: API schema
  projects_home: string;
  source: ProjectsHomeSettings["source"];
  editable: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  is_custom: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  environment_variable: string | null;
};

function fromApi(value: ApiProjectsHomeSettings): ProjectsHomeSettings {
  return {
    projectsHome: value.projects_home,
    source: value.source,
    editable: value.editable,
    isCustom: value.is_custom,
    environmentVariable: value.environment_variable,
  };
}

export async function loadProjectsHomeSettings(
  fallbackMessage: string,
): Promise<ProjectsHomeSettings> {
  const response = await authFetch("/api/settings/projects-home");
  if (!response.ok) {
    throw new Error(await readFastApiError(response, fallbackMessage));
  }
  return fromApi(await response.json());
}

export async function updateProjectsHomeSettings(
  projectsHome: string | null,
  fallbackMessage: string,
): Promise<ProjectsHomeSettings> {
  const response = await authFetch("/api/settings/projects-home", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    // biome-ignore lint/style/useNamingConvention: API schema
    body: JSON.stringify({ projects_home: projectsHome }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response, fallbackMessage));
  }
  return fromApi(await response.json());
}
