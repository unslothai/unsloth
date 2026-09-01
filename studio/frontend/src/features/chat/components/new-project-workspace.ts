// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ProjectWorkspaceMode = "managed" | "folder";

export interface ProjectWorkspaceCreation<TProject> {
  project: TProject;
  sourceUploadError: unknown | null;
}

export function projectFolderPickerDisabled(args: {
  nativePathLeasesSupported: boolean;
  busy: boolean;
  pickingFolder: boolean;
}): boolean {
  return !args.nativePathLeasesSupported || args.busy || args.pickingFolder;
}

export function projectCreateDisabled(args: {
  name: string;
  busy: boolean;
  pickingFolder: boolean;
  stagingSources: boolean;
  workspaceMode: ProjectWorkspaceMode;
  folderToken?: string | null;
}): boolean {
  return (
    !args.name.trim() ||
    args.busy ||
    args.pickingFolder ||
    args.stagingSources ||
    (args.workspaceMode === "folder" && !args.folderToken)
  );
}

/**
 * Native pickers settle asynchronously and the dialog stays mounted between
 * opens. This guard prevents a selection from an earlier dialog session from
 * restoring a folder after the user has cancelled or started over.
 */
export function createProjectFolderPickerGuard(): {
  begin: () => number;
  invalidate: () => void;
  isCurrent: (claim: number) => boolean;
} {
  let generation = 0;
  return {
    begin() {
      generation += 1;
      return generation;
    },
    invalidate() {
      generation += 1;
    },
    isCurrent(claim) {
      return claim === generation;
    },
  };
}

export async function createProjectWorkspace<
  TProject extends { id: string },
  TSource,
>(args: {
  name: string;
  workspaceMode: ProjectWorkspaceMode;
  folderToken?: string | null;
  sources: TSource[];
  createManaged: (name: string) => Promise<TProject>;
  openFolder: (nativePathToken: string, name: string) => Promise<TProject>;
  uploadSources: (projectId: string, sources: TSource[]) => Promise<void>;
}): Promise<ProjectWorkspaceCreation<TProject>> {
  const name = args.name.trim();
  if (!name) {
    throw new Error("Project name is required.");
  }
  let project: TProject;
  if (args.workspaceMode === "folder") {
    const folderToken = args.folderToken;
    if (!folderToken) {
      throw new Error("Choose a project folder before creating the project.");
    }
    project = await args.openFolder(folderToken, name);
  } else {
    project = await args.createManaged(name);
  }

  // Workspace files stay live in place. Sources remain a separate, explicit
  // RAG layer and are uploaded only after the project has a durable id. Source
  // attachment is not allowed to turn a successful project creation into an
  // apparent total failure that strands the newly created workspace.
  let sourceUploadError: unknown | null = null;
  if (args.sources.length > 0) {
    try {
      await args.uploadSources(project.id, args.sources);
    } catch (error) {
      sourceUploadError = error;
    }
  }
  return { project, sourceUploadError };
}
