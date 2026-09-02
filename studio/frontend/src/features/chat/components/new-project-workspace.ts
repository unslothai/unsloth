// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ProjectWorkspaceMode = "managed" | "folder";

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
}): Promise<{ project: TProject; sourceUploadError: unknown | null }> {
  const name = args.name.trim();
  if (!name) throw new Error("Project name is required.");

  let project: TProject;
  if (args.workspaceMode === "folder") {
    if (!args.folderToken) {
      throw new Error("Choose a project folder before creating it.");
    }
    project = await args.openFolder(args.folderToken, name);
  } else {
    project = await args.createManaged(name);
  }

  // A working folder is not a RAG source. Only files staged in the Sources
  // control are uploaded, after the durable project exists.
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
