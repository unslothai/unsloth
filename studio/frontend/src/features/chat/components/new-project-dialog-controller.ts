// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ProjectRecord } from "../types";
import {
  type ProjectWorkspaceMode,
  createProjectFolderPickerGuard,
  createProjectWorkspace,
  projectCreateDisabled,
  projectFolderPickerDisabled,
} from "./new-project-workspace.ts";

export interface ProjectFolderSelection {
  token: string;
  displayName: string;
  displayPath: string;
}

export interface NewProjectDialogState<TSource> {
  name: string;
  sources: TSource[];
  workspaceMode: ProjectWorkspaceMode;
  folder: ProjectFolderSelection | null;
  pickingFolder: boolean;
  busy: boolean;
  stagingSources: boolean;
}

export interface NewProjectDialogDependencies<TSource> {
  nativePathLeasesSupported: boolean;
  pickFolder: () => Promise<ProjectFolderSelection | null>;
  createManaged: (name: string) => Promise<ProjectRecord>;
  openFolder: (nativePathToken: string, name: string) => Promise<ProjectRecord>;
  uploadSources: (projectId: string, sources: TSource[]) => Promise<void>;
  currentRoute: () => string;
  onOpenChange: (open: boolean) => void;
  onCreated?: (
    project: ProjectRecord,
    context: { stayedOnRoute: boolean },
  ) => void | Promise<void>;
  activateProject: (projectId: string) => void;
  navigateToProject: (projectId: string) => void;
  showError: (title: string, description?: string) => void;
}

export interface NewProjectDialogController<TSource> {
  getState: () => NewProjectDialogState<TSource>;
  subscribe: (listener: () => void) => () => void;
  updateDependencies: (
    dependencies: NewProjectDialogDependencies<TSource>,
  ) => void;
  mount: () => void;
  unmount: () => void;
  reset: () => void;
  setName: (name: string) => void;
  setSources: (sources: TSource[]) => void;
  setStagingSources: (pending: boolean) => void;
  openChanged: (open: boolean) => void;
  clickCancel: () => void;
  clickManagedWorkspace: () => void;
  clickExistingFolder: () => Promise<void>;
  submit: () => Promise<void>;
  folderPickerDisabled: () => boolean;
  createDisabled: () => boolean;
}

function initialState<TSource>(): NewProjectDialogState<TSource> {
  return {
    name: "",
    sources: [],
    workspaceMode: "managed",
    folder: null,
    pickingFolder: false,
    busy: false,
    stagingSources: false,
  };
}

export function createNewProjectDialogController<TSource>(
  initialDependencies: NewProjectDialogDependencies<TSource>,
): NewProjectDialogController<TSource> {
  let dependencies = initialDependencies;
  let state = initialState<TSource>();
  let mounted = true;
  const listeners = new Set<() => void>();
  const pickerGuard = createProjectFolderPickerGuard();

  function publish(next: NewProjectDialogState<TSource>): void {
    state = next;
    if (!mounted) return;
    for (const listener of listeners) listener();
  }

  function patch(next: Partial<NewProjectDialogState<TSource>>): void {
    publish({ ...state, ...next });
  }

  function reset(): void {
    pickerGuard.invalidate();
    publish({ ...initialState<TSource>(), busy: state.busy });
  }

  function close(): void {
    if (state.busy) return;
    reset();
    dependencies.onOpenChange(false);
  }

  async function clickExistingFolder(): Promise<void> {
    if (
      projectFolderPickerDisabled({
        nativePathLeasesSupported: dependencies.nativePathLeasesSupported,
        busy: state.busy,
        pickingFolder: state.pickingFolder,
      })
    ) {
      return;
    }
    const pickerClaim = pickerGuard.begin();
    patch({ workspaceMode: "folder", pickingFolder: true });
    try {
      const selected = await dependencies.pickFolder();
      if (!(selected && mounted && pickerGuard.isCurrent(pickerClaim))) return;
      patch({
        folder: selected,
        name: state.name.trim() ? state.name : selected.displayName,
      });
    } catch {
      if (mounted && pickerGuard.isCurrent(pickerClaim)) {
        dependencies.showError(
          "Could not select project folder",
          "The folder was not opened. Try selecting it again.",
        );
      }
    } finally {
      if (mounted && pickerGuard.isCurrent(pickerClaim)) {
        patch({ pickingFolder: false });
      }
    }
  }

  async function submit(): Promise<void> {
    const submitted = state;
    const trimmed = submitted.name.trim();
    if (
      projectCreateDisabled({
        name: trimmed,
        busy: submitted.busy,
        pickingFolder: submitted.pickingFolder,
        stagingSources: submitted.stagingSources,
        workspaceMode: submitted.workspaceMode,
        folderToken: submitted.folder?.token,
      })
    ) {
      return;
    }

    patch({ busy: true });
    const origin = dependencies.currentRoute();
    try {
      let creation: {
        project: ProjectRecord;
        sourceUploadError: unknown | null;
      };
      try {
        creation = await createProjectWorkspace({
          name: trimmed,
          workspaceMode: submitted.workspaceMode,
          folderToken: submitted.folder?.token,
          sources: submitted.sources,
          createManaged: dependencies.createManaged,
          openFolder: dependencies.openFolder,
          uploadSources: dependencies.uploadSources,
        });
      } catch (error) {
        if (submitted.workspaceMode === "folder") {
          patch({ folder: null });
        }
        dependencies.showError(
          submitted.workspaceMode === "folder"
            ? "Failed to open project folder"
            : "Failed to create project",
          error instanceof Error
            ? `${error.message}${
                submitted.workspaceMode === "folder"
                  ? " Choose the folder again before retrying."
                  : ""
              }`
            : submitted.workspaceMode === "folder"
              ? "Choose the folder again before retrying."
              : undefined,
        );
        return;
      }
      if (!mounted) return;

      const { project, sourceUploadError } = creation;
      const stayedOnRoute = dependencies.currentRoute() === origin;
      dependencies.onOpenChange(false);
      reset();
      try {
        if (dependencies.onCreated) {
          await dependencies.onCreated(project, { stayedOnRoute });
        } else if (stayedOnRoute) {
          dependencies.activateProject(project.id);
          dependencies.navigateToProject(project.id);
        }
      } catch (error) {
        dependencies.showError(
          "Project created, but it could not be opened",
          error instanceof Error ? error.message : "Open it from Projects.",
        );
        return;
      }

      if (sourceUploadError) {
        dependencies.showError(
          "Project created, but sources were not added",
          sourceUploadError instanceof Error
            ? `${sourceUploadError.message} Add the sources again from the project Sources tab.`
            : "Add the sources again from the project Sources tab.",
        );
      }
    } finally {
      patch({ busy: false });
    }
  }

  return {
    getState: () => state,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    updateDependencies(next) {
      const supportChanged =
        dependencies.nativePathLeasesSupported !==
        next.nativePathLeasesSupported;
      dependencies = next;
      if (supportChanged) publish({ ...state });
    },
    mount() {
      mounted = true;
    },
    unmount() {
      mounted = false;
      pickerGuard.invalidate();
    },
    reset,
    setName(name) {
      patch({ name });
    },
    setSources(sources) {
      patch({ sources });
    },
    setStagingSources(stagingSources) {
      patch({ stagingSources });
    },
    openChanged(open) {
      if (open) dependencies.onOpenChange(true);
      else close();
    },
    clickCancel: close,
    clickManagedWorkspace() {
      pickerGuard.invalidate();
      patch({
        workspaceMode: "managed",
        folder: null,
        pickingFolder: false,
      });
    },
    clickExistingFolder,
    submit,
    folderPickerDisabled() {
      return projectFolderPickerDisabled({
        nativePathLeasesSupported: dependencies.nativePathLeasesSupported,
        busy: state.busy,
        pickingFolder: state.pickingFolder,
      });
    },
    createDisabled() {
      return projectCreateDisabled({
        name: state.name,
        busy: state.busy,
        pickingFolder: state.pickingFolder,
        stagingSources: state.stagingSources,
        workspaceMode: state.workspaceMode,
        folderToken: state.folder?.token,
      });
    },
  };
}
