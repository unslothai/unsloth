// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ProjectRecord } from "../types";

export const MANAGED_FOLDER_LABEL = "Unsloth-managed folder";

export type ProjectFolderState = {
  chosen: boolean;
  unavailable: boolean;
  label: string;
  path: string | null;
  branch: string | null;
  problem: string | null;
};

type FolderFields = Pick<
  ProjectRecord,
  | "workingFolder"
  | "workingFolderStatus"
  | "workingFolderDisplay"
  | "workingFolderBranch"
  | "sandboxPath"
>;

export function projectFolderState(project: FolderFields): ProjectFolderState {
  const folder = project.workingFolder ?? null;
  const managed = project.sandboxPath ?? null;
  if (!folder) {
    return {
      chosen: false,
      unavailable: false,
      label: MANAGED_FOLDER_LABEL,
      path: managed,
      branch: null,
      problem: null,
    };
  }
  const status = project.workingFolderStatus ?? "ok";
  const unavailable = status !== "ok";
  return {
    chosen: true,
    unavailable,
    label: project.workingFolderDisplay ?? folder,
    path: folder,
    branch: unavailable ? null : (project.workingFolderBranch ?? null),
    problem: !unavailable
      ? null
      : status === "replaced"
        ? "A different folder is at this path now. Tools are using the Unsloth-managed folder until you choose the folder again."
        : "This folder can't be found. Tools are using the Unsloth-managed folder until you choose it again.",
  };
}

export function shortenFolderLabel(label: string, max = 48): string {
  if (label.length <= max) return label;
  const tail = label.slice(-(max - 1));
  const cut = tail.search(/[\\/]/);
  return `…${cut > 0 ? tail.slice(cut) : tail}`;
}
