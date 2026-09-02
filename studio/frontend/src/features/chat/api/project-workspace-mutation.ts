// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export class ProjectWorkspaceMutationError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ProjectWorkspaceMutationError";
    this.status = status;
  }
}

export function projectWorkspaceMutationShouldStayBusy(
  error: unknown,
): boolean {
  return (
    error instanceof ProjectWorkspaceMutationError &&
    error.status === 409 &&
    error.message.toLowerCase().includes("workspace changed before")
  );
}
