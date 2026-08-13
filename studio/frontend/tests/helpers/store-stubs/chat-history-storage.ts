// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ProjectRecord } from "../../../src/features/chat/types.ts";

export type ListProjectsCall = {
  resolve: (projects: ProjectRecord[]) => void;
  reject: (error: unknown) => void;
};

let listCalls: ListProjectsCall[] = [];

/** Every list request the module under test has started, oldest first. */
export function listProjectsCalls(): ListProjectsCall[] {
  return listCalls;
}

export function resetListProjectsCalls(): void {
  listCalls = [];
}

export function listStoredChatProjects(): Promise<ProjectRecord[]> {
  let resolve!: (projects: ProjectRecord[]) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<ProjectRecord[]>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  listCalls.push({ resolve, reject });
  return promise;
}

export function isExpectedBackgroundChatStorageError(): boolean {
  return false;
}

function unused(name: string): never {
  throw new Error(`${name}: not used by these tests`);
}

export function createStoredChatProject(): never {
  return unused("createStoredChatProject");
}

export function deleteStoredChatProject(): never {
  return unused("deleteStoredChatProject");
}

export function moveStoredChatItemToProject(): never {
  return unused("moveStoredChatItemToProject");
}

export function updateStoredChatProject(): never {
  return unused("updateStoredChatProject");
}
