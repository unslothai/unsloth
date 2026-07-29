// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

let subjectKey = "subject:a";
let firstWriteResolve: (() => void) | null = null;
let firstWriteStartedResolve: (() => void) | null = null;
let firstWriteStartedPromise: Promise<void>;
let persisted: Record<string, unknown> | null = null;

export const writes: Record<string, unknown>[] = [];
export const MAX_EXECUTION_JSON_BYTES = 1_048_576;

function resetFirstWrite(): void {
  firstWriteResolve = null;
  firstWriteStartedPromise = new Promise<void>((resolve) => {
    firstWriteStartedResolve = resolve;
  });
}

resetFirstWrite();

export function getAuthSubjectKey(): string {
  return subjectKey;
}

export class UserAssetApiError extends Error {
  readonly status: number;
  readonly detail: Record<string, unknown>;

  constructor(status = 500, detail: Record<string, unknown> = {}) {
    super(`HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

export async function upsertServerRecipeExecution<T>(
  request: Record<string, unknown>,
): Promise<T> {
  writes.push(request);
  if (writes.length === 1) {
    firstWriteStartedResolve?.();
    await new Promise<void>((resolve) => {
      firstWriteResolve = resolve;
    });
  }
  if (persisted && request.revision !== persisted.revision) {
    throw new UserAssetApiError(409, { current: persisted });
  }
  const metadata = request.metadata as Record<string, unknown>;
  persisted = {
    ...metadata,
    revision: (persisted ? Number(persisted.revision) : 0) + 1,
    updatedAt: Date.now(),
  };
  return persisted as T;
}

export function listServerRecipeExecutions<T>(): Promise<T> {
  return Promise.reject(new Error("Not used by this test."));
}

export function releaseFirstWrite(): void {
  firstWriteResolve?.();
}

export function persistedJobId(): unknown {
  return persisted?.jobId;
}

export function resetExecutionWriteFixture(): void {
  subjectKey = "subject:a";
  persisted = null;
  writes.length = 0;
  resetFirstWrite();
}

export function setAuthSubject(value: string): void {
  subjectKey = value;
}

export function waitForFirstWrite(): Promise<void> {
  return firstWriteStartedPromise;
}
