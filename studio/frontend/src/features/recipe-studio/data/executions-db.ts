// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthSessionEpoch } from "@/features/auth";
import type { RecipeExecutionRecord } from "../execution-types";

function executionsUrl(recipeId: string): string {
  return `/api/data-recipe/recipes/${encodeURIComponent(recipeId)}/executions`;
}

export async function listRecipeExecutions(
  recipeId: string,
): Promise<RecipeExecutionRecord[]> {
  const res = await authFetch(executionsUrl(recipeId));
  if (!res.ok) throw new Error(`Load executions failed (${res.status})`);
  const { executions } = (await res.json()) as {
    executions: RecipeExecutionRecord[];
  };
  return executions;
}

function assertSameSession(epoch: number): void {
  if (getAuthSessionEpoch() !== epoch) {
    // 401 so the transient retry below never resends it.
    throw Object.assign(new Error("Signed out before the run was saved."), {
      status: 401,
    });
  }
}

async function putExecution(
  record: RecipeExecutionRecord,
  epoch: number,
): Promise<void> {
  assertSameSession(epoch);
  const res = await authFetch(
    `${executionsUrl(record.recipeId)}/${encodeURIComponent(record.id)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
    },
    { beforeRetry: () => assertSameSession(epoch) },
  );
  if (!res.ok) {
    throw Object.assign(new Error(`Save execution failed (${res.status})`), {
      status: res.status,
    });
  }
}

const RETRY_DELAYS_MS = [1000, 2000, 4000];

function isTransient(error: unknown): boolean {
  const status = (error as { status?: number }).status;
  return (
    status === undefined || status === 408 || status === 429 || status >= 500
  );
}

type Waiter = { resolve: () => void; reject: (error: unknown) => void };
type Queue = {
  record: RecipeExecutionRecord | null;
  epoch: number;
  waiters: Waiter[];
};

const queues = new Map<string, Queue>();

async function drain(id: string, queue: Queue): Promise<void> {
  let carried: Waiter[] = [];
  let attempt = 0;
  while (queue.record) {
    const { record, epoch } = queue;
    const waiters = [...carried, ...queue.waiters];
    queue.record = null;
    queue.waiters = [];
    carried = [];
    try {
      await putExecution(record, epoch);
      attempt = 0;
      for (const waiter of waiters) waiter.resolve();
    } catch (error) {
      // The final snapshot is never sent again, so retry it unless a newer one arrived.
      if (isTransient(error) && attempt < RETRY_DELAYS_MS.length) {
        await new Promise((r) => setTimeout(r, RETRY_DELAYS_MS[attempt]));
        attempt += 1;
        if (!queue.record) {
          queue.record = record;
          queue.epoch = epoch;
        }
        carried = waiters;
        continue;
      }
      attempt = 0;
      for (const waiter of waiters) waiter.reject(error);
    }
  }
  queues.delete(id);
}

// One write per run in flight, sending only the newest record after it, so an older PUT never lands last.
export function saveRecipeExecution(
  execution: RecipeExecutionRecord,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const running = queues.get(execution.id);
    const queue = running ?? { record: null, epoch: 0, waiters: [] };
    queue.record = execution;
    queue.epoch = getAuthSessionEpoch();
    queue.waiters.push({ resolve, reject });
    if (!running) {
      queues.set(execution.id, queue);
      void drain(execution.id, queue);
    }
  });
}
