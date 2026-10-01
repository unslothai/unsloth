// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
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

async function putExecution(record: RecipeExecutionRecord): Promise<void> {
  const res = await authFetch(
    `${executionsUrl(record.recipeId)}/${encodeURIComponent(record.id)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
    },
  );
  if (!res.ok) throw new Error(`Save execution failed (${res.status})`);
}

type Waiter = { resolve: () => void; reject: (error: unknown) => void };
type Queue = { record: RecipeExecutionRecord | null; waiters: Waiter[] };

const queues = new Map<string, Queue>();

async function drain(id: string, queue: Queue): Promise<void> {
  while (queue.record) {
    const { record, waiters } = queue;
    queue.record = null;
    queue.waiters = [];
    try {
      await putExecution(record);
      for (const waiter of waiters) waiter.resolve();
    } catch (error) {
      for (const waiter of waiters) waiter.reject(error);
    }
  }
  queues.delete(id);
}

// Progress events save the same run many times a second: keep one write per run in flight and
// send only the newest record after it, so a slow older PUT can never land last. Each caller
// settles with the PUT that carried its record (or a newer one).
export function saveRecipeExecution(
  execution: RecipeExecutionRecord,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const running = queues.get(execution.id);
    const queue = running ?? { record: null, waiters: [] };
    queue.record = execution;
    queue.waiters.push({ resolve, reject });
    if (!running) {
      queues.set(execution.id, queue);
      void drain(execution.id, queue);
    }
  });
}
