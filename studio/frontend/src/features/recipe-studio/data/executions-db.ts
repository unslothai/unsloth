// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { RecipeExecutionRecord } from "../execution-types";

const latest = new Map<string, RecipeExecutionRecord>();
const writes = new Map<string, Promise<void>>();

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

// Progress events save the same run many times a second: keep one write per run in flight and
// send only the newest record after it, so a slow older PUT can never land last.
export function saveRecipeExecution(
  execution: RecipeExecutionRecord,
): Promise<void> {
  const { id } = execution;
  latest.set(id, execution);
  const write = (writes.get(id) ?? Promise.resolve())
    .catch(() => undefined)
    .then(() => {
      const record = latest.get(id);
      if (!record) return;
      latest.delete(id);
      return putExecution(record);
    });
  writes.set(id, write);
  const settle = () => {
    if (writes.get(id) === write) writes.delete(id);
  };
  write.then(settle, settle);
  return write;
}
