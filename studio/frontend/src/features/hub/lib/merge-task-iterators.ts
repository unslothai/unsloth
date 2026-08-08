// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { combineAbortSignals } from "./abort-signals.ts";

type TaskPull<T> =
  | { status: "fulfilled"; result: IteratorResult<T> }
  | { status: "rejected"; reason: unknown };

interface TaskCursor<T> {
  iterator: AsyncGenerator<T>;
  readonly active: boolean;
  take: (onFailure: (reason: unknown) => void) => Promise<IteratorResult<T>>;
}

function createTaskCursor<T>(iterator: AsyncGenerator<T>): TaskCursor<T> {
  const pull = (): Promise<TaskPull<T>> =>
    Promise.resolve()
      .then(() => iterator.next())
      .then(
        (result): TaskPull<T> => ({ status: "fulfilled", result }),
        (reason): TaskPull<T> => ({ status: "rejected", reason }),
      );
  let pending = pull();
  let active = true;

  return {
    iterator,
    get active() {
      return active;
    },
    async take(
      onFailure: (reason: unknown) => void,
    ): Promise<IteratorResult<T>> {
      const next = await pending;
      if (next.status === "rejected") {
        active = false;
        onFailure(next.reason);
        return { done: true, value: undefined };
      }
      if (next.result.done) {
        active = false;
        return next.result;
      }
      pending = pull();
      return next.result;
    },
  };
}

function isDuplicateNamedValue<T>(seen: Set<string>, value: T): boolean {
  const name = (value as { name?: string }).name;
  if (!name) {
    return false;
  }
  const key = name.toLowerCase();
  if (seen.has(key)) {
    return true;
  }
  seen.add(key);
  return false;
}

export async function* mergeTaskIterators<Task, T>(
  tasks: readonly Task[],
  createIter: (
    task: Task | undefined,
    signal: AbortSignal,
  ) => AsyncGenerator<T>,
  parentSignal?: AbortSignal,
): AsyncGenerator<T> {
  const seen = new Set<string>();
  const taskList: readonly (Task | undefined)[] =
    tasks.length > 0 ? tasks : [undefined];
  const controller = new AbortController();
  const combined = combineAbortSignals(
    parentSignal ? [parentSignal, controller.signal] : [controller.signal],
  );
  const cursors: TaskCursor<T>[] = [];
  const failures: unknown[] = [];
  let yielded = false;

  try {
    for (const task of taskList) {
      cursors.push(createTaskCursor(createIter(task, combined.signal)));
    }
    while (cursors.some((cursor) => cursor.active)) {
      for (const cursor of cursors.filter((candidate) => candidate.active)) {
        const result = await cursor.take((reason) => failures.push(reason));
        if (result.done) {
          continue;
        }
        if (isDuplicateNamedValue(seen, result.value)) {
          continue;
        }
        yielded = true;
        yield result.value;
      }
    }

    if (!yielded && failures.length > 0) {
      throw failures[0];
    }
  } finally {
    controller.abort();
    combined.dispose();
    await Promise.allSettled(
      cursors.map(({ iterator }) => iterator.return(undefined)),
    );
  }
}
