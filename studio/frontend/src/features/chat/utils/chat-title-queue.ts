// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createSerialQueue } from "./serial-queue";

type TitleQueue = {
  run: ReturnType<typeof createSerialQueue>;
  pending: number;
};
const queues = new Map<string, TitleQueue>();

export function queueChatTitle<T>(
  key: string,
  task: () => Promise<T>,
): Promise<T> {
  const queue = queues.get(key) ?? { run: createSerialQueue(), pending: 0 };
  queues.set(key, queue);
  queue.pending += 1;
  return queue.run(task).finally(() => {
    queue.pending -= 1;
    if (queue.pending === 0) queues.delete(key);
  });
}
