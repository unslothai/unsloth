// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { updateChatThread } from "../api/chat-api";
import type { ThreadRecord } from "../types";

export async function updateChatTitles(
  threads: Pick<ThreadRecord, "id" | "title">[],
  title: string,
): Promise<void> {
  const results = await Promise.allSettled(
    threads.map((thread) =>
      updateChatThread(thread.id, { title }, { expectedTitle: thread.title }),
    ),
  );
  const failure = results.find((result) => result.status === "rejected");
  if (failure) {
    await Promise.all(
      threads.map((thread, index) =>
        results[index].status === "fulfilled"
          ? updateChatThread(
              thread.id,
              { title: thread.title },
              { expectedTitle: title },
            )
          : Promise.resolve(),
      ),
    );
    throw failure.reason;
  }
}
