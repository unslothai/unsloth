// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { lazy, Suspense, type ComponentProps } from "react";
import type { PromptQueueList as QueueList } from "./prompt-queue-list";

const List = lazy(() =>
  import("./prompt-queue-list").then((module) => ({
    default: module.PromptQueueList,
  })),
);

export function PromptQueueList(props: ComponentProps<typeof QueueList>) {
  return (
    <Suspense
      fallback={
        <div role="status" className="px-5 py-2 text-sm text-muted-foreground">
          Loading queued prompts
        </div>
      }
    >
      <List {...props} />
    </Suspense>
  );
}
