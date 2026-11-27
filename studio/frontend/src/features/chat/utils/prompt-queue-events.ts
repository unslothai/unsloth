// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Imports nothing on purpose: features/chat has an import cycle, and one import reopens it. */

export const PROMPT_QUEUE_STOP_EVENT = "unsloth:prompt-queue-stop";
export const PROMPT_QUEUE_RUN_FAILED_EVENT = "unsloth:prompt-queue-run-failed";

export type PromptQueueStopEventDetail = {
  threadIds?: string[];
  temporaryOnly?: boolean;
  localOnly?: boolean;
};

export type PromptQueueRunFailedEventDetail = {
  threadId?: string | null;
  localOnly?: boolean;
};
