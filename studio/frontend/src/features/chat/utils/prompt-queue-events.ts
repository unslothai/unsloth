// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Prompt-queue event names and their detail shapes. Split out of prompt-queue-boundary so
 *  thread.tsx can read them from module scope: features/chat is in an import cycle, so reaching
 *  these through the barrel risks the temporal dead zone that crashed use-model-memory.ts. Imports
 *  nothing on purpose: ESM evaluates dependencies first, so a module with none cannot be
 *  re-entered mid-initialization. One import reopens the hazard. */

export const PROMPT_QUEUE_STOP_EVENT = "unsloth:prompt-queue-stop";
export const PROMPT_QUEUE_RUN_FAILED_EVENT = "unsloth:prompt-queue-run-failed";

export type PromptQueueStopEventDetail = {
  threadIds?: string[];
  temporaryOnly?: boolean;
  localOnly?: boolean;
};

export type PromptQueueRunFailedEventDetail = {
  threadId?: string | null;
};
