// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Prompt-queue event names and their detail shapes.
 *
 * Split out of prompt-queue-boundary so they can be read while a module is
 * still loading. thread.tsx registers its window listeners at module scope, and
 * features/chat is inside an import cycle, so reading these through the barrel
 * could land in the temporal dead zone and throw at import time -- the same
 * failure use-model-memory.ts hit with CHAT_GPU_MEMORY_MODE_KEY.
 *
 * This module imports nothing on purpose. ESM evaluates dependencies before a
 * module body, so a module with no dependencies has nothing that can re-enter
 * it: its bindings are always initialized by the time any importer runs. Keep
 * it that way -- one import here reopens the hazard.
 */

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
