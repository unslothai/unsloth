// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  getChatSettings,
  saveChatSettingsPatchIfCurrent,
} from "./chat-settings-api";
import { ChatSettingsRequestError } from "../utils/settings-retry";
import {
  type PinOperation,
  PinnedChatsValidationError,
  validatePinnedIds,
} from "../utils/pinned-chats-sync";

/** Scoped to one auth session; every refresh probes for a backend upgrade. */
export function createChatPinsApi(sessionSignal: AbortSignal) {
  let supportsOperationId = false;

  async function bounded<T>(
    run: (signal: AbortSignal) => Promise<T>,
  ): Promise<T> {
    // The supported WebKit floor lacks AbortSignal.any; release listener and timer per request.
    const controller = new AbortController();
    const abort = () => controller.abort();
    sessionSignal.addEventListener("abort", abort);
    if (sessionSignal.aborted) abort();
    const timer = setTimeout(abort, 10_000);
    try {
      return await run(controller.signal);
    } finally {
      clearTimeout(timer);
      sessionSignal.removeEventListener("abort", abort);
    }
  }

  async function request(
    path: string,
    ids?: string[],
  ): Promise<string[] | undefined> {
    return bounded(async (signal) => {
      const response = await authFetch(path, {
        signal,
        ...(ids === undefined
          ? {}
          : {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ ids }),
            }),
      });
      const data: unknown = await response.json().catch((error: unknown) => {
        // A 200 whose body was cut off is a transport failure to retry, not an invalid answer.
        if (signal.aborted || response.ok) throw error;
        return null;
      });
      if (!response.ok) {
        const detail =
          data && typeof data === "object" && "detail" in data
            ? data.detail
            : null;
        throw new ChatSettingsRequestError(
          `Unable to sync pinned chats (${response.status})`,
          response.status,
          detail,
        );
      }
      if (!data || typeof data !== "object" || Array.isArray(data))
        throw new PinnedChatsValidationError("Invalid pinned chats response");
      supportsOperationId = true;
      return validatePinnedIds(
        "pinnedIds" in data ? data.pinnedIds : undefined,
      );
    });
  }

  function missingRoute(error: unknown): boolean {
    return (
      error instanceof ChatSettingsRequestError &&
      (error.status === 404 || error.status === 405)
    );
  }

  async function compareAndSet(
    expected: string[] | undefined,
    next: string[],
    operation?: PinOperation,
  ) {
    const result = await bounded((signal) =>
      saveChatSettingsPatchIfCurrent(
        expected === undefined ? {} : { pinnedChatIds: expected },
        { pinnedChatIds: next },
        expected === undefined ? ["pinnedChatIds"] : [],
        [],
        {
          signal,
          requireSupport: true,
          operationId: supportsOperationId
            ? (operation?.serverId ?? operation?.operationId)
            : undefined,
          unpinnedChatIds:
            supportsOperationId && operation && !operation.pinned
              ? operation.ids
              : undefined,
          cancelledOperationIds: supportsOperationId
            ? operation?.cancels
            : undefined,
        },
      ),
    );
    return {
      applied: result.applied,
      pinnedIds: validatePinnedIds(result.settings?.pinnedChatIds),
    };
  }

  return {
    async load(): Promise<string[] | undefined> {
      try {
        return await request("/api/chat/pins");
      } catch (error) {
        if (!missingRoute(error)) throw error;
        supportsOperationId = false;
        return validatePinnedIds(
          (await bounded((signal) => getChatSettings({ signal })))
            .pinnedChatIds,
        );
      }
    },
    async migrateLegacy(ids: string[]): Promise<string[] | undefined> {
      try {
        return await request("/api/chat/pins/migrate", ids);
      } catch (error) {
        if (!missingRoute(error)) throw error;
        supportsOperationId = false;
        const current = validatePinnedIds(
          (await bounded((signal) => getChatSettings({ signal })))
            .pinnedChatIds,
        );
        // A backend with the setting but no tombstones cannot merge stale browser lists.
        if (current !== undefined) return current;
        return (await compareAndSet(undefined, ids)).pinnedIds;
      }
    },
    compareAndSet,
  };
}
