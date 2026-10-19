// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a load unloaded to make room, as one notice. Chat raises it as its own toast; Images and
// Video put it on their "Model loaded" toast, so a swap never stacks two toasts.

import {
  notifyModelEjected,
  notifyModelLifecycle,
} from "@/lib/model-lifecycle-events";
import { toast } from "@/lib/toast";

export type EvictedKind = "chat" | "audio" | "image" | "video";
export type EvictedSlot = { kind: EvictedKind; model: string };

const KINDS: readonly EvictedKind[] = ["chat", "audio", "image", "video"];

function isKind(value: unknown): value is EvictedKind {
  return KINDS.includes(value as EvictedKind);
}

/** Reads `evicted_slots`; falls back to `evicted: string[]` as chat models; anything else is []. */
export function evictedSlotsFrom(answer: unknown): EvictedSlot[] {
  if (!answer || typeof answer !== "object") return [];
  const { evicted_slots: slots, evicted } = answer as {
    evicted_slots?: unknown;
    evicted?: unknown;
  };
  if (Array.isArray(slots)) {
    return slots.flatMap((slot) => {
      const { kind, model } = (slot ?? {}) as {
        kind?: unknown;
        model?: unknown;
      };
      return isKind(kind) && typeof model === "string" && model
        ? [{ kind, model }]
        : [];
    });
  }
  if (Array.isArray(evicted)) {
    return evicted
      .filter((model): model is string => typeof model === "string" && !!model)
      .map((model) => ({ kind: "chat" as const, model }));
  }
  return [];
}

const DESCRIPTIONS: Record<EvictedKind, [one: string, many: string]> = {
  chat: [
    "Select it in chat to load it back.",
    "Select them in chat to load them back.",
  ],
  image: [
    "Load it again on the Images page.",
    "Load them again on the Images page.",
  ],
  video: [
    "Load it again on the Video page.",
    "Load them again on the Video page.",
  ],
  audio: [
    "Load it again on its audio page.",
    "Load them again on their audio pages.",
  ],
};

export function evictedNotice(
  slots: EvictedSlot[],
): { title: string; description: string } | null {
  const models = [...new Set(slots.map((slot) => slot.model))];
  if (models.length === 0) return null;
  const kinds = [...new Set(slots.map((slot) => slot.kind))];
  const description =
    kinds.length > 1
      ? "Load them again where you use them."
      : DESCRIPTIONS[kinds[0]][models.length > 1 ? 1 : 0];
  return { title: `Unloaded ${models.join(", ")} to make room`, description };
}

/** For the Images and Video "Model loaded" toast: the notice as its description, or nothing. */
export function evictedToastOptions(slots: EvictedSlot[]): {
  description?: string;
} {
  const notice = evictedNotice(slots);
  return notice
    ? { description: `${notice.title}. ${notice.description}` }
    : {};
}

export function toastEvicted(slots: EvictedSlot[]): void {
  const notice = evictedNotice(slots);
  if (notice) toast.info(notice.title, { description: notice.description });
}

/**
 * Tell the other runtimes their model is gone. Never `self`: the Images page treats its own
 * eject as a cancel of the load in flight, which is this very load.
 */
export function resyncEvicted(slots: EvictedSlot[], self: EvictedKind): void {
  const kinds = new Set(slots.map((slot) => slot.kind));
  kinds.delete(self);
  if (kinds.has("image")) notifyModelEjected("image");
  if (kinds.has("video")) notifyModelEjected("video");
  if (kinds.has("chat") || kinds.has("audio")) {
    notifyModelLifecycle({ runtime: "chat", loading: false, model: null });
  }
}
