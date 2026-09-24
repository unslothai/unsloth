// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Media pages stay mounted off-screen, so one deleted elsewhere (the Library) has to be told.
const EVENT = "unsloth:gallery-item-removed";

export type GalleryKind = "image" | "video" | "audio";

type Detail = { kind: GalleryKind; id: string };

export function announceGalleryRemoval(kind: GalleryKind, id: string): void {
  window.dispatchEvent(new CustomEvent<Detail>(EVENT, { detail: { kind, id } }));
}

/** Calls `drop` with the id of each removed item of `kind`; returns the unsubscribe. */
export function onGalleryRemoval(kind: GalleryKind, drop: (id: string) => void): () => void {
  const listener = (event: Event) => {
    const detail = (event as CustomEvent<Detail>).detail;
    if (detail.kind === kind) drop(detail.id);
  };
  window.addEventListener(EVENT, listener);
  return () => window.removeEventListener(EVENT, listener);
}
