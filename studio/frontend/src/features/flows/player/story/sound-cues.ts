// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MediaValue } from "../../types";

export function playCues(
  names: string[],
  sounds: Record<string, MediaValue> | null | undefined,
  lookup: (media: MediaValue) => Promise<string>,
  current: () => boolean,
  play: (url: string) => Promise<unknown> = (url) => new Audio(url).play(),
): void {
  for (const name of names) {
    const media = sounds?.[name];
    if (!media) continue;
    // The lookup can outlive the playthrough it was cued in (a save load or restart).
    void lookup(media)
      .then((url) => (current() ? play(url) : undefined))
      .catch(() => undefined);
  }
}
