// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { GalleryImage } from "../api";

export function promptBoundaryIds(images: readonly Pick<GalleryImage, "id" | "prompt">[]): Set<string> {
  const boundaries = new Set<string>();
  for (let index = 1; index < images.length; index++) {
    const previous = images[index - 1].prompt;
    const current = images[index].prompt;
    // Missing prompts do not tell us whether the neighboring generations differ.
    if (previous.trim() && current.trim() && previous !== current) {
      boundaries.add(images[index].id);
    }
  }
  return boundaries;
}
