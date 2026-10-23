// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

import type { MediaValue } from "../types";
import { type PlayerApi, mediaObjectUrl } from "./player-api";

/** An object URL for `media` (fetched with auth, cached), or null while loading / on failure. */
export function useMediaUrl(
  api: PlayerApi,
  media: MediaValue | null | undefined,
): string | null {
  const key = media ? media.ref + (media.url ?? "") : "";
  const [state, setState] = useState<{ key: string; url: string | null }>({
    key: "",
    url: null,
  });
  // biome-ignore lint/correctness/useExhaustiveDependencies: `key` identifies the media; the object changes identity every render
  useEffect(() => {
    if (!media) {
      return;
    }
    let live = true;
    mediaObjectUrl(api, media)
      .then((url) => {
        if (live) setState({ key, url });
      })
      .catch(() => {
        if (live) setState({ key, url: null });
      });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, key]);
  return state.key === key ? state.url : null;
}
