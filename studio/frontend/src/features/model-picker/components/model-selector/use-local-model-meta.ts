// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reads the local GGUF header for the model-info panel. Separate from the Hub lookup, so one
// being unavailable does not hide the other.

import { fetchGgufStagedMetadata } from "@/features/chat/api/chat-api";
import { useEffect, useState } from "react";
import type { LocalModelMeta } from "./local-model-facts";

/** Header facts for `repoId` at `variant`, or null while unread. Never throws. */
export function useLocalModelMeta(
  repoId: string | null,
  {
    variant,
    hfToken,
    enabled,
  }: { variant?: string | null; hfToken?: string | undefined; enabled: boolean },
): LocalModelMeta | null {
  const [state, setState] = useState<{
    key: string;
    meta: LocalModelMeta | null;
  }>(() => ({ key: "", meta: null }));

  // `enabled` and the token are in the KEY, not just the deps: the effect returns early when
  // the probe is off, so a key blind to them keeps the last file's facts on screen — delete a
  // GGUF with its dialog open and the panel still reports its context length.
  const key = `${repoId ?? ""}::${variant ?? ""}::${enabled ? "on" : "off"}::${hfToken ?? ""}`;

  useEffect(() => {
    if (!(repoId && enabled)) return;
    let cancelled = false;

    fetchGgufStagedMetadata({
      model_path: repoId,
      gguf_variant: variant ?? null,
      hf_token: hfToken ?? null,
      includeChatTemplate: true,
    })
      .then((res) => {
        if (cancelled) return;
        setState({
          key,
          meta: {
            contextLength: res.contextLength,
            layerCount: res.layerCount,
            moeLayerCount: res.moeLayerCount,
            chatTemplate: res.chatTemplate,
          },
        });
      })
      // Unreadable is the same as nothing found: show the Hub half, not an error.
      .catch(() => {
        if (!cancelled) setState({ key, meta: null });
      });

    return () => {
      cancelled = true;
    };
  }, [repoId, variant, hfToken, enabled, key]);
  // Listed explicitly though `key` encodes them, so dropping one from the key cannot silently
  // stop the effect re-firing.

  return state.key === key ? state.meta : null;
}
