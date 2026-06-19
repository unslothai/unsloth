// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  loadPersonalization,
  savePersonalization,
} from "@/features/settings/api/personalization";
import { setTheme, useTheme } from "@/features/settings/stores/theme-store";
import { useEffect, useRef } from "react";
import { useUserProfileStore } from "../stores/user-profile-store";

const PUSH_DEBOUNCE_MS = 800;

/**
 * Keep profile + appearance in sync with the server so personalization follows
 * the account across browsers and devices (these were previously localStorage
 * only). When the account has a saved blob the server is authoritative; when it
 * does not, existing local settings are migrated up once so nothing is lost.
 * Writers (the profile panel, theme toggles) keep using the local stores; this
 * hook mirrors changes to the server, debounced.
 */
export function usePersonalizationSync(enabled: boolean): void {
  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);
  const avatarShape = useUserProfileStore((s) => s.avatarShape);
  const { theme } = useTheme();
  const hydratedRef = useRef(false);

  // Hydrate once when authenticated.
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    void (async () => {
      try {
        const remote = await loadPersonalization();
        if (cancelled) return;
        if (remote.saved) {
          // Server is authoritative.
          useUserProfileStore.setState({
            displayName: remote.profile.displayName ?? "",
            nickname: remote.profile.nickname ?? "",
            avatarDataUrl: remote.profile.avatarDataUrl ?? null,
            avatarShape: remote.profile.avatarShape === "rounded" ? "rounded" : "circle",
          });
          const t = remote.appearance.theme;
          if (t === "light" || t === "dark" || t === "system") {
            setTheme(t);
          }
        } else {
          // Never saved: migrate the current local settings up once.
          const s = useUserProfileStore.getState();
          await savePersonalization({
            version: 1,
            profile: {
              displayName: s.displayName,
              nickname: s.nickname,
              avatarDataUrl: s.avatarDataUrl,
              avatarShape: s.avatarShape,
            },
            appearance: { theme, language: null },
          });
        }
      } catch {
        // Non-fatal: fall back to local-only behavior.
      } finally {
        if (!cancelled) hydratedRef.current = true;
      }
    })();
    return () => {
      cancelled = true;
    };
    // theme is intentionally excluded: hydration reads getState()/the current
    // theme once and must not re-run on later theme changes (the write-through
    // effect handles those).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);

  // Write through subsequent local changes to the server, debounced. Skipped
  // until hydration completes so we never echo the just-hydrated values back.
  useEffect(() => {
    if (!enabled || !hydratedRef.current) return;
    const id = window.setTimeout(() => {
      void savePersonalization({
        version: 1,
        profile: { displayName, nickname, avatarDataUrl, avatarShape },
        appearance: { theme, language: null },
      }).catch(() => {});
    }, PUSH_DEBOUNCE_MS);
    return () => window.clearTimeout(id);
  }, [enabled, displayName, nickname, avatarDataUrl, avatarShape, theme]);
}
