// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  loadPersonalization,
  savePersonalization,
  setTheme,
  useTheme,
  type Theme,
} from "@/features/settings";
import {
  DEFAULT_LOCALE,
  getLocale,
  isSupportedLocale,
  setLocale,
  useLocale,
  type Locale,
} from "@/i18n";
import { useEffect, useRef, useState } from "react";
import { useUserProfileStore } from "../stores/user-profile-store";
import type { AvatarShape } from "../stores/user-profile-store";

const PUSH_DEBOUNCE_MS = 800;

type ProfileSnapshot = {
  displayName: string;
  nickname: string;
  avatarDataUrl: string | null;
  avatarShape: AvatarShape;
};

type PersonalizationWrite = Parameters<typeof savePersonalization>[0];

function profileSnapshot(): ProfileSnapshot {
  const s = useUserProfileStore.getState();
  return {
    displayName: s.displayName,
    nickname: s.nickname,
    avatarDataUrl: s.avatarDataUrl,
    avatarShape: s.avatarShape,
  };
}

function payload(
  profile: ProfileSnapshot,
  theme: Theme,
  language: Locale | null,
): PersonalizationWrite {
  return {
    version: 1,
    profile,
    appearance: { theme, language },
  };
}

function serialized(data: PersonalizationWrite): string {
  return JSON.stringify(data);
}

function hasLocalSettings(
  profile: ProfileSnapshot,
  theme: Theme,
  language: Locale,
): boolean {
  return Boolean(
    profile.displayName ||
      profile.nickname ||
      profile.avatarDataUrl ||
      profile.avatarShape !== "circle" ||
      theme !== "system" ||
      language !== DEFAULT_LOCALE,
  );
}

export function usePersonalizationSync(enabled: boolean): void {
  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);
  const avatarShape = useUserProfileStore((s) => s.avatarShape);
  const { theme } = useTheme();
  const language = useLocale();
  const [hydratedGeneration, setHydratedGeneration] = useState(0);
  const authGenerationRef = useRef(0);
  const latestThemeRef = useRef(theme);
  const latestLanguageRef = useRef(language);
  const lastSavedRef = useRef("");

  useEffect(() => {
    latestThemeRef.current = theme;
  }, [theme]);

  useEffect(() => {
    latestLanguageRef.current = language;
  }, [language]);

  useEffect(() => {
    authGenerationRef.current += 1;
    const generation = authGenerationRef.current;
    lastSavedRef.current = "";
    if (!enabled) {
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const remote = await loadPersonalization();
        if (cancelled) return;
        if (remote.saved) {
          const nextProfile: ProfileSnapshot = {
            displayName: remote.profile.displayName ?? "",
            nickname: remote.profile.nickname ?? "",
            avatarDataUrl: remote.profile.avatarDataUrl ?? null,
            avatarShape: remote.profile.avatarShape === "rounded" ? "rounded" : "circle",
          };
          const nextTheme = remote.appearance.theme;
          const nextLanguage = isSupportedLocale(remote.appearance.language)
            ? remote.appearance.language
            : latestLanguageRef.current;
          useUserProfileStore.setState(nextProfile);
          if (nextTheme !== latestThemeRef.current) setTheme(nextTheme);
          if (nextLanguage !== latestLanguageRef.current) setLocale(nextLanguage);
          lastSavedRef.current = serialized(
            payload(nextProfile, nextTheme, nextLanguage),
          );
        } else {
          const nextProfile = profileSnapshot();
          const nextTheme = latestThemeRef.current;
          const nextLanguage = getLocale();
          const nextPayload = payload(nextProfile, nextTheme, nextLanguage);
          if (hasLocalSettings(nextProfile, nextTheme, nextLanguage)) {
            await savePersonalization(nextPayload);
          }
          lastSavedRef.current = serialized(nextPayload);
        }
        if (!cancelled && authGenerationRef.current === generation) {
          setHydratedGeneration(generation);
        }
      } catch {
        if (!cancelled && authGenerationRef.current === generation) {
          lastSavedRef.current = "";
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [enabled]);

  useEffect(() => {
    if (!enabled || hydratedGeneration !== authGenerationRef.current) return;
    const current = payload(
      { displayName, nickname, avatarDataUrl, avatarShape },
      theme,
      language,
    );
    const currentSerialized = serialized(current);
    if (currentSerialized === lastSavedRef.current) return;
    const id = window.setTimeout(() => {
      void savePersonalization(current)
        .then(() => {
          lastSavedRef.current = currentSerialized;
        })
        .catch(() => {});
    }, PUSH_DEBOUNCE_MS);
    return () => window.clearTimeout(id);
  }, [
    enabled,
    hydratedGeneration,
    displayName,
    nickname,
    avatarDataUrl,
    avatarShape,
    theme,
    language,
  ]);
}
