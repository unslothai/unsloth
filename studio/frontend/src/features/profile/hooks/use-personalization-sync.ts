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
  DEFAULT_LOCALE_PREFERENCE,
  getLocalePreference,
  isLocalePreference,
  setLocale,
  useLocalePreference,
  type LocalePreference,
} from "@/i18n";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  PROFILE_TEXT_MAX_LENGTH,
  useUserProfileStore,
} from "../stores/user-profile-store";
import type { AvatarShape } from "../stores/user-profile-store";

const PUSH_DEBOUNCE_MS = 800;

// Version 2 payloads store the language preference ("auto" or a pinned
// locale). Version 1 always serialized the resolved locale, so its "en" is
// usually the old default rather than an explicit pick.
const PERSONALIZATION_VERSION = 2;

type ProfileSnapshot = {
  displayName: string;
  nickname: string;
  avatarDataUrl: string | null;
  avatarShape: AvatarShape;
};

type PersonalizationWrite = Parameters<typeof savePersonalization>[0];
type QueuedSave = {
  data: PersonalizationWrite;
  generation: number;
  serialized: string;
};
type RefValue<T> = { current: T };

function profileText(value: string): string {
  return value.slice(0, PROFILE_TEXT_MAX_LENGTH);
}

function normalizeProfile(profile: ProfileSnapshot): ProfileSnapshot {
  return {
    ...profile,
    displayName: profileText(profile.displayName),
    nickname: profileText(profile.nickname),
  };
}

function sameProfile(a: ProfileSnapshot, b: ProfileSnapshot): boolean {
  return (
    a.displayName === b.displayName &&
    a.nickname === b.nickname &&
    a.avatarDataUrl === b.avatarDataUrl &&
    a.avatarShape === b.avatarShape
  );
}

function drainQueuedSave(
  saveInFlightRef: RefValue<boolean>,
  queuedSaveRef: RefValue<QueuedSave | null>,
  authGenerationRef: RefValue<number>,
  lastSavedRef: RefValue<string>,
): void {
  if (saveInFlightRef.current) return;
  const next = queuedSaveRef.current;
  if (!next) return;
  queuedSaveRef.current = null;
  saveInFlightRef.current = true;
  void savePersonalization(next.data)
    .then(() => {
      if (authGenerationRef.current === next.generation) {
        lastSavedRef.current = next.serialized;
      }
    })
    .catch(() => {
      if (authGenerationRef.current === next.generation) {
        lastSavedRef.current = "";
      }
    })
    .finally(() => {
      saveInFlightRef.current = false;
      const queued = queuedSaveRef.current;
      if (queued && authGenerationRef.current === queued.generation) {
        drainQueuedSave(
          saveInFlightRef,
          queuedSaveRef,
          authGenerationRef,
          lastSavedRef,
        );
      }
    });
}

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
  language: LocalePreference | null,
): PersonalizationWrite {
  return {
    version: PERSONALIZATION_VERSION,
    profile: normalizeProfile(profile),
    appearance: { theme, language },
  };
}

function serialized(data: PersonalizationWrite): string {
  return JSON.stringify(data);
}

// Version 1 clients wrote language on every save, so a legacy "en" usually
// means the user never picked a language. Map it to auto; explicit picks of
// other locales (the old default was English) are kept. Version 2 payloads
// are trusted verbatim, so a deliberate English pick stays pinned.
export function remoteLanguagePreference(
  version: unknown,
  language: unknown,
): unknown {
  const isLegacy = typeof version !== "number" || version < 2;
  if (isLegacy && language === "en") return DEFAULT_LOCALE_PREFERENCE;
  return language;
}

function hasLocalSettings(
  profile: ProfileSnapshot,
  theme: Theme,
  language: LocalePreference,
): boolean {
  return Boolean(
    profile.displayName ||
      profile.nickname ||
      profile.avatarDataUrl ||
      profile.avatarShape !== "circle" ||
      theme !== "system" ||
      language !== DEFAULT_LOCALE_PREFERENCE,
  );
}

export function usePersonalizationSync(enabled: boolean): void {
  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);
  const avatarShape = useUserProfileStore((s) => s.avatarShape);
  const { theme } = useTheme();
  const language = useLocalePreference();
  const [hydratedGeneration, setHydratedGeneration] = useState(0);
  const authGenerationRef = useRef(0);
  const latestThemeRef = useRef(theme);
  const latestLanguageRef = useRef(language);
  const lastSavedRef = useRef("");
  const saveInFlightRef = useRef(false);
  const queuedSaveRef = useRef<QueuedSave | null>(null);

  const drainSaveQueue = useCallback(() => {
    drainQueuedSave(
      saveInFlightRef,
      queuedSaveRef,
      authGenerationRef,
      lastSavedRef,
    );
  }, []);

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
    queuedSaveRef.current = null;
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
          const remoteLanguage = remoteLanguagePreference(
            remote.version,
            remote.appearance.language,
          );
          const nextLanguage = isLocalePreference(remoteLanguage)
            ? remoteLanguage
            : latestLanguageRef.current;
          useUserProfileStore.setState(nextProfile);
          if (nextTheme !== latestThemeRef.current) setTheme(nextTheme);
          if (nextLanguage !== latestLanguageRef.current) setLocale(nextLanguage);
          lastSavedRef.current = serialized(
            payload(nextProfile, nextTheme, nextLanguage),
          );
        } else {
          const rawProfile = profileSnapshot();
          const nextProfile = normalizeProfile(rawProfile);
          if (!sameProfile(rawProfile, nextProfile)) {
            useUserProfileStore.setState(nextProfile);
          }
          const nextTheme = latestThemeRef.current;
          const nextLanguage = getLocalePreference();
          const nextPayload = payload(nextProfile, nextTheme, nextLanguage);
          const nextSerialized = serialized(nextPayload);
          if (hasLocalSettings(nextProfile, nextTheme, nextLanguage)) {
            try {
              await savePersonalization(nextPayload);
              lastSavedRef.current = nextSerialized;
            } catch {
              lastSavedRef.current = "";
            }
          } else {
            lastSavedRef.current = nextSerialized;
          }
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
      queuedSaveRef.current = {
        data: current,
        generation: authGenerationRef.current,
        serialized: currentSerialized,
      };
      drainSaveQueue();
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
    drainSaveQueue,
  ]);
}
