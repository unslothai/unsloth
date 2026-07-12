// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type AppearanceCustomization,
  type Palette,
  type Theme,
  isDefaultCustomization,
  isPalette,
  loadPersonalization,
  sanitizeCustomization,
  savePersonalization,
  setPalette,
  setTheme,
  useAppearanceCustomStore,
  usePalette,
  useTheme,
} from "@/features/settings";
import {
  DEFAULT_LOCALE,
  type Locale,
  getLocale,
  isSupportedLocale,
  setLocale,
  useLocale,
} from "@/i18n";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  PROFILE_TEXT_MAX_LENGTH,
  useUserProfileStore,
} from "../stores/user-profile-store";
import type { AvatarShape } from "../stores/user-profile-store";

const PUSH_DEBOUNCE_MS = 800;

type ProfileSnapshot = {
  displayName: string;
  nickname: string;
  avatarDataUrl: string | null;
  avatarShape: AvatarShape;
  showGreetingSloth: boolean;
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
    a.avatarShape === b.avatarShape &&
    a.showGreetingSloth === b.showGreetingSloth
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
    showGreetingSloth: s.showGreetingSloth,
  };
}

function payload(
  profile: ProfileSnapshot,
  theme: Theme,
  palette: Palette,
  customization: AppearanceCustomization,
  language: Locale | null,
): PersonalizationWrite {
  return {
    version: 1,
    profile: normalizeProfile(profile),
    appearance: { theme, palette, language, customization },
  };
}

function serialized(data: PersonalizationWrite): string {
  return JSON.stringify(data);
}

function hasLocalSettings(
  profile: ProfileSnapshot,
  theme: Theme,
  palette: Palette,
  customization: AppearanceCustomization,
  language: Locale,
): boolean {
  return Boolean(
    profile.displayName ||
      profile.nickname ||
      profile.avatarDataUrl ||
      profile.avatarShape !== "circle" ||
      !profile.showGreetingSloth ||
      theme !== "system" ||
      palette !== "standard" ||
      !isDefaultCustomization(customization) ||
      language !== DEFAULT_LOCALE,
  );
}

export function usePersonalizationSync(enabled: boolean): void {
  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);
  const avatarShape = useUserProfileStore((s) => s.avatarShape);
  const showGreetingSloth = useUserProfileStore((s) => s.showGreetingSloth);
  const { theme } = useTheme();
  const { palette } = usePalette();
  const customization = useAppearanceCustomStore((s) => s.customization);
  const language = useLocale();
  const [hydratedGeneration, setHydratedGeneration] = useState(0);
  const authGenerationRef = useRef(0);
  const latestThemeRef = useRef(theme);
  const latestPaletteRef = useRef(palette);
  const latestCustomizationRef = useRef(customization);
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
    latestPaletteRef.current = palette;
  }, [palette]);

  useEffect(() => {
    latestCustomizationRef.current = customization;
  }, [customization]);

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
            avatarShape:
              remote.profile.avatarShape === "rounded" ? "rounded" : "circle",
            showGreetingSloth: remote.profile.showGreetingSloth !== false,
          };
          const nextTheme = remote.appearance.theme;
          const nextPalette = isPalette(remote.appearance.palette)
            ? remote.appearance.palette
            : latestPaletteRef.current;
          const remoteCustomization = sanitizeCustomization(
            remote.appearance.customization,
          );
          // A record saved before the customization field existed reports
          // customizationSaved=false with the field server-defaulted. Keep local
          // overrides (and re-push them below) instead of wiping them; an
          // explicit remote reset reports customizationSaved=true and still wins.
          const localCustomization = latestCustomizationRef.current;
          const keepLocalCustomization =
            remote.customizationSaved === false &&
            !isDefaultCustomization(localCustomization);
          const nextCustomization = keepLocalCustomization
            ? localCustomization
            : remoteCustomization;
          const nextLanguage = isSupportedLocale(remote.appearance.language)
            ? remote.appearance.language
            : latestLanguageRef.current;
          useUserProfileStore.setState(nextProfile);
          if (nextTheme !== latestThemeRef.current) setTheme(nextTheme);
          if (nextPalette !== latestPaletteRef.current) setPalette(nextPalette);
          if (
            !keepLocalCustomization &&
            JSON.stringify(nextCustomization) !==
              JSON.stringify(latestCustomizationRef.current)
          ) {
            useAppearanceCustomStore.getState().replaceAll(nextCustomization);
          }
          if (nextLanguage !== latestLanguageRef.current)
            setLocale(nextLanguage);
          lastSavedRef.current = serialized(
            payload(
              nextProfile,
              nextTheme,
              nextPalette,
              // Record what the server actually has so the debounced push
              // re-uploads any preserved local customization.
              remoteCustomization,
              nextLanguage,
            ),
          );
        } else {
          const rawProfile = profileSnapshot();
          const nextProfile = normalizeProfile(rawProfile);
          if (!sameProfile(rawProfile, nextProfile)) {
            useUserProfileStore.setState(nextProfile);
          }
          const nextTheme = latestThemeRef.current;
          const nextPalette = latestPaletteRef.current;
          const nextCustomization = latestCustomizationRef.current;
          const nextLanguage = getLocale();
          const nextPayload = payload(
            nextProfile,
            nextTheme,
            nextPalette,
            nextCustomization,
            nextLanguage,
          );
          const nextSerialized = serialized(nextPayload);
          if (
            hasLocalSettings(
              nextProfile,
              nextTheme,
              nextPalette,
              nextCustomization,
              nextLanguage,
            )
          ) {
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
      { displayName, nickname, avatarDataUrl, avatarShape, showGreetingSloth },
      theme,
      palette,
      customization,
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
    showGreetingSloth,
    theme,
    palette,
    customization,
    language,
    drainSaveQueue,
  ]);
}
