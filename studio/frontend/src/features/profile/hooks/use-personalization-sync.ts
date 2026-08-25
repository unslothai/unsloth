// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type AppearanceCustomization,
  type Palette,
  type Theme,
  isDefaultCustomization,
  isPalette,
  loadPersonalization,
  migrateShippedSidebarNavDefault,
  sanitizeCustomization,
  savePersonalization,
  setPalette,
  setTheme,
  useAppearanceCustomStore,
  usePalette,
  useTheme,
} from "@/features/settings";
import {
  DEFAULT_LOCALE_PREFERENCE,
  LOCALE_INITIALIZATION_TIMEOUT_MS,
  type LocalePreference,
  getLocalePreference,
  isLocalePreference,
  setLocale,
  useLocalePreference,
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
// Version 3 migrates untouched sidebar layouts to keep Video under More.
// Version 4 pins Video under Images. Version 5 pins notebooks under Images.
// Without this bump a synced profile rehydrates its stored layout over the 
// local migration.
const PERSONALIZATION_VERSION = 5;

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
  language: LocalePreference | null,
): PersonalizationWrite {
  return {
    version: PERSONALIZATION_VERSION,
    profile: normalizeProfile(profile),
    appearance: { theme, palette, language, customization },
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
  palette: Palette,
  customization: AppearanceCustomization,
  language: LocalePreference,
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
      language !== DEFAULT_LOCALE_PREFERENCE,
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
  const language = useLocalePreference();
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
    const localeHydrationController = new AbortController();
    void (async () => {
      try {
        const remote = await loadPersonalization();
        if (cancelled) return;
        if (remote.saved) {
          // Legacy records predating a field come back server-defaulted. Keep
          // the local value and re-push it (lastSaved below records the remote
          // default so the push detects the diff) rather than treating the
          // default as an explicit remote choice. A record that actually stored
          // the field reports <field>Saved=true and still wins.
          const localGreeting =
            useUserProfileStore.getState().showGreetingSloth;
          const remoteGreeting = remote.profile.showGreetingSloth !== false;
          const keepLocalGreeting =
            remote.greetingSlothSaved === false && localGreeting === false;
          const nextProfile: ProfileSnapshot = {
            displayName: remote.profile.displayName ?? "",
            nickname: remote.profile.nickname ?? "",
            avatarDataUrl: remote.profile.avatarDataUrl ?? null,
            avatarShape:
              remote.profile.avatarShape === "rounded" ? "rounded" : "circle",
            showGreetingSloth: keepLocalGreeting
              ? localGreeting
              : remoteGreeting,
          };
          const nextTheme = remote.appearance.theme;
          const localPalette = latestPaletteRef.current;
          const remotePalette = isPalette(remote.appearance.palette)
            ? remote.appearance.palette
            : localPalette;
          const keepLocalPalette =
            remote.paletteSaved === false && localPalette !== "standard";
          const nextPalette = keepLocalPalette ? localPalette : remotePalette;
          const storedRemoteCustomization = sanitizeCustomization(
            remote.appearance.customization,
          );
          const remoteCustomization = migrateShippedSidebarNavDefault(
            storedRemoteCustomization,
            remote.version,
            PERSONALIZATION_VERSION,
          );
          const localCustomization = latestCustomizationRef.current;
          const keepLocalCustomization =
            remote.customizationSaved === false &&
            !isDefaultCustomization(localCustomization);
          const nextCustomization = keepLocalCustomization
            ? localCustomization
            : remoteCustomization;
          const remoteLanguage = remoteLanguagePreference(
            remote.version,
            remote.appearance.language,
          );
          const nextLanguage = isLocalePreference(remoteLanguage)
            ? remoteLanguage
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
          if (nextLanguage !== latestLanguageRef.current) {
            const localeResult = await setLocale(nextLanguage, {
              signal: localeHydrationController.signal,
              // A catalog that will not load must not decide whether the rest of
              // personalization syncs. Adopting the preference and rendering
              // English keeps the local preference equal to the server's, so the
              // baseline below is honest and the debounced push cannot overwrite
              // the remote language with a stale local one.
              adoptOnFailure: true,
              // And a catalog request that is accepted but never completes
              // must not hold hydration, and with it every save for the rest
              // of the session, open forever. Same bound as startup.
              timeoutMs: LOCALE_INITIALIZATION_TIMEOUT_MS,
            });
            if (cancelled) return;
            // "superseded" means a newer request took over, so this language is
            // no longer the one in effect and must not be recorded as the
            // synchronized baseline: the newer request may itself have failed,
            // leaving the local preference on neither value. Hydration still has
            // to finish, or every later save stays paused for the session; an
            // empty baseline makes the next push send whatever is in effect.
            if (localeResult === "cancelled") return;
            if (localeResult === "superseded") {
              if (authGenerationRef.current === generation) {
                lastSavedRef.current = "";
                setHydratedGeneration(generation);
              }
              return;
            }
          }
          // lastSaved records what the server actually has (server-side defaults
          // for legacy fields) so the debounced push re-uploads preserved local
          // values.
          lastSavedRef.current = serialized({
            ...payload(
              { ...nextProfile, showGreetingSloth: remoteGreeting },
              nextTheme,
              remotePalette,
              storedRemoteCustomization,
              nextLanguage,
            ),
            // Preserve the server's actual version here so a legacy record is
            // re-saved even when the sidebar layout itself was customized.
            version: remote.version,
          });
        } else {
          const rawProfile = profileSnapshot();
          const nextProfile = normalizeProfile(rawProfile);
          if (!sameProfile(rawProfile, nextProfile)) {
            useUserProfileStore.setState(nextProfile);
          }
          const nextTheme = latestThemeRef.current;
          const nextPalette = latestPaletteRef.current;
          const nextCustomization = latestCustomizationRef.current;
          const nextLanguage = getLocalePreference();
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
      localeHydrationController.abort();
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
