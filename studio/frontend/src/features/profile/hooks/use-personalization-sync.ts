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

// Version 2 payloads store the language preference ("auto" or a pinned locale). Version 1 always
// serialized the resolved locale, so its "en" is usually the old default rather than an explicit
// pick. Version 3 migrates untouched sidebar layouts to keep Video under More. Version 4 pins Video
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

// v1 wrote language on every save, so a legacy "en" maps to auto; v2 payloads are trusted verbatim.
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
          // Legacy records return server defaults for missing fields: keep and re-push the local value.
          // A record with <field>Saved=true still wins.
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
          const keepLocalChatWidth =
            remote.chatWidthSaved === false ||
            remote.appearance.customization?.chatWidth === undefined;
          const keepLocalSentAttachments =
            remote.sentAttachmentsSaved === false ||
            remote.appearance.customization?.sentAttachments === undefined;
          const nextCustomization = keepLocalCustomization
            ? localCustomization
            : {
                ...remoteCustomization,
                ...(keepLocalChatWidth && {
                  chatWidth: localCustomization.chatWidth,
                }),
                ...(keepLocalSentAttachments && {
                  sentAttachments: localCustomization.sentAttachments,
                }),
              };
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
              // A failing catalog must not block sync; adopting keeps local equal to server so the push is honest.
              adoptOnFailure: true,
              // A request that never completes must not hold hydration (and all saves) open forever.
              timeoutMs: LOCALE_INITIALIZATION_TIMEOUT_MS,
            });
            if (cancelled) return;
            // "superseded" is not the language in effect, so record no baseline; hydration must still finish.
            if (localeResult === "cancelled") return;
            if (localeResult === "superseded") {
              if (authGenerationRef.current === generation) {
                lastSavedRef.current = "";
                setHydratedGeneration(generation);
              }
              return;
            }
          }
          // lastSaved records the server's actual values so the push re-uploads preserved local values.
          lastSavedRef.current = serialized({
            ...payload(
              { ...nextProfile, showGreetingSloth: remoteGreeting },
              nextTheme,
              remotePalette,
              storedRemoteCustomization,
              nextLanguage,
            ),
            // Keep the server's version so a legacy record is re-saved even with a customized layout.
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
