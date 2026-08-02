// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback } from "react";
import { useLocale } from "./locale-store";
import { translate } from "./messages";
import type { InterpolationValues } from "./types";
import type { TranslationKey } from "./messages";

export {
  AUTO_LOCALE,
  DEFAULT_LOCALE,
  DEFAULT_LOCALE_PREFERENCE,
  LOCALE_STORAGE_KEY,
  getLocale,
  getLocalePreference,
  initializeLocale,
  isLocalePreference,
  setLocale,
  subscribeLocale,
  useLocale,
  useLocalePreference,
} from "./locale-store";
export type { LocalePreference } from "./locale-store";
export {
  LOCALES,
  isSupportedLocale,
  messages,
  translate,
} from "./messages";
export type { Locale, TranslationKey } from "./messages";
export type {
  DeepPartialMessageTree,
  InterpolationValues,
  MessageKey,
  MessageTree,
} from "./types";

export function useT(): (
  key: TranslationKey,
  values?: InterpolationValues,
) => string {
  const locale = useLocale();

  return useCallback(
    (key: TranslationKey, values?: InterpolationValues) =>
      translate(key, values, locale),
    [locale],
  );
}
