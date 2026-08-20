


import { useCallback } from "react";
import { useLocale } from "./locale-store";
import { translate } from "./messages";
import type { TranslationKey } from "./messages";
import type { InterpolationValues } from "./types";

export {
  AUTO_LOCALE,
  DEFAULT_LOCALE,
  DEFAULT_LOCALE_PREFERENCE,
  LOCALE_STORAGE_KEY,
  LOCALE_INITIALIZATION_TIMEOUT_MS,
  LOCALE_SELECTION_TIMEOUT_MS,
  getLocale,
  getLocaleCatalogFailed,
  getLocalePreference,
  getPendingLocalePreference,
  initializeLocale,
  isLocalePreference,
  setLocale,
  subscribeLocale,
  useLocale,
  useLocaleCatalogFailed,
  useLocalePreference,
  usePendingLocalePreference,
} from "./locale-store";
export type {
  LocaleChangeResult,
  LocalePreference,
  SetLocaleOptions,
} from "./locale-store";
export {
  LOCALES,
  isSupportedLocale,
  messages,
  translate,
} from "./messages";
export type { Locale, TranslationKey } from "./messages";
export { formatRelativeTime } from "./relative-time";
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
