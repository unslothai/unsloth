// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect } from "react";
import { LOCALE_OPTIONS, type TranslationKey } from "./messages";
import { syncAutoDomTranslations } from "./auto-dom";
import { translate, useI18nStore } from "./store";

export { LOCALE_OPTIONS } from "./messages";
export type { LocaleCode, TranslationKey } from "./messages";

export function useI18n() {
  const locale = useI18nStore((state) => state.locale);
  const setLocale = useI18nStore((state) => state.setLocale);

  const t = useCallback(
    (key: TranslationKey) => translate(locale, key),
    [locale],
  );

  return {
    locale,
    setLocale,
    t,
    locales: LOCALE_OPTIONS,
  };
}

export function I18nEffects() {
  const locale = useI18nStore((state) => state.locale);

  useEffect(() => {
    document.documentElement.lang = locale;
    syncAutoDomTranslations(locale);
  }, [locale]);

  return null;
}
