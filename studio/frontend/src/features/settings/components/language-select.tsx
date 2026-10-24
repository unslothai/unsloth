// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import {
  AUTO_LOCALE,
  LOCALES,
  isLocalePreference,
  setLocale,
  useLocale,
  useLocaleCatalogFailed,
  useLocalePreference,
  usePendingLocalePreference,
  useT,
} from "@/i18n";

export function LanguageSelect() {
  const t = useT();
  const locale = useLocale();
  const preference = useLocalePreference();
  const pendingPreference = usePendingLocalePreference();
  const catalogFailed = useLocaleCatalogFailed();
  // After a catalog failure name no entry (Radix ""), since a controlled Select cannot re-pick its own
  // value and both the failed preference and the fallback must stay pickable.
  const fallbackLabel = catalogFailed ? LOCALES[locale].nativeLabel : "";

  return (
    <Select
      value={pendingPreference ?? (catalogFailed ? "" : preference)}
      onValueChange={(value) => {
        if (isLocalePreference(value)) setLocale(value);
      }}
    >
      <SelectTrigger
        aria-label={t("settings.appearance.language.label")}
        aria-busy={pendingPreference !== null}
        className="w-40 data-[placeholder]:text-foreground"
        size="sm"
      >
        <SelectValue placeholder={fallbackLabel} />
        {pendingPreference !== null ? <Spinner className="size-3.5" /> : null}
      </SelectTrigger>
      <SelectContent
        style={{
          maxHeight: "min(18rem, var(--radix-select-content-available-height))",
        }}
      >
        <SelectItem value={AUTO_LOCALE}>
          {t("settings.appearance.language.autoDetect")}
        </SelectItem>
        {Object.entries(LOCALES).map(([value, metadata]) => (
          <SelectItem key={value} value={value}>
            {metadata.nativeLabel}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
