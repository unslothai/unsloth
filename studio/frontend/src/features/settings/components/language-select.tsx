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
  // Show what is in effect whenever the preference's own catalog failed and a
  // fallback was adopted: a controlled Select never fires onValueChange for the
  // value it already holds, so naming the failed preference would leave no way
  // to retry it. This has to come from the store rather than a
  // `preference !== locale` comparison, because a failed `auto` still reads as
  // `auto`, and its fallback locale is indistinguishable from a successful
  // detection that landed on the same language.
  const activePreference = catalogFailed ? locale : preference;

  return (
    <Select
      value={pendingPreference ?? activePreference}
      onValueChange={(value) => {
        if (isLocalePreference(value)) setLocale(value);
      }}
    >
      <SelectTrigger
        aria-label={t("settings.appearance.language.label")}
        aria-busy={pendingPreference !== null}
        className="w-40"
        size="sm"
      >
        <SelectValue />
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
