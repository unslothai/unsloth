// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  LOCALES,
  isSupportedLocale,
  setLocale,
  useT,
  useLocale,
} from "@/i18n";

export function LanguageSelect() {
  const t = useT();
  const locale = useLocale();

  return (
    <Select
      value={locale}
      onValueChange={(value) => {
        if (isSupportedLocale(value)) setLocale(value);
      }}
    >
      <SelectTrigger
        aria-label={t("settings.appearance.language.label")}
        className="w-40"
        size="sm"
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {Object.entries(LOCALES).map(([value, metadata]) => (
          <SelectItem key={value} value={value}>
            {metadata.nativeLabel}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
