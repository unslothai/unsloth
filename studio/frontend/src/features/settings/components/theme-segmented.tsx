// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PillSegmented } from "@/components/pill-segmented";
import { type TranslationKey, useT } from "@/i18n";
import { LaptopIcon, Moon02Icon, Sun02Icon } from "@hugeicons/core-free-icons";
import { type Theme, useTheme } from "../stores/theme-store";

const OPTIONS: {
  value: Theme;
  labelKey: TranslationKey;
  icon: typeof Sun02Icon;
}[] = [
  {
    value: "light",
    labelKey: "settings.appearance.theme.light",
    icon: Sun02Icon,
  },
  {
    value: "dark",
    labelKey: "settings.appearance.theme.dark",
    icon: Moon02Icon,
  },
  {
    value: "system",
    labelKey: "settings.appearance.theme.system",
    icon: LaptopIcon,
  },
];

export function ThemeSegmented() {
  const t = useT();
  const { theme, setTheme } = useTheme();
  return (
    <PillSegmented
      value={theme}
      options={OPTIONS.map((option) => ({
        value: option.value,
        label: t(option.labelKey),
        icon: option.icon,
      }))}
      onChange={setTheme}
      ariaLabel={t("settings.appearance.theme.label")}
    />
  );
}
