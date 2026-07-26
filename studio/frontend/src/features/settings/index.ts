// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { SettingsDialog } from "./settings-dialog";
export { loadEmbeddingModelSettings } from "./api/embedding-model";
export {
  loadPersonalization,
  savePersonalization,
} from "./api/personalization";
export {
  isPalette,
  setPalette,
  setTheme,
  usePalette,
  useTheme,
} from "./stores/theme-store";
export {
  DEFAULT_CUSTOMIZATION,
  applyCustomizationToDocument,
  isDefaultCustomization,
  prefersReducedMotion,
  sanitizeCustomization,
  useAppearanceCustomStore,
} from "./stores/appearance-custom-store";
export type {
  AppearanceCustomization,
  CustomModeColors,
  ReduceMotionSetting,
} from "./stores/appearance-custom-store";
export { useMonitorOverlayStore } from "./stores/monitor-overlay-store";
export type {
  Personalization,
  PersonalizationAppearance,
  PersonalizationProfile,
} from "./api/personalization";
export { useSettingsDialogStore } from "./stores/settings-dialog-store";
export type { SettingsTab } from "./stores/settings-dialog-store";
export { SettingsRow } from "./components/settings-row";
export { SettingsSection } from "./components/settings-section";

export type { Palette, ResolvedTheme, Theme } from "./stores/theme-store";
