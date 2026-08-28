// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { SettingsDialog } from "./settings-dialog";
export {
  type DownloadTransportMode,
  type DownloadTransportSettings,
  loadDownloadTransportSettings,
  subscribeDownloadTransportSettings,
  updateDownloadTransportSettings,
} from "./api/download-transport";
export { loadEmbeddingModelSettings } from "./api/embedding-model";
export { loadOpenAIAutoSwitchSettings } from "./api/openai-auto-switch";
export {
  loadHuggingFaceCacheSettings,
  updateHuggingFaceCacheSettings,
} from "./api/hugging-face-cache";
export type { HuggingFaceCacheSettings } from "./api/hugging-face-cache";
export {
  formatUploadSize,
  getCachedUploadLimitBytes,
  getCachedUploadLimitLabel,
  loadUploadLimitSettings,
  subscribeUploadLimitSettings,
} from "./api/upload-limit";
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
  migrateShippedSidebarNavDefault,
  prefersReducedMotion,
  sanitizeCustomization,
  useAppearanceCustomStore,
} from "./stores/appearance-custom-store";
export type {
  AppearanceCustomization,
  CustomModeColors,
  ReduceMotionSetting,
  SidebarNavItemId,
  SidebarNavItemPref,
} from "./stores/appearance-custom-store";
export { useMonitorOverlayStore } from "./stores/monitor-overlay-store";
export {
  applyInterfaceScale,
  useInterfaceScaleStore,
} from "./stores/interface-scale-store";
// The runtime module, not the store, so consumers outside this feature do not have to pull
// zustand in with them. native-drop-position.ts imports it directly for that reason.
export {
  NATIVE_MAC_TITLEBAR_HEIGHT_VAR,
  NATIVE_MAC_TRAFFIC_LIGHT_INSET_VAR,
} from "./lib/interface-scale-runtime";
export {
  type MonitorFrame,
  useMonitorFrameStore,
} from "./stores/monitor-frame-store";
export type {
  Personalization,
  PersonalizationAppearance,
  PersonalizationProfile,
} from "./api/personalization";
export {
  COMPOSER_INPUT_SELECTOR,
  isSurfaceBackgrounded,
  isSurfaceInForeground,
  useShortcut,
  useShortcutLabel,
  useShortcutLabels,
} from "./hooks/use-shortcut";
export { Shortcut } from "./components/shortcut";
export {
  currentBinding,
  useKeyboardShortcutsStore,
} from "./stores/keyboard-shortcuts-store";
export type { ShortcutId } from "./lib/keyboard-shortcuts";
export { useSettingsDialogStore } from "./stores/settings-dialog-store";
export type { SettingsTab } from "./stores/settings-dialog-store";
export type { Palette, ResolvedTheme, Theme } from "./stores/theme-store";
