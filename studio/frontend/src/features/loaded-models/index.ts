// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The preference module comes FIRST, and the order is load-bearing. The
// indicator imports the settings barrel, which eagerly evaluates SettingsDialog
// and so general-tab, whose top-level reset list dereferences
// LOADED_MODELS_PREFERENCE_KEYS through this same barrel. Export the indicator
// first and an app that reaches this barrel before the settings one enters
// general-tab while this module is still initialising, and startup dies with
// "Cannot access 'LOADED_MODELS_PREFERENCE_KEYS' before initialization".
// Evaluating the constant first makes the cycle harmless whichever side is
// entered first. Verified in Vite dev, which serves native ESM and so has no
// bundler to reorder this.
export {
  LOADED_MODELS_PREFERENCE_KEYS,
  getLoadedModelsDismissed,
  getShowLoadedModels,
  setLoadedModelsDismissed,
  setShowLoadedModels,
  useLoadedModelsDismissed,
  useShowLoadedModels,
} from "./show-loaded-models-pref";
export { LoadedModelsIndicator } from "./loaded-models-indicator";
export type {
  LoadedModelEntry,
  LoadedModelKind,
  LoadedModelSource,
} from "./loaded-models-sources";
