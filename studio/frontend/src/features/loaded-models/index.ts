// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Preference module first: the settings barrel cycles back here, else a TDZ error on its keys.
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
