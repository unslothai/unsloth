// SPDX-License-Identifier: AGPL-3.0-only
export { ChatBackground } from "./mount";
export {
  WindowMaterialMount,
  nativeMaterialAvailable,
  useMaterialStore,
} from "./material";
export { MATERIAL_LABELS } from "./material-labels";
export { MATERIALS, sanitizeAdjustments, type WindowMaterial } from "./model";
export {
  useChatWelcomeShown,
  useMarkChatCompare,
  useMarkChatWelcome,
} from "./welcome";
export {
  beginBackgroundUpload,
  finishBackgroundUpload,
  hydrateBackground,
  resetChatBackground,
  useBackgroundStore,
} from "./storage";
export { redrawImage } from "./image";
export { uploadBackgroundImage } from "./upload";
