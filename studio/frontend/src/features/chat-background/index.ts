// SPDX-License-Identifier: AGPL-3.0-only
export { ChatBackground } from "./mount";
export {
  WindowMaterialMount,
  nativeMaterialAvailable,
  useMaterialStore,
} from "./material";
export { MATERIAL_LABELS } from "./material-labels";
export { MATERIALS, type WindowMaterial } from "./model";
export { useChatWelcomeShown, useMarkChatWelcome } from "./welcome";
export { resetChatBackground, useBackgroundStore } from "./storage";
