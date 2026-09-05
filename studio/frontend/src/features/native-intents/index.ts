// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { NativeModelChip } from "./components/native-model-chip";
export { NativeModelDropOverlay } from "./components/native-model-drop-overlay";
export {
  consumeNativePathToken,
  openModelsDir,
  pickNativeDocumentFolder,
  pickNativeProjectWorkspace,
  pickHuggingFaceCacheDir,
  readNativeAttachmentFile,
  registerNativeAttachmentPath,
  registerNativeDatasetPath,
} from "./api";
export type {
  NativeDocumentFolderSelection,
  NativeProjectWorkspaceSelection,
} from "./api";
export { nativeFileName } from "./drop-paths";
export { nativeDropTargetAt } from "./native-drop-targets";
export { nativeAttachmentIntentToFile } from "./native-attachment-file";
export { useNativeDropTarget } from "./use-native-drop-target";
export { useNativeFileDrop } from "./use-native-file-drop";
export type {
  NativeFileDrop,
  NativeFileDropOptions,
} from "./use-native-file-drop";
export {
  NativeAttachmentTargetContext,
  useNativeAttachmentTargetKey,
} from "./attachment-target";
export { useNativeIntentStore } from "./store";
export type { NativeIntent } from "./types";
export { useNativeModelDrop } from "./use-native-drop";
export type { NativeModelDropState } from "./use-native-drop";
export { useNativePathLeasesSupported } from "./use-native-readiness";
