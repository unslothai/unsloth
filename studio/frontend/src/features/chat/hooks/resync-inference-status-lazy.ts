// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// For callers outside the chat page: a static import pulls the chat runtime into the startup bundle.
export const resyncInferenceStatusAfterServerModelChange = (): Promise<void> =>
  import("./use-chat-model-runtime").then((runtime) =>
    runtime.resyncInferenceStatusAfterServerModelChange(),
  );
