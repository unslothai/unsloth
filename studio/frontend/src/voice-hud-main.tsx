// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Entry for the voice-typing overlay window (voice-hud.html): the same bundle also
// serves the main window, but this page only ever mounts the small indicator.

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { VoiceTypingHud } from "./features/voice-typing/voice-typing-hud";
import { initializeLocale } from "./i18n";

void initializeLocale();

const container = document.getElementById("root");
if (container) {
  createRoot(container).render(
    <StrictMode>
      <VoiceTypingHud />
    </StrictMode>,
  );
}
