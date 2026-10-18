// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mounts exactly one voice-typing controller for the lifetime of the desktop app.

import { useEffect } from "react";
import { isTauri } from "@/lib/api-base";
import { VoiceTypingController } from "./voice-typing-controller";

export function useVoiceTyping(): void {
  useEffect(() => {
    if (!isTauri) return;
    const controller = new VoiceTypingController();
    void controller.init();
    return () => controller.dispose();
  }, []);
}
