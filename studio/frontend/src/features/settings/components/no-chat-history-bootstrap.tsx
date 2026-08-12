// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat";
import { useEffect } from "react";
import {
  loadNoChatHistorySettings,
  subscribeNoChatHistorySettings,
} from "../api/no-chat-history";

function applyNoChatHistoryPolicy(enabled: boolean) {
  const store = useChatRuntimeStore.getState();
  store.setIncognitoLocked(enabled);
  if (enabled) {
    store.setIncognito(true);
  }
}

export function NoChatHistoryBootstrap() {
  useEffect(() => {
    let active = true;
    void loadNoChatHistorySettings()
      .then((settings) => {
        if (!active) return;
        applyNoChatHistoryPolicy(settings.enabled);
      })
      .catch(() => {
        // Fail open: a settings outage must not block chat.
      });
    const unsubscribe = subscribeNoChatHistorySettings((settings) => {
      if (!active) return;
      applyNoChatHistoryPolicy(settings.enabled);
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, []);
  return null;
}
