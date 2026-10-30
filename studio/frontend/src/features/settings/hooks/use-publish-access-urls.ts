// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useEffect } from "react";
import { loadLanAccess, publishLanUrls } from "../api/lan-access";
import { loadRemoteAccess } from "../api/remote-access";
import { remoteAccessPollDelay } from "../api/remote-access-state";

/**
 * Publishes the tunnel and LAN URLs the API examples build on, without the Remote & LAN panels.
 */
export function usePublishAccessUrls(enabled: boolean) {
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let timer: number | undefined;
    let settling = false;
    // Re-read only mid-transition: a Start left on another tab publishes its URL once the tunnel is up.
    // A failed read while settling retries too, or the last transitional state would stick.
    const readRemote = () => {
      loadRemoteAccess()
        .then((next) => {
          if (cancelled) return;
          usePlatformStore.setState({ cloudflareUrl: next.url });
          settling = next.state === "starting" || next.state === "stopping";
          if (settling) {
            timer = window.setTimeout(readRemote, remoteAccessPollDelay(next));
          }
        })
        .catch(() => {
          if (!cancelled && settling) {
            timer = window.setTimeout(readRemote, remoteAccessPollDelay(null));
          }
        });
    };
    readRemote();
    loadLanAccess()
      .then((next) => {
        if (!cancelled) publishLanUrls(next);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [enabled]);
}
