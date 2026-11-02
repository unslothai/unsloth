// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { useEffect, useRef } from "react";
import { KeylessApiAccessSection } from "../components/keyless-api-access-section";
import { LanAccessSection } from "../components/lan-access-section";
import { RemoteAccessSection } from "../components/remote-access-section";
import { scrollIntoPane } from "../lib/scroll-into-pane";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

export function RemoteLanTab() {
  const t = useT();
  const keylessRef = useRef<HTMLDivElement>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const consumeScrollTarget = useSettingsDialogStore((s) => s.consumeScrollTarget);

  // Again once the sections above have loaded: their rows arrive late and push it back down.
  useEffect(() => {
    if (scrollTarget !== "remote-lan-keyless") return;
    const land = () => {
      if (keylessRef.current) scrollIntoPane(keylessRef.current);
    };
    const frame = window.requestAnimationFrame(land);
    // Consumed here, not on the first frame: clearing the target re-runs this effect and its cleanup.
    const settle = window.setTimeout(() => {
      land();
      consumeScrollTarget("remote-lan-keyless");
    }, 600);
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(settle);
    };
  }, [consumeScrollTarget, scrollTarget]);

  return (
    <div className="settings-page">
      <header className="flex min-w-0 flex-col gap-1">
        <h1
          data-settings-label={t("settings.remoteLan.title")}
          className="text-xl font-semibold font-heading"
        >
          {t("settings.remoteLan.title")}
        </h1>
        <p
          data-settings-label={t("settings.remoteLan.description")}
          className="text-xs text-muted-foreground"
        >
          {t("settings.remoteLan.description")}
        </p>
      </header>

      <RemoteAccessSection />

      <LanAccessSection />

      <div ref={keylessRef}>
        <KeylessApiAccessSection />
      </div>
    </div>
  );
}
