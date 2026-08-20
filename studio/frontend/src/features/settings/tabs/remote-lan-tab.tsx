// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { LanAccessSection } from "../components/lan-access-section";
import { RemoteAccessSection } from "../components/remote-access-section";

/** Reaching Studio from another device, without the API token list in the way. */
export function RemoteLanTab() {
  const t = useT();

  return (
    <div className="flex min-w-0 max-w-full flex-col gap-6">
      <header className="flex min-w-0 flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.remoteLan.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.remoteLan.description")}
        </p>
      </header>

      <RemoteAccessSection />

      <LanAccessSection />
    </div>
  );
}
