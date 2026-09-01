// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import { fetchCurrentAccount } from "../api/accounts";
import { KeylessApiAccessSection } from "./keyless-api-access-section";
import { LanAccessSection } from "./lan-access-section";
import { RemoteAccessSection } from "./remote-access-section";

export function OwnerServerAccessSections({
  onSettingsChange,
}: {
  onSettingsChange?: (settings: {
    scope: KeylessApiAccessScope;
    tools: boolean;
    exposure: KeylessApiAccessExposure | null;
  }) => void;
}) {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);

  // Refetched per mount rather than memoised at module scope. Signing out is an
  // SPA navigation, not a reload, so a cached role outlives the account it was
  // read for and the next account sees the wrong panel until a hard refresh.
  // One /api/auth/me per tab switch is the cheaper side of that trade.
  useEffect(() => {
    let cancelled = false;
    fetchCurrentAccount()
      .then((account) => {
        if (!cancelled) setIsAdmin(account.is_admin);
      })
      .catch((cause: unknown) => {
        // Not surfaced: a role lookup that fails must not take the whole panel
        // down with it. This component only decides what to draw, and every
        // route behind these controls is independently gated by
        // _require_install_admin, so drawing them is not an escalation.
        console.warn("Could not read the account role for server access", cause);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Draw for the owner until told otherwise, rather than holding a skeleton on
  // every first paint. The single-account install is the overwhelmingly common
  // case and this panel is presentation only, so showing the controls a moment
  // early (or when the role is unknown) grants a managed account nothing.
  if (isAdmin === false) {
    return (
      <div className="rounded-xl border bg-card p-4 text-sm">
        Only the installation owner can change Remote, LAN, or keyless server access.
      </div>
    );
  }

  return (
    <>
      <RemoteAccessSection />
      <LanAccessSection />
      <KeylessApiAccessSection onSettingsChange={onSettingsChange} />
    </>
  );
}
