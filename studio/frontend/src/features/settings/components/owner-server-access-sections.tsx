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
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchCurrentAccount()
      .then((account) => {
        if (!cancelled) setIsAdmin(account.is_admin);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(
            cause instanceof Error ? cause.message : "Failed to load account role",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div role="alert" className="rounded-lg border border-destructive/25 bg-destructive/5 p-3 text-sm text-destructive">
        {error}
      </div>
    );
  }
  if (isAdmin === null) {
    return <div className="h-28 animate-pulse rounded-xl bg-muted/40" />;
  }
  if (!isAdmin) {
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
