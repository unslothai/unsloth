// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken, OWNER_USERNAME } from "@/features/auth";
import { decodeJwtSubject } from "../utils/jwt-subject";
import { useUserProfileStore } from "../stores/user-profile-store";

// The owner's id is the literal "unsloth"; shared so every login-id fallback agrees.
export function loginDisplayName(sessionSub: string | null): string {
  return sessionSub === OWNER_USERNAME ? "Unsloth" : (sessionSub ?? "");
}

export function useEffectiveProfile() {
  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);

  const sessionSub = decodeJwtSubject(getAuthToken());
  const dn = displayName.trim();
  const login = loginDisplayName(sessionSub);
  const addressName = nickname.trim() || dn.split(/\s+/)[0] || login;
  return {
    sessionSub,
    displayTitle: dn || login || "Unsloth",
    addressName,
    avatarDataUrl,
  };
}
