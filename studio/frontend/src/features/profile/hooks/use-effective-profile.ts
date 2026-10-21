// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken, OWNER_USERNAME } from "@/features/auth";
import { decodeJwtSubject } from "../utils/jwt-subject";
import { useUserProfileStore } from "../stores/user-profile-store";

// The owner's id is the reserved literal "unsloth", so a verbatim subject spells
// the brand lower case. Only that id maps; a chosen username stays as chosen.
// Shared, not copied: every surface falling back to the login id must agree.
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
  // Name to address the user by: nickname, else first name, else login id.
  const addressName = nickname.trim() || dn.split(/\s+/)[0] || login;
  return {
    sessionSub,
    displayTitle: dn || login || "Unsloth",
    addressName,
    avatarDataUrl,
  };
}
