// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken } from "@/features/auth";
import { decodeJwtSubject } from "../utils/jwt-subject";
import { useUserProfileStore } from "../stores/user-profile-store";

export function useEffectiveProfile() {
  const displayName = useUserProfileStore((s) => s.displayName);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);

  const sessionSub = decodeJwtSubject(getAuthToken());
  const dn = displayName.trim();
  return {
    sessionSub,
    displayTitle: dn || "Unsloth",
    avatarDataUrl,
  };
}
