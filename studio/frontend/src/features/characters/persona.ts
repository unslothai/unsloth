// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken } from "@/features/auth";
import { loginDisplayName } from "@/features/profile/hooks/use-effective-profile";
import { addressName } from "@/features/profile/utils/address-name";
import { useUserProfileStore } from "@/features/profile/stores/user-profile-store";
import { decodeJwtSubject } from "@/features/profile/utils/jwt-subject";
import { pickPersona } from "./meta";
import type { ChatScene, Persona, ResolvedPersona } from "./types";

function accountNameOf(profile: { nickname: string; displayName: string }): string {
  return addressName(profile, loginDisplayName(decodeJwtSubject(getAuthToken()))) || "User";
}

/** The name the app addresses the account by, read now. */
export function currentAccountName(): string {
  return accountNameOf(useUserProfileStore.getState());
}

export function useAccountName(): string {
  return useUserProfileStore(accountNameOf);
}

/** The account speaking as itself. */
export function profilePersona(accountName = currentAccountName()): ResolvedPersona {
  return { id: null, name: accountName, description: "", hasAvatar: false, avatarRev: 0 };
}

/** The persona a chat speaks as (see pickPersona), else the account. */
export function resolvePersona(
  scene: ChatScene | null,
  personas: Persona[],
  accountName = currentAccountName(),
): ResolvedPersona {
  const chosen = pickPersona(scene, personas);
  if (!chosen) return profilePersona(accountName);
  return {
    id: chosen.id,
    name: chosen.name,
    description: chosen.description,
    hasAvatar: chosen.hasAvatar,
    avatarRev: chosen.avatarRev,
  };
}
