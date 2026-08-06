


import { getAuthToken } from "@/features/auth";
import { decodeJwtSubject } from "../utils/jwt-subject";
import { useUserProfileStore } from "../stores/user-profile-store";

export function useEffectiveProfile() {
  const displayName = useUserProfileStore((s) => s.displayName);
  const nickname = useUserProfileStore((s) => s.nickname);
  const avatarDataUrl = useUserProfileStore((s) => s.avatarDataUrl);

  const sessionSub = decodeJwtSubject(getAuthToken());
  const dn = displayName.trim();
  // Name to address the user by: nickname, else first name, else login id.
  const addressName = nickname.trim() || dn.split(/\s+/)[0] || sessionSub || "";
  return {
    sessionSub,
    displayTitle: dn || "Unsloth",
    addressName,
    avatarDataUrl,
  };
}
