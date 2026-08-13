import { useEffect } from "react";

import { usePlatformSessionStore } from "@/integrations/platform-backend";
import { useUserProfileStore } from "../stores/user-profile-store";

/**
 * Mirrors the authenticated Rag Platform identity into the app-wide profile
 * store used by the sidebar, chat greeting and Profile personalization panel.
 * Mutations still go through the typed platform profile service; this hook is
 * deliberately one-way so local persistence can never overwrite the backend.
 */
export function usePlatformProfileSync(enabled: boolean): void {
  const user = usePlatformSessionStore((state) => state.user);

  useEffect(() => {
    if (!enabled) return;

    useUserProfileStore.setState({
      avatarDataUrl: user?.avatar ?? null,
      displayName: user?.nickname ?? "",
      nickname: user?.nickname ?? "",
    });
  }, [enabled, user]);
}
