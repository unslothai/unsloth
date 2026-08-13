import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { usePlatformSessionStore } from "@/integrations/platform-backend";
import { useUserProfileStore } from "../stores/user-profile-store";
import { usePlatformProfileSync } from "./use-platform-profile-sync";

describe("usePlatformProfileSync", () => {
  afterEach(() => {
    usePlatformSessionStore.getState().reset();
    useUserProfileStore.setState({
      avatarDataUrl: null,
      displayName: "",
      nickname: "",
    });
  });

  it("hydrates the app profile from the authenticated backend user", () => {
    usePlatformSessionStore.getState().setUser({
      active: true,
      avatar: "data:image/png;base64,backend-avatar",
      colorScheme: "",
      createdAt: 1_786_502_416_241,
      email: "profile@example.test",
      id: "user-1",
      language: "tr",
      loginChannel: "password",
      nickname: "Backend Profile",
      superuser: false,
      timezone: "Europe/Istanbul",
      updatedAt: 1_786_502_416_398,
    });

    renderHook(() => usePlatformProfileSync(true));

    expect(useUserProfileStore.getState()).toMatchObject({
      avatarDataUrl: "data:image/png;base64,backend-avatar",
      displayName: "Backend Profile",
      nickname: "Backend Profile",
    });
  });
});
