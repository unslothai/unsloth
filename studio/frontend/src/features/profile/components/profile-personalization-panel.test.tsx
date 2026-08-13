import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useUserProfileStore } from "../stores/user-profile-store";
import { ProfilePersonalizationPanel } from "./profile-personalization-panel";

const CREATED_LABEL = /Oluşturuldu/;
const UPDATED_LABEL = /Güncellendi/;

const platform = vi.hoisted(() => ({
  updateProfile: vi.fn(),
  user: {
    active: true,
    avatar: null,
    colorScheme: "",
    createdAt: Date.parse("2024-01-02T03:04:00Z"),
    email: "profile@example.test",
    id: "user-1",
    language: "tr",
    loginChannel: "password",
    nickname: "Backend Profile",
    superuser: false,
    timezone: "Europe/Istanbul",
    updatedAt: Date.parse("2025-05-06T07:08:00Z"),
  },
}));

vi.mock("@/integrations/platform-backend", () => ({
  isPlatformAuthEnabled: () => true,
  updatePlatformProfile: platform.updateProfile,
  usePlatformSessionStore: (
    selector: (state: { user: typeof platform.user }) => unknown,
  ) => selector({ user: platform.user }),
}));

vi.mock("@/features/auth", () => ({
  getAuthToken: () => null,
}));

vi.mock("@/shared/toast", () => ({
  toastError: vi.fn(),
  toastSuccess: vi.fn(),
}));

describe("ProfilePersonalizationPanel platform profile", () => {
  beforeEach(() => {
    useUserProfileStore.setState({
      avatarDataUrl: null,
      displayName: "Backend Profile",
      nickname: "Backend Profile",
    });
    platform.updateProfile.mockResolvedValue({
      active: true,
      avatar: null,
      colorScheme: "",
      createdAt: Date.parse("2024-01-02T03:04:00Z"),
      email: "profile@example.test",
      id: "user-1",
      language: "tr",
      loginChannel: "password",
      nickname: "Updated Profile",
      superuser: false,
      timezone: "Europe/Istanbul",
      updatedAt: Date.parse("2025-05-06T07:08:00Z"),
    });
  });

  it("saves the canonical backend nickname and removes the unsupported duplicate field", async () => {
    const { container } = render(<ProfilePersonalizationPanel />);
    const displayName = container.querySelector<HTMLInputElement>(
      "#profile-display-name",
    );

    expect(displayName).not.toBeNull();
    if (!displayName) {
      throw new Error("Display name input was not rendered");
    }
    expect(displayName).toHaveValue("Backend Profile");
    expect(container.querySelector("#profile-nickname")).toBeNull();
    expect(screen.getByLabelText("E-posta")).toHaveValue(
      "profile@example.test",
    );
    expect(screen.getByLabelText("E-posta")).toHaveAttribute("readonly");
    expect(screen.getByText(CREATED_LABEL)).toBeVisible();
    expect(screen.getByText(UPDATED_LABEL)).toBeVisible();
    expect(container.querySelectorAll("time")).toHaveLength(2);
    expect(screen.queryByText("Hesap bilgileri")).not.toBeInTheDocument();
    expect(screen.queryByText("Aktif çalışma alanı")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Model profilini kaydet"),
    ).not.toBeInTheDocument();

    fireEvent.change(displayName, { target: { value: "Updated Profile" } });
    fireEvent.blur(displayName);

    await waitFor(() =>
      expect(platform.updateProfile).toHaveBeenCalledWith(
        { nickname: "Updated Profile" },
        expect.any(AbortSignal),
      ),
    );
    await waitFor(() =>
      expect(useUserProfileStore.getState()).toMatchObject({
        displayName: "Updated Profile",
        nickname: "Updated Profile",
      }),
    );
  });
});
