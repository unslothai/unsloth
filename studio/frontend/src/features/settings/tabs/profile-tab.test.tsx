import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ProfileTab } from "./profile-tab";

vi.mock("@/features/profile", () => ({
  ProfilePersonalizationPanel: () => <div>Personalization</div>,
  ProfileStatsPanel: () => <div>Profile stats</div>,
}));

vi.mock("@/i18n", () => ({
  useT: () => (key: string) => key,
}));

describe("ProfileTab", () => {
  it("keeps the profile identity and stats panels in a simple order", () => {
    render(<ProfileTab />);

    const personalization = screen.getByText("Personalization");
    const stats = screen.getByText("Profile stats");
    expect(personalization).toBeVisible();
    expect(stats).toBeVisible();
    expect(
      personalization.compareDocumentPosition(stats) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(screen.queryByText("Hesap bilgileri")).not.toBeInTheDocument();
  });
});
