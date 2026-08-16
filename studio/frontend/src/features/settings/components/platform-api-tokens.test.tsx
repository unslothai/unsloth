import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  create: vi.fn(),
  revoke: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", () => ({
  listPlatformApiTokens: mocks.list,
  createPlatformApiToken: mocks.create,
  revokePlatformApiToken: mocks.revoke,
  getPlatformUiError: () => ({ message: "Güvenli hata", retryable: false }),
}));

import { PlatformApiTokens } from "./platform-api-tokens";

describe("PlatformApiTokens", () => {
  beforeEach(() => {
    mocks.list.mockResolvedValue([
      {
        id: "token-1",
        label: "API token 1",
        maskedToken: "rag-••••oken",
        createdAt: "2026-08-16T10:00:00Z",
        revokeKey: "raw-list-secret-token",
      },
    ]);
    mocks.create.mockResolvedValue({
      token: "raw-created-token",
      compatibilityToken: "raw-created-compatibility-token",
    });
    mocks.revoke.mockResolvedValue(undefined);
  });

  it("masks listed tokens, reveals a created token once, and revokes by the hidden key", async () => {
    render(<PlatformApiTokens />);

    expect(await screen.findByText("rag-••••oken")).toBeVisible();
    expect(screen.queryByText("raw-list-secret-token")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Token oluştur" }));
    expect(await screen.findByText("raw-created-token")).toBeVisible();
    expect(screen.getByText("raw-created-compatibility-token")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Bitti" }));
    expect(screen.queryByText("raw-created-token")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "API token 1 token'ını iptal et" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Token'ı iptal et" }));
    await waitFor(() =>
      expect(mocks.revoke).toHaveBeenCalledWith(
        "raw-list-secret-token",
        expect.any(AbortSignal),
      ),
    );
  });

  it("distinguishes a load failure from an empty token list", async () => {
    mocks.list.mockRejectedValueOnce(new Error("network"));
    render(<PlatformApiTokens />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Güvenli hata");
    expect(
      screen.queryByText("Henüz API token'ı yok."),
    ).not.toBeInTheDocument();
  });
});
