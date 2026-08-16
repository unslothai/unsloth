import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  get: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  remove: vi.fn(),
}));

vi.mock("@/integrations/platform-backend", () => ({
  getPlatformLangfuseConfig: mocks.get,
  createPlatformLangfuseConfig: mocks.create,
  updatePlatformLangfuseConfig: mocks.update,
  deletePlatformLangfuseConfig: mocks.remove,
  getPlatformUiError: () => ({ message: "Güvenli hata", retryable: false }),
}));

import { PlatformLangfuseSettings } from "./platform-langfuse-settings";

describe("PlatformLangfuseSettings", () => {
  beforeEach(() => {
    mocks.get.mockResolvedValue(null);
    mocks.create.mockResolvedValue({
      configured: true,
      host: "https://trace.example.test",
      maskedPublicKey: "pk-c••••eate",
      projectId: null,
      projectName: null,
    });
    mocks.remove.mockResolvedValue(undefined);
  });

  it("submits ephemeral credentials and never renders the secret after save", async () => {
    render(<PlatformLangfuseSettings />);
    const host = await screen.findByLabelText("Langfuse adresi");
    fireEvent.change(host, { target: { value: "https://trace.example.test" } });
    fireEvent.change(screen.getByLabelText("Langfuse public key"), {
      target: { value: "pk-create" },
    });
    fireEvent.change(screen.getByLabelText("Langfuse secret key"), {
      target: { value: "sk-ephemeral" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Bağla" }));

    await waitFor(() =>
      expect(mocks.create).toHaveBeenCalledWith(
        {
          host: "https://trace.example.test",
          publicKey: "pk-create",
          secretKey: "sk-ephemeral",
        },
        expect.any(AbortSignal),
      ),
    );
    expect(await screen.findByText("pk-c••••eate")).toBeVisible();
    expect(screen.queryByDisplayValue("sk-ephemeral")).not.toBeInTheDocument();
  });

  it("shows configured metadata and requires confirmation before deletion", async () => {
    mocks.get.mockResolvedValueOnce({
      configured: true,
      host: "https://trace.example.test",
      maskedPublicKey: "pk-r••••only",
      projectId: "project-1",
      projectName: "Rag Platform",
    });
    render(<PlatformLangfuseSettings />);
    expect(await screen.findByText("Rag Platform")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Bağlantıyı kaldır" }));
    expect(mocks.remove).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Kaldır" }));
    await waitFor(() => expect(mocks.remove).toHaveBeenCalledTimes(1));
  });
});
