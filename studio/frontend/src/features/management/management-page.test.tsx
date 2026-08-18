import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  user: { email: "user@example.test", superuser: false } as { email: string; superuser: boolean } | null,
  tenants: vi.fn<() => Promise<unknown[]>>(async () => []),
  members: vi.fn<() => Promise<unknown[]>>(async () => []),
}));

vi.mock("@/integrations/platform-backend", () => ({
  PHASE14_OPERATIONS: [],
  acceptPlatformTenantInvite: vi.fn(async () => true),
  executeManagementOperation: vi.fn(async () => true),
  getPlatformAdminDashboard: vi.fn(async () => []),
  getPlatformDifyHealth: vi.fn(async () => ({ status: "ok" })),
  getPlatformUiError: (error: unknown) => ({
    message: error instanceof Error ? error.message : "İstek reddedildi.",
  }),
  invitePlatformTenantMember: vi.fn(async () => true),
  listPlatformChatChannels: vi.fn(async () => []),
  listPlatformCompilationBuiltins: vi.fn(async () => []),
  listPlatformCompilationTemplateGroups: vi.fn(async () => []),
  listPlatformCompilationWikiPresets: vi.fn(async () => []),
  listPlatformTenantMembers: mocks.members,
  listPlatformTenants: mocks.tenants,
  loginPlatformAdmin: vi.fn(async () => ({ email: "admin@example.test", token: "opaque" })),
  logoutPlatformAdmin: vi.fn(async () => undefined),
  parseManagementJson: () => ({}),
  redactManagementData: (value: unknown) => value,
  toManagementRecords: (value: unknown) =>
    Array.isArray(value)
      ? value.map((entry, index) => ({ id: String(index), label: String(index), values: entry }))
      : [],
  usePlatformSessionStore: (selector: (state: { user: typeof mocks.user }) => unknown) =>
    selector({ user: mocks.user }),
}));

import { ManagementPage } from "./management-page";

describe("Phase 14 management product route", () => {
  afterEach(() => {
    mocks.user = { email: "user@example.test", superuser: false };
    mocks.tenants.mockReset().mockResolvedValue([]);
    mocks.members.mockReset().mockResolvedValue([]);
  });

  it("keeps the admin navigation hidden from a non-superuser", () => {
    render(<ManagementPage />);
    expect(screen.queryByRole("button", { name: "Sistem ve admin" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Tenant ve ekip" })).toBeInTheDocument();
  });

  it("requires a separate admin reauthentication and never pre-fills a password", () => {
    mocks.user = { email: "admin@example.test", superuser: true };
    render(<ManagementPage />);
    expect(screen.getByLabelText("E-posta")).toHaveValue("admin@example.test");
    expect(screen.getByLabelText("Parola")).toHaveValue("");
    expect(screen.getByText(/yalnızca bellekte tutulan bir oturum/i)).toBeInTheDocument();
  });

  it("renders loading followed by a tenant empty state", async () => {
    let resolveTenants: ((value: unknown[]) => void) | undefined;
    mocks.tenants.mockImplementation(
      () => new Promise<unknown[]>((resolve) => { resolveTenants = resolve; }),
    );
    render(<ManagementPage />);
    fireEvent.click(screen.getByRole("button", { name: "Tenant ve ekip" }));
    expect(screen.getByText("Yükleniyor…")).toBeInTheDocument();
    resolveTenants?.([]);
    expect(await screen.findByText("Bu kullanıcıya bağlı tenant bulunamadı.")).toBeInTheDocument();
  });

  it("renders tenant permission and request failures as an error state", async () => {
    mocks.tenants.mockRejectedValue(new Error("Bu işlem için yetkiniz yok."));
    render(<ManagementPage />);
    fireEvent.click(screen.getByRole("button", { name: "Tenant ve ekip" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Bu işlem için yetkiniz yok.");
  });
});
