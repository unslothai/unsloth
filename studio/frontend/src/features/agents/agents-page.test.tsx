import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { platformTestServer } from "@/integrations/platform-backend/__tests__/test-server";
import { AgentsPage } from "./agents-page";

const ok = (data: unknown) =>
  HttpResponse.json({ code: 0, data, message: "success" });

describe("Phase 11 Agents page", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    platformTestServer.use(
      http.all("http://platform.test/api/v1/*", ({ request }) => {
        const url = new URL(request.url);
        if (url.pathname === "/api/v1/agents") {
          return ok({
            canvas: [{ id: "agent-1", title: "Destek Agent" }],
            total: 1,
          });
        }
        if (url.pathname === "/api/v1/agents/agent-1") {
          return ok({
            id: "agent-1",
            title: "Destek Agent",
            dsl: { components: {} },
            tags: [],
          });
        }
        if (
          url.pathname.endsWith("/sessions") ||
          url.pathname.endsWith("/versions")
        )
          return ok([]);
        if (url.pathname === "/api/v1/components")
          return ok([{ name: "Begin", category: "flow" }]);
        if (url.pathname === "/api/v1/mcp/servers")
          return ok({ mcp_servers: [], total: 0 });
        if (url.pathname === "/api/v1/plugin/tools") return ok([]);
        return ok({});
      }),
    );
  });

  afterEach(() => vi.unstubAllEnvs());

  it("exposes every agent capability through separate product tabs", async () => {
    render(<AgentsPage />);
    expect(await screen.findByText("Destek Agent")).toBeInTheDocument();
    expect(screen.getByText("Agent ayarları")).toBeInTheDocument();

    for (const name of [
      "Genel",
      "Canvas",
      "Çalıştırma",
      "Oturumlar",
      "Sürümler",
      "Araçlar",
    ]) {
      expect(screen.getByRole("tab", { name })).toBeInTheDocument();
    }
    expect(
      screen.getByRole("button", { name: "Taslağı kaydet" }),
    ).toBeEnabled();
    expect(screen.getByRole("button", { name: "Yayınla" })).toBeEnabled();
    expect(
      screen.getByRole("button", { name: "Durumu sıfırla" }),
    ).toBeEnabled();
    expect(screen.getByRole("button", { name: "Agent’ı sil" })).toBeEnabled();
  });

  it("renders permission failures without leaking backend details", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/agents/agent-1", () =>
        HttpResponse.json(
          { code: 103, message: "internal ownership detail" },
          { status: 403 },
        ),
      ),
    );
    render(<AgentsPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Bu işlem için yetkiniz yok."),
      ).toBeInTheDocument();
    });
    expect(
      screen.queryByText("internal ownership detail"),
    ).not.toBeInTheDocument();
  });
});
