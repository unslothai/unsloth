import { render, screen } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { afterEach, describe, expect, it } from "vitest";

import { PlatformBackendConnectionStatus } from "../backend-connection-status";
import { usePlatformConnectionStore } from "../connection-store";
import { platformTestServer } from "./test-server";

afterEach(() => usePlatformConnectionStore.getState().reset());

describe("PlatformBackendConnectionStatus", () => {
  it("provides a real Settings UI path for connection readiness", async () => {
    let healthRequests = 0;
    platformTestServer.use(
      http.get("/api/v1/system/ping", () => new HttpResponse("pong")),
      http.get("/api/v1/system/version", () =>
        HttpResponse.json({ code: 0, data: "v0.26.4", message: "success" }),
      ),
      http.get("/api/v1/system/healthz", () => {
        healthRequests += 1;
        return HttpResponse.json({ status: "ok", db: "ok", redis: "ok" });
      }),
    );

    const firstRender = render(<PlatformBackendConnectionStatus enabled />);

    expect(
      screen.getByRole("heading", { name: "Rag Platform backend" }),
    ).toBeVisible();
    expect(await screen.findByText("Connected")).toBeVisible();
    expect(screen.getByText("Version v0.26.4")).toBeVisible();

    firstRender.unmount();
    render(<PlatformBackendConnectionStatus enabled />);
    expect(screen.getByText("Connected")).toBeVisible();
    expect(healthRequests).toBe(1);
  });

  it("shows the rollout-disabled empty state without a network call", () => {
    render(<PlatformBackendConnectionStatus enabled={false} />);

    expect(screen.getByText("Disabled")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Check connection" }),
    ).toBeDisabled();
  });

  it("renders a permission error without hiding it as an empty state", async () => {
    platformTestServer.use(
      http.get("/api/v1/system/ping", () =>
        HttpResponse.json(
          { code: 403, data: null, message: "Permission denied" },
          { status: 403 },
        ),
      ),
      http.get("/api/v1/system/version", () =>
        HttpResponse.json({ code: 0, data: "v0.26.4", message: "success" }),
      ),
      http.get("/api/v1/system/healthz", () =>
        HttpResponse.json({ status: "ok" }),
      ),
    );

    render(<PlatformBackendConnectionStatus enabled />);

    expect(await screen.findByText("Permission required")).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent("Permission denied");
  });
});
