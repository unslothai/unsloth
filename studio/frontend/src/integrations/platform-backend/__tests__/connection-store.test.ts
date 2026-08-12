import { act, renderHook } from "@testing-library/react";
import { HttpResponse, delay, http } from "msw";
import { afterEach, describe, expect, it } from "vitest";

import { usePlatformConnectionStore } from "../connection-store";
import { platformTestServer } from "./test-server";

function healthyHandlers() {
  return [
    http.get("/api/v1/system/ping", () => new HttpResponse("pong")),
    http.get("/api/v1/system/version", () =>
      HttpResponse.json({ code: 0, data: "v0.26.4", message: "success" }),
    ),
    http.get("/api/v1/system/healthz", () =>
      HttpResponse.json({ status: "ok", db: "ok", redis: "ok" }),
    ),
  ];
}

afterEach(() => usePlatformConnectionStore.getState().reset());

describe("usePlatformConnectionStore", () => {
  it("moves through loading to connected without persisting data", async () => {
    platformTestServer.use(...healthyHandlers());
    const { result } = renderHook(() => usePlatformConnectionStore());

    let pending: Promise<void>;
    act(() => {
      pending = result.current.checkConnection();
    });
    expect(result.current.status).toBe("checking");
    await act(async () => pending);

    expect(result.current).toMatchObject({
      status: "connected",
      ping: "pong",
      version: "v0.26.4",
      error: null,
    });
    expect(result.current.lastCheckedAt).not.toBeNull();
  });

  it("distinguishes a dependency failure from a disconnected backend", async () => {
    platformTestServer.use(
      http.get("/api/v1/system/ping", () => new HttpResponse("pong")),
      http.get("/api/v1/system/version", () =>
        HttpResponse.json({ code: 0, data: "v0.26.4", message: "success" }),
      ),
      http.get("/api/v1/system/healthz", () =>
        HttpResponse.json({ status: "nok", db: "nok" }, { status: 500 }),
      ),
    );
    const { result } = renderHook(() => usePlatformConnectionStore());

    await act(() => result.current.checkConnection());
    expect(result.current.status).toBe("degraded");
    expect(result.current.error?.kind).toBe("api");
  });

  it("represents permission errors and cleans up an aborted check", async () => {
    platformTestServer.use(
      http.get("/api/v1/system/ping", () =>
        HttpResponse.json(
          { code: 401, data: null, message: "Unauthorized" },
          { status: 401 },
        ),
      ),
      ...healthyHandlers().slice(1),
    );
    const { result } = renderHook(() => usePlatformConnectionStore());
    await act(() => result.current.checkConnection());
    expect(result.current).toMatchObject({
      status: "unauthorized",
      error: { kind: "permission" },
    });

    platformTestServer.use(
      http.get(/\/api\/v1\/system\/.+/, async () => {
        await delay(100);
        return HttpResponse.json({ code: 0, data: true });
      }),
    );
    const controller = new AbortController();
    let pending: Promise<void>;
    act(() => {
      pending = result.current.checkConnection(controller.signal);
    });
    controller.abort();
    await act(async () => pending);
    expect(result.current).toMatchObject({ status: "idle", error: null });
  });
});
