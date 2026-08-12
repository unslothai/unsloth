import { HttpResponse, http } from "msw";
import { describe, expect, it } from "vitest";

import { platformRequest } from "../client";
import { getPlatformBackendConfig } from "../config";
import {
  getSystemHealth,
  getSystemPing,
  getSystemVersion,
} from "../system-api";
import { platformTestServer } from "./test-server";

describe("system API contracts", () => {
  it("reads the live Python ping, envelope version and raw healthz shapes", async () => {
    expect(getPlatformBackendConfig().apiPrefix).toBe("/api/v1");
    platformTestServer.use(
      http.get("/api/v1/system/ping", () => new HttpResponse("pong")),
      http.get("/api/v1/system/version", () =>
        HttpResponse.json({ code: 0, data: "v0.26.4", message: "success" }),
      ),
      http.get("/api/v1/system/healthz", () =>
        HttpResponse.json({
          db: "ok",
          doc_engine: "ok",
          redis: "ok",
          storage: "ok",
          status: "ok",
        }),
      ),
    );

    await expect(getSystemPing()).resolves.toBe("pong");
    await expect(getSystemVersion()).resolves.toBe("v0.26.4");
    await expect(getSystemHealth()).resolves.toMatchObject({ status: "ok" });
  });

  it("does not expose the raw /system/configs runtime-secret surface", async () => {
    const module = await import("../system-api");
    expect(Object.keys(module).sort()).toEqual([
      "getSystemHealth",
      "getSystemPing",
      "getSystemVersion",
    ]);
  });

  it("keeps the legacy /v1 health shim API-only but contract compatible", async () => {
    platformTestServer.use(
      http.get("/v1/system/healthz", () =>
        HttpResponse.json({ status: "ok", db: "ok", redis: "ok" }),
      ),
    );

    await expect(
      platformRequest<{ status: string }>("/v1/system/healthz", {
        pathMode: "root",
        responseType: "json",
      }),
    ).resolves.toEqual(expect.objectContaining({ status: "ok" }));
  });
});
