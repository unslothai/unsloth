import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createPlatformApiToken,
  createPlatformLangfuseConfig,
  createPlatformSystemKeyAlias,
  deletePlatformLangfuseConfig,
  getPlatformLangfuseConfig,
  getPlatformOperationsStatus,
  getPlatformUsageStats,
  listPlatformApiTokens,
  listPlatformSystemKeysAlias,
  revokePlatformApiToken,
  revokePlatformSystemKeyAlias,
  updatePlatformLangfuseConfig,
} from "../operations-api";
import { platformTestServer } from "./test-server";

const success = (data: unknown = true) =>
  HttpResponse.json({ code: 0, data, message: "success" });

describe("Rag Platform Phase 9 operations service", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    localStorage.clear();
  });

  afterEach(() => vi.unstubAllEnvs());

  it("normalizes status and stats without retaining dependency error details", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/system/status", () =>
        success({
          doc_engine: {
            type: "elasticsearch",
            status: "red",
            elapsed: "12.5",
            error: "password=must-not-survive",
          },
          storage: { storage: "minio", status: "green", elapsed: "2.1" },
          task_executor_heartbeats: {
            "private-worker-id": [{ secret: "must-not-survive" }],
          },
        }),
      ),
      http.get("http://platform.test/api/v1/system/stats", ({ request }) => {
        const url = new URL(request.url);
        expect(url.searchParams.get("from_date")).toBe("2026-08-01");
        expect(url.searchParams.get("to_date")).toBe("2026-08-16");
        return success({
          pv: [["2026-08-16", 12]],
          uv: [["2026-08-16", 5]],
          speed: [["2026-08-16", 3.5]],
          tokens: [["2026-08-16", 8]],
          round: [["2026-08-16", 4]],
          thumb_up: [["2026-08-16", 2]],
        });
      }),
    );

    const status = await getPlatformOperationsStatus();
    expect(status).toEqual({
      overall: "degraded",
      services: [
        {
          id: "doc_engine",
          label: "doc engine",
          status: "degraded",
          type: "elasticsearch",
          latencyMs: 12.5,
        },
        {
          id: "storage",
          label: "storage",
          status: "healthy",
          type: "minio",
          latencyMs: 2.1,
        },
      ],
      taskExecutorCount: 1,
    });
    expect(JSON.stringify(status)).not.toContain("password");
    expect(JSON.stringify(status)).not.toContain("private-worker-id");

    await expect(
      getPlatformUsageStats({
        fromDate: "2026-08-01",
        toDate: "2026-08-16",
      }),
    ).resolves.toMatchObject({
      pageViews: [{ at: "2026-08-16", value: 12 }],
      tokensThousands: [{ at: "2026-08-16", value: 8 }],
    });
  });

  it("uses exact token CRUD contracts and returns secrets only from create", async () => {
    const methods: string[] = [];
    platformTestServer.use(
      http.get("http://platform.test/api/v1/system/tokens", () =>
        success([
          {
            token: "rag-platform-secret-list-token",
            beta: "compatibility-secret",
            create_time: 1_776_336_000,
          },
        ]),
      ),
      http.post(
        "http://platform.test/api/v1/system/tokens",
        async ({ request }) => {
          methods.push(request.method);
          expect(await request.text()).toBe("");
          return success({
            token: "rag-platform-created-token",
            beta: "created-compatibility-token",
          });
        },
      ),
      http.delete(
        "http://platform.test/api/v1/system/tokens/:key",
        ({ params }) => {
          expect(params.key).toBe("rag-platform-secret-list-token");
          methods.push("DELETE");
          return success(true);
        },
      ),
    );

    const listed = await listPlatformApiTokens();
    expect(listed[0]?.maskedToken).not.toContain("secret-list");
    expect(listed[0]?.revokeKey).toBe("rag-platform-secret-list-token");
    expect(listed[0]?.createdAt).toBe("2026-04-16T10:40:00.000Z");
    await expect(createPlatformApiToken()).resolves.toEqual({
      token: "rag-platform-created-token",
      compatibilityToken: "created-compatibility-token",
    });
    await revokePlatformApiToken("rag-platform-secret-list-token");
    expect(methods).toEqual(["POST", "DELETE"]);
    expect(JSON.stringify(localStorage)).not.toContain(
      "rag-platform-created-token",
    );
  });

  it("keeps the Go /system/keys alias contract covered without a duplicate UI", async () => {
    const calls: string[] = [];
    platformTestServer.use(
      http.get("http://platform.test/api/v1/system/keys", () => {
        calls.push("GET");
        return success([{ token: "alias-secret-token" }]);
      }),
      http.post("http://platform.test/api/v1/system/keys", () => {
        calls.push("POST");
        return success({ token: "new-alias-token", beta: "new-alias-beta" });
      }),
      http.delete("http://platform.test/api/v1/system/keys/:key", () => {
        calls.push("DELETE");
        return success(true);
      }),
    );

    await listPlatformSystemKeysAlias();
    await createPlatformSystemKeyAlias();
    await revokePlatformSystemKeyAlias("alias-secret-token");
    expect(calls).toEqual(["GET", "POST", "DELETE"]);
  });

  it("covers Langfuse GET/POST/PUT/DELETE and redacts returned secrets", async () => {
    const methods: string[] = [];
    platformTestServer.use(
      http.get("http://platform.test/api/v1/langfuse/api-key", () =>
        success({
          host: "https://trace.example.test",
          public_key: "public-key-value",
          secret_key: "server-secret-must-not-survive",
          project_id: "project-1",
          project_name: "Rag Platform",
        }),
      ),
      http.post(
        "http://platform.test/api/v1/langfuse/api-key",
        async ({ request }) => {
          methods.push("POST");
          expect(await request.json()).toEqual({
            host: "https://trace.example.test",
            public_key: "pk-create",
            secret_key: "sk-create",
          });
          return success({
            host: "https://trace.example.test",
            public_key: "pk-create",
            secret_key: "sk-create",
          });
        },
      ),
      http.put(
        "http://platform.test/api/v1/langfuse/api-key",
        async ({ request }) => {
          methods.push("PUT");
          expect(await request.json()).toEqual({
            host: "https://trace.example.test",
            public_key: "pk-update",
            secret_key: "sk-update",
          });
          return success({
            host: "https://trace.example.test",
            public_key: "pk-update",
            secret_key: "sk-update",
          });
        },
      ),
      http.delete("http://platform.test/api/v1/langfuse/api-key", () => {
        methods.push("DELETE");
        return success(true);
      }),
    );

    const config = await getPlatformLangfuseConfig();
    expect(config?.projectName).toBe("Rag Platform");
    expect(JSON.stringify(config)).not.toContain("server-secret");
    const created = await createPlatformLangfuseConfig({
      host: "https://trace.example.test/",
      publicKey: "pk-create",
      secretKey: "sk-create",
    });
    const updated = await updatePlatformLangfuseConfig({
      host: "https://trace.example.test",
      publicKey: "pk-update",
      secretKey: "sk-update",
    });
    await deletePlatformLangfuseConfig();
    expect(JSON.stringify([created, updated])).not.toContain("sk-");
    expect(methods).toEqual(["POST", "PUT", "DELETE"]);
    expect(JSON.stringify(localStorage)).not.toContain("sk-update");
  });
});
