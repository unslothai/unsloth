import { http, HttpResponse } from "msw";
import forge from "node-forge";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  PHASE14_OPERATIONS,
  createPlatformUserPublicToken,
  executeManagementOperation,
  getPlatformAdminDashboard,
  getPlatformDifyHealth,
  getPlatformTenant,
  invitePlatformTenantMember,
  listPlatformChatChannels,
  listPlatformCompilationBuiltins,
  listPlatformCompilationTemplateGroups,
  listPlatformCompilationWikiPresets,
  listPlatformTenantMembers,
  listPlatformTenants,
  listPlatformUserPublicTokens,
  loginPlatformAdmin,
  pollPlatformAimlapiAuthorization,
  removePlatformTenantMember,
  rotatePlatformUserPublicToken,
  startPlatformAimlapiAuthorization,
  updatePlatformTenant,
  updatePlatformTenantMemberRole,
} from "../management-api";
import { redactManagementData } from "../management-types";
import { platformTestServer } from "./test-server";

interface Seen {
  authorization: string | null;
  body: unknown;
  method: string;
  path: string;
  query: string;
}

const ok = (data: unknown) =>
  HttpResponse.json({ code: 0, message: "success", data });

describe("Rag Platform Phase 14 management contracts", () => {
  const seen: Seen[] = [];

  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    seen.length = 0;
    platformTestServer.use(
      http.all("http://platform.test/*", async ({ request }) => {
        const url = new URL(request.url);
        const body = ["GET", "HEAD"].includes(request.method)
          ? null
          : await request.json().catch(() => null);
        seen.push({
          authorization: request.headers.get("authorization"),
          body,
          method: request.method,
          path: url.pathname,
          query: url.search,
        });
        if (url.pathname === "/api/v1/admin/auth") return ok(true);
        if (url.pathname === "/api/v1/admin/users")
          return ok([{ id: "user-1", email: "admin@example.test", access_token: "never-render" }]);
        if (url.pathname === "/api/v1/tenants")
          return ok([{ tenant_id: "tenant-1", name: "Team" }]);
        if (url.pathname === "/api/v1/tenants/tenant-1" && request.method === "GET")
          return ok({ tenant_id: "tenant-1", name: "Team", role: "owner" });
        if (url.pathname.endsWith("/users") && request.method === "GET")
          return ok([{ user_id: "user-1", email: "member@example.test" }]);
        if (url.pathname.includes("/admin/users/") && url.pathname.endsWith("/tokens")) {
          return request.method === "POST"
            ? ok({ tenant_id: "tenant-1", token: "api-secret", beta: "public-secret" })
            : ok([{ tenant_id: "tenant-1", token: "listed-secret", beta: "listed-public" }]);
        }
        if (url.pathname === "/api/v1/chat-channels") return ok([]);
        if (url.pathname === "/api/v1/compilation_template_groups") return ok([]);
        if (url.pathname.includes("/compilation_templates/")) return ok([]);
        if (url.pathname === "/api/v1/dify/retrieval/health") return ok({ status: "ok" });
        if (url.pathname === "/api/v1/llm/aimlapi/authorize/start")
          return ok({ request_id: "request-1", verification_uri: "https://aimlapi.example/authorize?request=request-1", interval: 5, expires_in: 900 });
        if (url.pathname === "/api/v1/llm/aimlapi/authorize/poll")
          return ok({ status: "ready", api_key: "one-time-provider-key" });
        return ok(true);
      }),
    );
  });

  it("uses the user-scoped AIMLAPI device flow without sending a device code", async () => {
    await expect(startPlatformAimlapiAuthorization()).resolves.toMatchObject({
      requestId: "request-1",
      verificationUri: "https://aimlapi.example/authorize?request=request-1",
    });
    await expect(pollPlatformAimlapiAuthorization("request-1")).resolves.toEqual({
      status: "ready",
      apiKey: "one-time-provider-key",
    });
    expect(seen).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/llm/aimlapi/authorize/start",
          body: null,
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/llm/aimlapi/authorize/poll",
          body: { request_id: "request-1" },
        }),
      ]),
    );
    expect(JSON.stringify(seen)).not.toContain("device_code");
  });

  afterEach(() => vi.unstubAllEnvs());

  it("uses a raw, in-memory admin token and loads every dashboard family without exposing secrets", async () => {
    const dashboard = await getPlatformAdminDashboard({ token: "opaque-admin" });
    expect(dashboard.length).toBeGreaterThan(15);
    expect(seen.find((entry) => entry.path === "/api/v1/admin/auth")?.authorization).toBe("opaque-admin");
    expect(seen.find((entry) => entry.path === "/api/v1/admin/users")).toMatchObject({
      query: "?page=1&page_size=50",
    });
    expect(JSON.stringify(dashboard)).not.toContain("never-render");
    expect(JSON.stringify(dashboard)).toContain("••••••••");
  });

  it("extracts the opaque admin login header without converting it to Bearer", async () => {
    const pair = forge.pki.rsa.generateKeyPair({ bits: 1024, e: 0x10001 });
    platformTestServer.use(
      http.post("http://platform.test/api/v1/admin/login", () =>
        new HttpResponse(JSON.stringify({ code: 0, data: true }), {
          headers: { "Content-Type": "application/json", Authorization: "opaque-admin-session" },
        }),
      ),
    );
    await expect(
      loginPlatformAdmin("admin@example.test", "password", {
        publicKeyPem: forge.pki.publicKeyToPem(pair.publicKey),
      }),
    ).resolves.toEqual({ email: "admin@example.test", token: "opaque-admin-session" });
  });

  it("uses exact tenant membership, channel, template and compatibility paths", async () => {
    await listPlatformTenants();
    await listPlatformTenantMembers("tenant-1");
    await getPlatformTenant("tenant-1");
    await updatePlatformTenant("tenant-1", "Updated Team");
    await invitePlatformTenantMember("tenant-1", "member@example.test");
    await removePlatformTenantMember("tenant-1", "user-1");
    await updatePlatformTenantMemberRole("tenant-1", "user-1", "admin");
    await listPlatformChatChannels();
    await listPlatformCompilationTemplateGroups();
    await listPlatformCompilationBuiltins();
    await listPlatformCompilationWikiPresets();
    await getPlatformDifyHealth();
    expect(seen).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ method: "GET", path: "/api/v1/tenants" }),
        expect.objectContaining({ method: "GET", path: "/api/v1/tenants/tenant-1" }),
        expect.objectContaining({ method: "PUT", path: "/api/v1/tenants/tenant-1", body: { name: "Updated Team" } }),
        expect.objectContaining({ method: "POST", path: "/api/v1/tenants/tenant-1/users", body: { email: "member@example.test" } }),
        expect.objectContaining({ method: "DELETE", path: "/api/v1/tenants/tenant-1/users", body: { user_id: "user-1" } }),
        expect.objectContaining({ method: "PUT", path: "/api/v1/tenants/tenant-1/users/user-1/role", body: { role: "admin" } }),
        expect.objectContaining({ path: "/api/v1/chat-channels" }),
        expect.objectContaining({ path: "/api/v1/compilation_template_groups" }),
        expect.objectContaining({ path: "/api/v1/compilation_templates/builtins" }),
        expect.objectContaining({ path: "/api/v1/compilation_templates/wiki_presets" }),
        expect.objectContaining({ path: "/api/v1/dify/retrieval/health", authorization: null }),
      ]),
    );
  });

  it("creates, redacts, rotates and revokes the paired API/public token contract", async () => {
    const created = await createPlatformUserPublicToken(
      "opaque-admin",
      "admin@example.test",
    );
    expect(created).toEqual({ tenantId: "tenant-1", token: "api-secret", beta: "public-secret" });

    const listed = await listPlatformUserPublicTokens(
      "opaque-admin",
      "admin@example.test",
    );
    expect(JSON.stringify(listed)).not.toContain("listed-secret");
    expect(JSON.stringify(listed)).toContain("••••••••");

    await rotatePlatformUserPublicToken(
      "opaque-admin",
      "admin@example.test",
      "old-api-token",
    );
    const encodedUsername = encodeURIComponent(btoa("admin@example.test"));
    expect(seen).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          authorization: "opaque-admin",
          method: "POST",
          path: `/api/v1/admin/users/${encodedUsername}/tokens`,
        }),
        expect.objectContaining({
          authorization: "opaque-admin",
          method: "DELETE",
          path: `/api/v1/admin/users/${encodedUsername}/tokens/old-api-token`,
        }),
      ]),
    );
  });

  it("requires an audit reason and explicit caller confirmation metadata for dangerous operations", async () => {
    const operation = PHASE14_OPERATIONS.find((entry) => entry.id === "admin-service-stop");
    expect(operation).toBeDefined();
    await expect(
      executeManagementOperation(operation!, {
        adminToken: "opaque-admin",
        pathParameters: { service_id: "ingestor" },
      }),
    ).rejects.toThrow(/denetim gerekçesi/i);
    await executeManagementOperation(operation!, {
      adminToken: "opaque-admin",
      auditReason: "INC-123 kontrollü bakım",
      pathParameters: { service_id: "ingestor" },
    });
    expect(seen.at(-1)).toMatchObject({
      authorization: "opaque-admin",
      method: "DELETE",
      path: "/api/v1/admin/services/ingestor",
      body: { audit_reason: "INC-123 kontrollü bakım" },
    });
  });

  it("redacts nested credentials without mutating the input", () => {
    const source = {
      provider: { api_key: "key-1", nested: [{ access_token: "token-1" }] },
      status: "ok",
    };
    expect(redactManagementData(source)).toEqual({
      provider: { api_key: "••••••••", nested: [{ access_token: "••••••••" }] },
      status: "ok",
    });
    expect(source.provider.api_key).toBe("key-1");
  });

  it("propagates backend cross-tenant permission denials and supports abort cleanup", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/tenants/other/users", () =>
        HttpResponse.json({ code: 403, message: "cross-tenant denied", data: null }, { status: 403 }),
      ),
    );
    await expect(invitePlatformTenantMember("other", "member@example.test")).rejects.toMatchObject({
      httpStatus: 403,
    });

    const controller = new AbortController();
    controller.abort();
    await expect(listPlatformTenants(controller.signal)).rejects.toMatchObject({
      code: "CLIENT_ABORTED",
    });
  });

  it("exposes the source-backed ingestor shutdown operation with danger gates", () => {
    expect(PHASE14_OPERATIONS.length).toBeGreaterThan(100);
    const shutdown =
      PHASE14_OPERATIONS.find(
        (operation) => operation.method === "DELETE" && operation.endpoint === "/admin/ingestors",
      );
    expect(shutdown).toMatchObject({ danger: true, needsAdminToken: true, requiresAuditReason: true });
  });
});
