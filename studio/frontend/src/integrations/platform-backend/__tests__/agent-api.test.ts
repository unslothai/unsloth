import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../agent-api";
import { EMPTY_PLATFORM_AGENT_DSL, redactAgentSecrets } from "../agent-types";
import { platformTestServer } from "./test-server";

interface Seen {
  method: string;
  path: string;
  query: string;
  body: unknown;
}

const ok = (data: unknown) =>
  HttpResponse.json({ code: 0, data, message: "success" });

describe("Rag Platform Phase 11 agent contracts", () => {
  const seen: Seen[] = [];

  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    seen.length = 0;
    platformTestServer.use(
      http.all("http://platform.test/api/v1/*", async ({ request }) => {
        const url = new URL(request.url);
        const contentType = request.headers.get("content-type") ?? "";
        const body = ["GET", "HEAD"].includes(request.method)
          ? null
          : contentType.includes("multipart/form-data")
            ? "multipart"
            : await request.json().catch(() => null);
        seen.push({
          method: request.method,
          path: url.pathname,
          query: url.search,
          body,
        });

        if (url.pathname === "/api/v1/agents" && request.method === "GET") {
          return ok({ canvas: [{ id: "agent-1", title: "Demo" }], total: 1 });
        }
        if (url.pathname.endsWith("/sessions") && request.method === "GET") {
          return ok([{ id: "session-1", name: "Oturum" }]);
        }
        if (url.pathname.endsWith("/versions") && request.method === "GET") {
          return ok([{ id: "version-1" }]);
        }
        if (url.pathname === "/api/v1/components")
          return ok([{ name: "Begin" }]);
        if (
          url.pathname === "/api/v1/mcp/servers" &&
          request.method === "GET"
        ) {
          return ok({
            mcp_servers: [
              {
                id: "mcp-1",
                name: "Tools",
                url: "https://mcp.test",
                server_type: "sse",
              },
            ],
            total: 1,
          });
        }
        if (url.pathname === "/api/v1/plugin/tools")
          return ok([{ name: "search" }]);
        if (
          url.pathname.includes("/attachments/") ||
          url.pathname === "/api/v1/agents/download"
        ) {
          return new HttpResponse(new Uint8Array([1, 2]), {
            headers: { "content-type": "application/octet-stream" },
          });
        }
        return ok({ id: url.pathname.includes("mcp") ? "mcp-1" : "agent-1" });
      }),
    );
  });

  afterEach(() => vi.unstubAllEnvs());

  it("redacts nested database and MCP credentials from diagnostic output", () => {
    expect(
      redactAgentSecrets({
        password: "db-secret",
        headers: { Authorization: "Bearer token" },
        nested: { api_key: "provider-key", safe: "visible" },
      }),
    ).toEqual({
      password: "<redacted>",
      headers: "<redacted>",
      nested: { api_key: "<redacted>", safe: "visible" },
    });
  });

  it("uses exact CRUD, lifecycle, component, session, version, file and webhook contracts", async () => {
    await api.listAgents({ keywords: "demo" });
    await api.createAgent({ title: "Demo", dsl: EMPTY_PLATFORM_AGENT_DSL });
    await api.getAgent("agent-1");
    await api.updateAgent("agent-1", {
      title: "Yeni",
      dsl: EMPTY_PLATFORM_AGENT_DSL,
    });
    await api.updateAgentTags("agent-1", ["prod"]);
    await api.publishAgent("agent-1", { dsl: EMPTY_PLATFORM_AGENT_DSL });
    await api.resetAgent("agent-1");
    await api.cancelAgentRun("agent-1");
    await api.cancelAgentSession("session-1");
    await api.getAgentComponentInputForm("agent-1", "begin");
    await api.debugAgentComponent("agent-1", "begin", {
      query: { value: "hello" },
    });
    await api.listAgentSessions("agent-1");
    await api.createAgentSession("agent-1", "Oturum");
    await api.getAgentSession("agent-1", "session-1");
    await api.deleteAgentSession("agent-1", "session-1");
    await api.deleteAgentSessions("agent-1", { ids: ["one", "two"] });
    await api.deleteAgentSessions("agent-1", { deleteAll: true });
    await api.listAgentVersions("agent-1");
    await api.getAgentVersion("agent-1", "version-1");
    await api.deleteAgentVersion("agent-1", "version-1");
    await api.getAgentLogs("agent-1", "message-1");
    await api.getAgentWebhookLogs("agent-1");
    await api.testAgentWebhook("agent-1", { question: "hello" });
    for (const method of ["GET", "PUT", "PATCH", "DELETE", "HEAD"] as const) {
      await api.testAgentWebhook("agent-1", { question: "hello" }, method);
    }
    await api.rerunAgentDocument({
      id: "doc-1",
      component_id: "node-1",
      dsl: EMPTY_PLATFORM_AGENT_DSL,
    });
    await api.testAgentDatabaseConnection({
      db_type: "mysql",
      database: "rag",
      username: "user",
      host: "db.test",
      port: 3306,
      password: "secret",
    });
    await api.listAgentComponents();
    await api.listAgentTemplates();
    await api.getAgentPrompts();
    await api.listAvailableAgentTags();
    await api.uploadAgentFiles("agent-1", [new File(["x"], "note.txt")]);
    await api.downloadAgentFile("file-1");
    await api.previewAgentAttachment("attachment-1", {
      ext: "pdf",
      mimeType: "application/pdf",
      filename: "report.pdf",
    });
    await api.downloadAgentAttachment("attachment-1", {
      ext: "pdf",
      mimeType: "application/pdf",
      filename: "report.pdf",
    });
    await api.deleteAgent("agent-1");

    expect(seen).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/agents",
          body: { title: "Demo", dsl: EMPTY_PLATFORM_AGENT_DSL },
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/agents/templates",
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/agents/prompts",
        }),
        expect.objectContaining({ method: "GET", path: "/api/v1/agents/tags" }),
        expect.objectContaining({
          method: "PUT",
          path: "/api/v1/agents/agent-1/tags",
          body: { tags: ["prod"] },
        }),
        expect.objectContaining({
          method: "DELETE",
          path: "/api/v1/agents/agent-1/run",
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/tasks/session-1/cancel",
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/agents/agent-1/components/begin/debug",
          body: { params: { query: { value: "hello" } } },
        }),
        expect.objectContaining({
          method: "DELETE",
          path: "/api/v1/agents/agent-1/sessions",
          query: "?ids=one%2Ctwo",
        }),
        expect.objectContaining({
          method: "DELETE",
          path: "/api/v1/agents/agent-1/sessions",
          query: "?delete_all=true",
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/agents/agent-1/upload",
          body: "multipart",
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/agents/attachments/attachment-1/preview",
          query: "?ext=pdf&mime_type=application%2Fpdf&filename=report.pdf",
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/agents/agent-1/webhook/test",
          body: { question: "hello" },
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/agents/test_db_connection",
          body: expect.objectContaining({ password: "secret" }),
        }),
      ]),
    );
  });

  it("uses exact MCP server and plugin-tool contracts", async () => {
    const input = {
      name: "Tools",
      url: "https://mcp.test",
      server_type: "sse",
      variables: { REGION: "eu" },
      headers: { Authorization: "Bearer ephemeral" },
      timeout: 10,
    };
    await api.listMcpServers();
    await api.createMcpServer(input);
    await api.getMcpServer("mcp-1");
    await api.updateMcpServer("mcp-1", { description: "Updated" });
    await api.testMcpServer("mcp-1", input);
    await api.importMcpServers(
      { Tools: { url: "https://mcp.test", type: "sse" } },
      10,
    );
    await api.deleteMcpServer("mcp-1");
    await api.listPluginTools();

    expect(seen).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/mcp/servers",
          query: "?page=1&page_size=100&orderby=create_time&desc=true",
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/mcp/servers",
          body: input,
        }),
        expect.objectContaining({
          method: "PUT",
          path: "/api/v1/mcp/servers/mcp-1",
          body: { description: "Updated" },
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/mcp/servers/mcp-1/test",
          body: input,
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/mcp/servers/import",
          body: {
            mcpServers: { Tools: { url: "https://mcp.test", type: "sse" } },
            timeout: 10,
          },
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/plugin/tools",
        }),
      ]),
    );
  });
});
