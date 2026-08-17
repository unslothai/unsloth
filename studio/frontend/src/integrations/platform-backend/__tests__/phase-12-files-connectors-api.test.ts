import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as connectors from "../connector-api";
import {
  clearPendingConnectorOAuth,
  matchesConnectorOAuthCorrelation,
  parseConnectorOAuthWindowName,
  readPendingConnectorOAuth,
  savePendingConnectorOAuth,
} from "../connector-oauth-state";
import { redactConnectorSecrets } from "../connector-types";
import * as files from "../file-api";
import { platformTestServer } from "./test-server";

interface Seen {
  method: string;
  path: string;
  query: string;
  body: unknown;
}

const ok = (data: unknown) =>
  HttpResponse.json({ code: 0, data, message: "success" });

describe("Rag Platform Phase 12 connector and file contracts", () => {
  const seen: Seen[] = [];

  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    seen.length = 0;
    sessionStorage.clear();
    platformTestServer.use(
      http.all("http://platform.test/*", async ({ request }) => {
        const url = new URL(request.url);
        const contentType = request.headers.get("content-type") ?? "";
        const body = ["GET", "HEAD"].includes(request.method)
          ? null
          : contentType.includes("multipart/form-data")
            ? "multipart"
            : await request.json().catch(() => null);
        seen.push({ method: request.method, path: url.pathname, query: url.search, body });
        if (url.pathname.endsWith("/logs")) return ok({ total: 1, logs: [{ id: "log-1", connector_id: "connector-1" }] });
        if (url.pathname === "/api/v1/datasets/dataset-1" && request.method === "GET") return ok({ id: "dataset-1", connectors: [{ id: "connector-existing", auto_parse: "0" }] });
        if (url.pathname === "/api/v1/connectors" && request.method === "GET") return ok([{ id: "connector-1", name: "API", source: "rest_api" }]);
        if (url.pathname.includes("/oauth/web/start")) return ok({ flow_id: "flow-1", authorization_url: "https://provider.test/auth", expires_in: 600 });
        if (url.pathname.includes("/oauth/web/result")) return ok({ credentials: { account: "memory-only" } });
        if (url.pathname === "/api/v1/files" && request.method === "GET") return ok({ total: 1, files: [{ id: "file-1", name: "note.txt", type: "file" }], parent_folder: null });
        if (url.pathname === "/api/v1/files" && request.method === "POST") return ok(contentType.includes("multipart") ? [{ id: "file-1", name: "note.txt", type: "file" }] : { id: "folder-1", name: "Docs", type: "folder" });
        if (url.pathname.endsWith("/parent")) return ok({ parent_folder: { id: "folder-1", type: "folder" } });
        if (url.pathname.endsWith("/ancestors")) return ok({ parent_folders: [{ id: "folder-1", type: "folder" }] });
        if (url.pathname.endsWith("/versions")) return ok([{ commit_id: "commit-1", operation: "add" }]);
        if (url.pathname.endsWith("/changes")) return ok([
          { file_id: "file-1", file_name: "renamed.txt", operation: "rename", old_name: "note.txt", new_name: "renamed.txt" },
          { file_id: "file-1", file_name: "renamed.txt", operation: "move", old_parent_id: "root", new_parent_id: "folder-1" },
        ]);
        if (url.pathname.endsWith("/commits/diff")) return ok([{ file_id: "file-1", operation: "modify" }]);
        if (url.pathname.endsWith("/commits") && request.method === "GET") return ok({ total: 1, page: 1, page_size: 20, commits: [{ id: "commit-1", message: "Initial" }] });
        if (url.pathname.endsWith("/files/file-1/content")) return ok({ content: "hello" });
        if (url.pathname.endsWith("/tree")) return ok({ root: [] });
        if (url.pathname.endsWith("/files") && url.pathname.includes("/commits/")) return ok([{ id: "item-1", file_id: "file-1" }]);
        if (url.pathname.endsWith("/commits/commit-1")) return ok({ id: "commit-1", files: [] });
        if (url.pathname.endsWith("/commits") && request.method === "POST") return ok({ id: "commit-1", message: "Initial" });
        if (url.pathname === "/api/v1/files/file-1") return new HttpResponse('{"hello":"world"}', { headers: { "content-type": "application/json" } });
        return ok(true);
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    sessionStorage.clear();
  });

  it("uses exact connector CRUD, test, rebuild, log and OAuth contracts", async () => {
    await connectors.listPlatformConnectors();
    await connectors.createPlatformConnector({ name: "API", source: "rest_api", config: { url: "https://example.test" }, refreshFrequency: 60, pruneFrequency: 120, timeoutSeconds: 30 });
    await connectors.getPlatformConnector("connector-1");
    await connectors.updatePlatformConnector("connector-1", { status: "SCHEDULE", reschedule: true });
    await connectors.testPlatformConnector("connector-1");
    await connectors.linkAndRebuildPlatformConnector("connector-1", "dataset-1");
    await connectors.listPlatformConnectorLogs("connector-1", { page: 2, pageSize: 10 });
    await connectors.startGoogleConnectorOAuth("gmail", { web: { client_id: "id" } }, "http://app.test/callback");
    await connectors.startBoxConnectorOAuth("id", "secret", "http://app.test/callback");
    await connectors.completeConnectorOAuthCallback("box", { state: "flow-1", code: "code-1" });
    expect(await connectors.waitForConnectorOAuthResult("gmail", "flow-1")).toEqual({ account: "memory-only" });
    expect(await connectors.waitForConnectorOAuthResult("box", "flow-1")).toEqual({ account: "memory-only" });
    await connectors.deletePlatformConnector("connector-1");
    expect(seen).toEqual(expect.arrayContaining([
      expect.objectContaining({ method: "POST", path: "/api/v1/connectors/connector-1/test" }),
      expect.objectContaining({ method: "POST", path: "/api/v1/connectors/connector-1/rebuild", body: { kb_id: "dataset-1" } }),
      expect.objectContaining({ method: "PUT", path: "/api/v1/datasets/dataset-1", body: { connectors: [{ id: "connector-existing", auto_parse: "0" }, { id: "connector-1", auto_parse: "1" }] } }),
      expect.objectContaining({ method: "GET", path: "/api/v1/connectors/connector-1/logs", query: "?page=2&page_size=10" }),
      expect.objectContaining({ method: "POST", path: "/api/v1/connectors/google/oauth/web/start", query: "?type=gmail" }),
      expect.objectContaining({ method: "POST", path: "/api/v1/connectors/box/oauth/web/start", body: { client_id: "id", client_secret: "secret", redirect_uri: "http://app.test/callback" } }),
      expect.objectContaining({ method: "GET", path: "/connectors/box/oauth/web/callback", query: "?state=flow-1&code=code-1" }),
      expect.objectContaining({ method: "POST", path: "/api/v1/connectors/google/oauth/web/result", query: "?type=gmail", body: { flow_id: "flow-1" } }),
      expect.objectContaining({ method: "POST", path: "/api/v1/connectors/box/oauth/web/result", body: { flow_id: "flow-1" } }),
    ]));
  });

  it("uses exact file, dataset-link, version and commit contracts", async () => {
    await files.listPlatformFiles({ parentId: "root", keywords: "note", page: 2, pageSize: 25 });
    await files.createPlatformFolder("Docs", "root");
    await files.uploadPlatformFiles([new File(["hello"], "note.txt")], "root");
    await files.movePlatformFiles(["file-1"], { destinationFolderId: "folder-1", newName: "renamed.txt" });
    await files.linkPlatformFilesToDatasets(["file-1"], ["dataset-1"], "replace");
    await files.getPlatformFileParent("file-1");
    await files.getPlatformFileAncestors("file-1");
    const downloaded = await files.downloadPlatformFile("file-1");
    await expect(downloaded.text()).resolves.toBe('{"hello":"world"}');
    await files.listPlatformFileVersions("file-1");
    await files.listPlatformCommits("datasets", "dataset-1", { page: 1, pageSize: 20 });
    await files.createPlatformCommit("datasets", "dataset-1", "Initial", [
      { fileId: "file-1", fileName: "note.txt", operation: "add", content: "hello" },
      { fileId: "file-1", fileName: "renamed.txt", operation: "rename", oldName: "note.txt", newName: "renamed.txt" },
      { fileId: "file-1", fileName: "renamed.txt", operation: "move", oldParentId: "root", newParentId: "folder-1" },
    ]);
    await files.getPlatformCommit("datasets", "dataset-1", "commit-1");
    await files.listPlatformCommitFiles("datasets", "dataset-1", "commit-1");
    await files.diffPlatformCommits("datasets", "dataset-1", "one", "two");
    const changes = await files.getPlatformUncommittedChanges("datasets", "dataset-1");
    expect(changes).toEqual([
      expect.objectContaining({ operation: "rename", oldName: "note.txt", newName: "renamed.txt" }),
      expect.objectContaining({ operation: "move", oldParentId: "root", newParentId: "folder-1" }),
    ]);
    await files.getPlatformCommitTree("datasets", "dataset-1", "commit-1");
    await files.getPlatformCommitFileContent("datasets", "dataset-1", "commit-1", "file-1");
    await files.deletePlatformFiles(["file-1"]);
    expect(seen).toEqual(expect.arrayContaining([
      expect.objectContaining({ method: "POST", path: "/api/v1/files/move", body: { src_file_ids: ["file-1"], dest_file_id: "folder-1", new_name: "renamed.txt" } }),
      expect.objectContaining({ method: "POST", path: "/api/v1/files/link-to-datasets", query: "?mode=replace", body: { file_ids: ["file-1"], kb_ids: ["dataset-1"] } }),
      expect.objectContaining({ method: "POST", path: "/api/v1/datasets/dataset-1/commits", body: { message: "Initial", files: [
        { file_id: "file-1", file_name: "note.txt", operation: "add", content: "hello" },
        { file_id: "file-1", file_name: "renamed.txt", operation: "rename", old_name: "note.txt", new_name: "renamed.txt" },
        { file_id: "file-1", file_name: "renamed.txt", operation: "move", old_parent_id: "root", new_parent_id: "folder-1" },
      ] } }),
      expect.objectContaining({ method: "GET", path: "/api/v1/datasets/dataset-1/commits/diff", query: "?from=one&to=two" }),
    ]));
  });

  it("redacts secrets and stores only non-secret OAuth correlation state", () => {
    expect(redactConnectorSecrets({ token: "secret", nested: { password: "secret", safe: "visible" } })).toEqual({ token: "[redacted]", nested: { password: "[redacted]", safe: "visible" } });
    savePendingConnectorOAuth({ source: "gmail", flowId: "flow-1", returnTo: "/files", startedAt: 1 });
    expect(readPendingConnectorOAuth()).toEqual({ source: "gmail", flowId: "flow-1", returnTo: "/files", startedAt: 1 });
    expect(sessionStorage.getItem("rag-platform.connector-oauth.pending")).not.toContain("token");
    expect(parseConnectorOAuthWindowName("rag-platform-oauth:box:flow-2")).toEqual({ source: "box", flowId: "flow-2" });
    expect(matchesConnectorOAuthCorrelation("gmail", "flow-1", "")).toBe(true);
    expect(matchesConnectorOAuthCorrelation("gmail", "wrong-state", "")).toBe(false);
    expect(matchesConnectorOAuthCorrelation("box", "flow-2", "rag-platform-oauth:box:flow-2")).toBe(true);
    clearPendingConnectorOAuth("flow-1");
    expect(readPendingConnectorOAuth()).toBeNull();
  });
});
