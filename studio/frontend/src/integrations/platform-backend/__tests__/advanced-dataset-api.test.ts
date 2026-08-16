import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../advanced-dataset-api";
import { platformTestServer } from "./test-server";

interface Seen {
  method: string;
  path: string;
  query: string;
  body: unknown;
}
const ok = (data: unknown) =>
  HttpResponse.json({ code: 0, data, message: "success" });

describe("Rag Platform Phase 10 advanced dataset contracts", () => {
  const seen: Seen[] = [];
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    seen.length = 0;
    platformTestServer.use(
      http.all("http://platform.test/api/v1/*", async ({ request }) => {
        const url = new URL(request.url);
        const body =
          request.method === "GET" || request.method === "HEAD"
            ? null
            : await request.json().catch(() => null);
        seen.push({
          method: request.method,
          path: url.pathname,
          query: url.search,
          body,
        });
        if (url.pathname.endsWith("/metadata/config"))
          return ok({ metadata: [], built_in_metadata: [] });
        if (url.pathname.endsWith("/metadata/summary"))
          return ok({ summary: {} });
        if (
          url.pathname.endsWith("/tags") ||
          url.pathname.endsWith("/tags/aggregation")
        )
          return ok([]);
        if (
          url.pathname.endsWith("/any_artifact") ||
          url.pathname.endsWith("/any_skill")
        )
          return ok({ has: true });
        if (url.pathname.endsWith("/artifacts/graph"))
          return ok({ entities: [], relations: [] });
        if (url.pathname.endsWith("/artifacts") && request.method === "GET")
          return ok({ total: 0, items: [] });
        if (url.pathname.includes("/artifacts/entity/"))
          return ok({ slug: "topic/one", content_md: "# One" });
        if (url.pathname.endsWith("/index") && request.method === "POST")
          return ok({ task_id: "task-1" });
        if (url.pathname.endsWith("/index"))
          return ok({
            id: "task-1",
            doc_id: "graph_raptor_x",
            task_type: "graphrag",
            progress: 0.5,
          });
        if (url.pathname.endsWith("/embedding/check"))
          return ok({
            summary: {
              kb_id: "dataset-1",
              model: "emb",
              sampled: 1,
              valid: 1,
              avg_cos_sim: 1,
              min_cos_sim: 1,
              max_cos_sim: 1,
              match_mode: "content_only",
            },
            results: [],
          });
        if (url.pathname.endsWith("/embedding"))
          return ok({ scheduled_count: 2 });
        if (url.pathname.endsWith("/ingestions/summary"))
          return ok({ doc_num: 2, chunk_num: 3, token_num: 4, status: {} });
        if (url.pathname.endsWith("/ingestions"))
          return ok({ total: 0, logs: [] });
        if (url.pathname.includes("/ingestions/")) return ok({ id: "log-1" });
        if (url.pathname.endsWith("/skills"))
          return ok({ skill_kwd: "root", children: [] });
        if (url.pathname.includes("/skills/"))
          return ok({ skill_kwd: "topic/one", content_md: "# Skill" });
        return ok({});
      }),
    );
  });
  afterEach(() => vi.unstubAllEnvs());

  it("uses exact metadata, tag, graph, artifact, index, embedding and ingestion contracts", async () => {
    const metadata = {
      metadata: [
        { key: "department", type: "string", description: null, enum: [] },
      ],
      built_in_metadata: [],
    };
    const batch = {
      selector: { document_ids: ["doc-1"], metadata_condition: {} },
      updates: [{ key: "department", value: "R&D" }],
      deletes: [],
    };
    await api.getDatasetMetadataConfig("dataset-1");
    await api.updateDatasetMetadataConfig("dataset-1", metadata);
    await api.getFlattenedDatasetMetadata(["dataset-1"]);
    await api.getDatasetMetadataSummary("dataset-1", ["doc-1"]);
    await api.updateDocumentMetadataConfig("dataset-1", "doc-1", {
      enabled: true,
    });
    await api.batchUpdateDatasetMetadata("dataset-1", batch);
    await api.patchDatasetDocumentMetadata("dataset-1", batch);
    await api.batchUpdateDatasetDocumentStatus("dataset-1", ["doc-1"], 1);
    await api.listDatasetTags("dataset-1");
    await api.aggregateDatasetTags(["dataset-1"]);
    await api.renameDatasetTag("dataset-1", "old", "new");
    await api.removeDatasetTags("dataset-1", ["old"]);
    await api.getDatasetGraph("dataset-1");
    await api.searchDatasets(["dataset-1"], "What is RAG?");
    await api.hasDatasetArtifacts("dataset-1");
    await api.listDatasetArtifacts("dataset-1", {
      page: 1,
      pageSize: 200,
      pageType: "entity",
    });
    await api.getDatasetArtifactGraph("dataset-1", "topic/one");
    await api.getDatasetArtifactPage("dataset-1", "entity", "topic/one");
    await api.updateDatasetArtifactPage("dataset-1", "entity", "topic/one", {
      content_md: "# One",
      title: "One",
      comments: "edit",
    });
    await api.clearDatasetArtifacts("dataset-1");
    await api.startDatasetIndex("dataset-1", "graph");
    await api.getDatasetIndexStatus("dataset-1", "raptor");
    await api.deleteDatasetIndex("dataset-1", "mindmap", true);
    await api.deleteDatasetIndexByQuery("dataset-1", "mindmap", false);
    await api.runDatasetEmbedding("dataset-1");
    await api.checkDatasetEmbedding("dataset-1", "emb", 5);
    await api.getDatasetIngestionSummary("dataset-1");
    await api.listDatasetIngestionLogs("dataset-1", {
      page: 1,
      pageSize: 30,
      operationStatus: ["running", "done"],
      logType: "file",
    });
    await api.getDatasetIngestionLog("dataset-1", "log-1");
    await api.hasDatasetSkills("dataset-1");
    await api.getDatasetSkillTree("dataset-1");
    await api.getDatasetSkillPage("dataset-1", "topic/one");

    expect(seen).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          method: "PUT",
          path: "/api/v1/datasets/dataset-1/metadata/config",
          body: metadata,
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/datasets/metadata/flattened",
          query: "?dataset_ids=dataset-1",
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/datasets/dataset-1/metadata/update",
          body: batch,
        }),
        expect.objectContaining({
          method: "PATCH",
          path: "/api/v1/datasets/dataset-1/documents/metadatas",
          body: batch,
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/datasets/dataset-1/documents/batch-update-status",
          body: { doc_ids: ["doc-1"], status: 1 },
        }),
        expect.objectContaining({
          method: "PUT",
          path: "/api/v1/datasets/dataset-1/tags",
          body: { from_tag: "old", to_tag: "new" },
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/datasets/dataset-1/artifacts/entity/topic/one",
        }),
        expect.objectContaining({
          method: "DELETE",
          path: "/api/v1/datasets/dataset-1/mindmap",
          query: "?wipe=true",
        }),
        expect.objectContaining({
          method: "DELETE",
          path: "/api/v1/datasets/dataset-1/index",
          query: "?type=mindmap&wipe=false",
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/datasets/search",
          body: expect.objectContaining({
            dataset_ids: ["dataset-1"],
            question: "What is RAG?",
          }),
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/datasets/dataset-1/embedding/check",
          body: { embd_id: "emb", check_num: 5 },
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/datasets/dataset-1/ingestions",
          query:
            "?page=1&page_size=30&orderby=create_time&desc=true&operation_status=running&operation_status=done&log_type=file",
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/datasets/dataset-1/skills/topic/one",
        }),
      ]),
    );
  });

  it("uses exact tenant-owned global skill space/config/search/index contracts", async () => {
    platformTestServer.use(
      http.all("http://platform.test/api/v1/skills/*", async ({ request }) => {
        const url = new URL(request.url);
        const body =
          request.method === "GET"
            ? null
            : await request.json().catch(() => null);
        seen.push({
          method: request.method,
          path: url.pathname,
          query: url.search,
          body,
        });
        if (url.pathname.endsWith("/spaces") && request.method === "GET")
          return ok({ spaces: [], total: 0 });
        if (url.pathname.endsWith("/search"))
          return ok({
            skills: [],
            total: 0,
            query: "rag",
            search_type: "hybrid",
          });
        if (url.pathname.endsWith("/config"))
          return ok({
            space_id: "space-1",
            embd_id: "emb",
            vector_similarity_weight: 0.3,
            similarity_threshold: 0.2,
            field_config: {},
            top_k: 10,
          });
        if (url.pathname.endsWith("/index") && request.method === "POST")
          return ok({ indexed_count: 1 });
        if (url.pathname.endsWith("/index")) return ok(true);
        if (request.method === "DELETE")
          return ok({ deleting: true, space_id: "space-1" });
        return ok({
          id: "space-1",
          name: "Docs",
          folder_id: "folder-1",
          top_k: 10,
          status: "active",
        });
      }),
    );
    const fields = {
      name: { enabled: true, weight: 3 },
      tags: { enabled: true, weight: 2 },
      description: { enabled: true, weight: 1 },
      content: { enabled: false, weight: 0.5 },
    };
    await api.listGlobalSkillSpaces();
    await api.createGlobalSkillSpace({
      name: "Docs",
      description: "",
      embd_id: "emb",
      rerank_id: "",
    });
    await api.getGlobalSkillSpace("space-1");
    await api.updateGlobalSkillSpace("space-1", {
      name: "Docs",
      description: "x",
      embd_id: "emb",
      rerank_id: "rerank",
      top_k: 10,
    });
    await api.getGlobalSkillSpaceByFolder("folder-1");
    await api.getGlobalSkillSearchConfig("space-1", "emb");
    await api.updateGlobalSkillSearchConfig({
      space_id: "space-1",
      embd_id: "emb",
      vector_similarity_weight: 0.3,
      similarity_threshold: 0.2,
      field_config: fields,
      rerank_id: "",
      top_k: 10,
    });
    await api.searchGlobalSkills({
      space_id: "space-1",
      query: "rag",
      page: 1,
      page_size: 25,
      sort_by: "relevance",
      sort_order: "desc",
    });
    await api.indexGlobalSkills(
      "space-1",
      [
        {
          id: "skill-1",
          folder_id: "folder-1",
          name: "RAG",
          description: "",
          tags: [],
          content: "body",
          version: "1.0.0",
        },
      ],
      "emb",
    );
    await api.reindexGlobalSkills("space-1", "emb");
    await api.deleteGlobalSkillIndex("space-1", "skill-1");
    await api.deleteGlobalSkillSpace("space-1");
    expect(seen).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/skills/config",
          body: expect.objectContaining({
            space_id: "space-1",
            field_config: fields,
          }),
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/skills/search",
          body: expect.objectContaining({ space_id: "space-1", query: "rag" }),
        }),
        expect.objectContaining({
          method: "POST",
          path: "/api/v1/skills/index",
          body: expect.objectContaining({
            space_id: "space-1",
            embd_id: "emb",
          }),
        }),
        expect.objectContaining({
          method: "DELETE",
          path: "/api/v1/skills/index",
          query: "?space_id=space-1&skill_id=skill-1",
        }),
        expect.objectContaining({
          method: "GET",
          path: "/api/v1/skills/space/by-folder",
          query: "?folder_id=folder-1",
        }),
        expect.objectContaining({
          method: "DELETE",
          path: "/api/v1/skills/spaces/space-1",
        }),
      ]),
    );
  });
});
