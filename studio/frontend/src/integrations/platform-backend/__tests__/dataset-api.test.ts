import { delay, http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import fixtureText from "../../../../../../docs/rag-platform/fixtures/dataset.json?raw";
import { createDatasetKnowledgeBase } from "../../../features/rag/api/platform-dataset-adapter";
import {
  createPlatformDataset,
  deletePlatformDatasets,
  getPlatformDataset,
  listPlatformDatasets,
  updatePlatformDataset,
} from "../dataset-api";
import { platformTestServer } from "./test-server";

const fixture = JSON.parse(fixtureText) as {
  interactions: Array<{
    name: string;
    response: { body: Record<string, unknown> };
  }>;
};

function bodyFor(name: string): Record<string, unknown> {
  const interaction = fixture.interactions.find((item) => item.name === name);
  if (!interaction) throw new Error("Dataset fixture is missing " + name);
  return interaction.response.body;
}

describe("Rag Platform Phase 4 dataset service", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
  });

  afterEach(() => vi.unstubAllEnvs());

  it("preserves exact list pagination, search, sorting and total metadata", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/datasets", ({ request }) => {
        const query = new URL(request.url).searchParams;
        expect(Object.fromEntries(query)).toEqual({
          page: "2",
          page_size: "8",
          name: "Product",
          orderby: "update_time",
          desc: "false",
        });
        return HttpResponse.json(bodyFor("dataset.list"));
      }),
    );

    const result = await listPlatformDatasets({
      page: 2,
      pageSize: 8,
      name: " Product ",
      orderby: "update_time",
      desc: false,
    });
    expect(result.total).toBe(1);
    expect(result.items[0]).toMatchObject({
      id: expect.any(String),
      name: expect.any(String),
      chunk_method: "naive",
    });
  });

  it("uses the active hybrid CRUD contracts and never retries mutations", async () => {
    const calls: Array<{ method: string; body: unknown }> = [];
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/datasets/dataset-1",
        () => HttpResponse.json(bodyFor("dataset.get")),
      ),
      http.post(
        "http://platform.test/api/v1/datasets",
        async ({ request }) => {
          calls.push({ method: "POST", body: await request.json() });
          return HttpResponse.json(bodyFor("dataset.create"));
        },
      ),
      http.put(
        "http://platform.test/api/v1/datasets/dataset-1",
        async ({ request }) => {
          calls.push({ method: "PUT", body: await request.json() });
          return HttpResponse.json(bodyFor("dataset.update"));
        },
      ),
      http.delete(
        "http://platform.test/api/v1/datasets",
        async ({ request }) => {
          calls.push({ method: "DELETE", body: await request.json() });
          return HttpResponse.json({ code: 0, data: { success_count: 1 } });
        },
      ),
    );

    await expect(getPlatformDataset("dataset-1")).resolves.toMatchObject({
      id: expect.any(String),
    });
    await createPlatformDataset({
      name: "Product",
      description: "Docs",
      embedding_model: "embedding-model-1",
      permission: "team",
      chunk_method: "naive",
      parser_config: { chunk_token_num: 512 },
    });
    await updatePlatformDataset("dataset-1", {
      name: "Product 2",
      description: "",
      embedding_model: "embedding-model-1",
      permission: "me",
      parser_id: "book",
      parse_type: 1,
    });
    await deletePlatformDatasets(["dataset-1"]);

    expect(calls).toEqual([
      {
        method: "POST",
        body: {
          name: "Product",
          description: "Docs",
          embedding_model: "embedding-model-1",
          permission: "team",
          chunk_method: "naive",
          parser_config: { chunk_token_num: 512 },
        },
      },
      {
        method: "PUT",
        body: {
          name: "Product 2",
          description: "",
          embedding_model: "embedding-model-1",
          permission: "me",
          parser_id: "book",
          parse_type: 1,
        },
      },
      { method: "DELETE", body: { ids: ["dataset-1"] } },
    ]);
  });

  it("maps pipeline mode without sending the mutually exclusive chunk_method", async () => {
    let body: unknown;
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/datasets",
        async ({ request }) => {
          body = await request.json();
          return HttpResponse.json(bodyFor("dataset.create"));
        },
      ),
    );

    await createDatasetKnowledgeBase({
      name: "Pipeline dataset",
      embeddingModel: "embedding-model-1",
      permission: "me",
      chunkMethod: "book",
      pipelineId: "0123456789abcdef0123456789abcdef",
    });

    expect(body).toEqual({
      name: "Pipeline dataset",
      embedding_model: "embedding-model-1",
      permission: "me",
      pipeline_id: "0123456789abcdef0123456789abcdef",
      parse_type: 2,
    });
  });

  it("forwards abort and rejects HTTP-200 business errors", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/datasets", async () => {
        await delay(100);
        return HttpResponse.json(bodyFor("dataset.list"));
      }),
      http.get(
        "http://platform.test/api/v1/datasets/missing",
        () => HttpResponse.json(bodyFor("dataset.not_found")),
      ),
    );

    const controller = new AbortController();
    const pending = listPlatformDatasets(
      { page: 1, pageSize: 8 },
      controller.signal,
    );
    controller.abort();
    await expect(pending).rejects.toMatchObject({ code: "CLIENT_ABORTED" });
    await expect(getPlatformDataset("missing")).rejects.toMatchObject({
      code: 102,
      endpoint: "/datasets/missing",
    });
  });
});
