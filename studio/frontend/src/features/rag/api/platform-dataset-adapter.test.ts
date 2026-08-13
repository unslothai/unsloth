import { describe, expect, it } from "vitest";

import {
  datasetEmbeddingModelReference,
  mapDatasetToKnowledgeBase,
} from "./platform-dataset-adapter";

describe("Phase 4 dataset domain adapter", () => {
  it("maps Python and Go dataset aliases into the stable KnowledgeBase domain", () => {
    expect(
      mapDatasetToKnowledgeBase({
        id: "dataset-1",
        name: "Product docs",
        description: "Reference",
        document_count: "3",
        embedding_model: "embedding-1",
        permission: "team",
        parser_id: "book",
        parser_config: { chunk_token_num: 256 },
        pipeline_id: null,
        create_time: 1_786_502_416_419,
        update_date: "2026-08-14T12:00:00",
      }),
    ).toEqual({
      id: "dataset-1",
      name: "Product docs",
      description: "Reference",
      documentCount: 3,
      embeddingModel: "embedding-1",
      permission: "team",
      chunkMethod: "book",
      parserConfig: { chunk_token_num: 256 },
      pipelineId: null,
      createdAt: "2026-08-12T02:40:16.419Z",
      updatedAt: "2026-08-14T12:00:00",
    });
  });

  it("rejects malformed entity responses instead of creating ghost rows", () => {
    expect(() => mapDatasetToKnowledgeBase({ id: "", name: "Missing id" })).toThrow(
      /id veya ad eksik/,
    );
  });

  it("uses a tenant model id when canonical and otherwise builds the backend composite reference", () => {
    const base = {
      name: "nvidia/nemotron-3-embed-1b:free@openai",
      providerId: "provider-1",
      providerName: "VLLM",
      instanceId: "instance-1",
      instanceName: "mdy",
      capabilities: ["embedding"],
      status: "active",
      maxTokens: null,
    };

    expect(
      datasetEmbeddingModelReference({
        ...base,
        id: "0123456789abcdef0123456789abcdef",
      }),
    ).toBe("0123456789abcdef0123456789abcdef");
    expect(
      datasetEmbeddingModelReference({ ...base, id: base.name }),
    ).toBe("nvidia/nemotron-3-embed-1b:free@openai@mdy@VLLM");
  });
});
