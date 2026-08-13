import { describe, expect, it } from "vitest";

import {
  evaluatePlatformReadiness,
  mapPipelineToDatasetFields,
} from "../model-readiness";
import type { PlatformDefaultModel, PlatformModel } from "../model-types";

const model = (id: string, capability: string): PlatformModel => ({
  id,
  name: id,
  providerId: "provider-1",
  providerName: "Provider",
  instanceId: "instance-1",
  instanceName: "primary",
  capabilities: [capability],
  status: "active",
  maxTokens: null,
});

const selected = (id: string, capability: string): PlatformDefaultModel => ({
  capability,
  enabled: true,
  instanceName: "primary",
  modelId: id,
  modelName: id,
  providerName: "Provider",
});

describe("Phase 3 readiness and dataset mapping", () => {
  it("requires server-confirmed chat and embedding defaults backed by compatible models", () => {
    const ready = evaluatePlatformReadiness(
      [model("chat-1", "chat"), model("embedding-1", "embedding")],
      [selected("chat-1", "chat"), selected("embedding-1", "embedding")],
      "both",
    );
    expect(ready).toMatchObject({ ready: true, missing: [] });

    const missing = evaluatePlatformReadiness(
      [model("chat-1", "chat")],
      [selected("chat-1", "chat"), selected("wrong-type", "embedding")],
      "both",
    );
    expect(missing).toMatchObject({ ready: false, missing: ["embedding"] });
  });

  it("maps pipeline selection to the exact Go dataset create contract", () => {
    expect(mapPipelineToDatasetFields(" pipeline-1 ")).toEqual({
      pipeline_id: "pipeline-1",
      parse_type: 2,
    });
    expect(mapPipelineToDatasetFields(" ")).toBeNull();
  });
});
