import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  addInstanceModel,
  addProvider,
  chatToModel,
  createEmbeddings,
  createProviderInstance,
  deleteInstanceModels,
  deleteProviderInstances,
  getDefaultModels,
  getPipelineDsl,
  getProviderInstance,
  getProviderInstanceBalance,
  getProviderTask,
  listAvailableProviders,
  listInstanceModels,
  listPipelines,
  listProviderInstances,
  listProviderTasks,
  listSupportedInstanceModels,
  listTenantModels,
  ocrFile,
  parseFile,
  rerankDocuments,
  setDefaultModel,
  synthesizeSpeech,
  testProviderConnection,
  testProviderInstanceConnection,
  transcribeAudio,
  updateInstanceModel,
  updateProviderInstance,
} from "../model-api";
import { platformTestServer } from "./test-server";

const success = (data: unknown = true) =>
  HttpResponse.json({ code: 0, data, message: "success" });

describe("Rag Platform Phase 3 model service", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    localStorage.clear();
  });

  afterEach(() => vi.unstubAllEnvs());

  it("normalizes providers and permanently drops returned api_key material", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/providers", ({ request }) => {
        expect(new URL(request.url).searchParams.get("available")).toBe("true");
        return success([{ id: "openai", name: "OpenAI", description: "Chat" }]);
      }),
      http.get(
        "http://platform.test/api/v1/providers/OpenAI/instances/primary",
        () =>
          success({
            id: "instance-1",
            instance_name: "primary",
            provider_id: "openai",
            api_key: "server-secret-must-not-survive",
            base_url: "https://example.test",
          }),
      ),
    );

    await expect(listAvailableProviders()).resolves.toEqual([
      { id: "openai", name: "OpenAI", description: "Chat", hasInstance: false },
    ]);
    const instance = await getProviderInstance("OpenAI", "primary");
    expect(instance.hasCredential).toBe(true);
    expect(JSON.stringify(instance)).not.toContain(
      "server-secret-must-not-survive",
    );
  });

  it("uses exact provider CRUD, connection, balance and task contracts without persisting secrets", async () => {
    const calls: string[] = [];
    platformTestServer.use(
      http.put("http://platform.test/api/v1/providers", async ({ request }) => {
        expect(await request.json()).toEqual({ provider_name: "OpenAI" });
        calls.push("provider");
        // Python get_result() omits data for successful no-content mutations.
        return HttpResponse.json({ code: 0 });
      }),
      http.post(
        "http://platform.test/api/v1/providers/OpenAI/instances",
        async ({ request }) => {
          expect(await request.json()).toEqual({
            instance_name: "primary",
            api_key: "ephemeral-key",
            base_url: "https://api.test",
            region: "eu",
          });
          calls.push("instance");
          return success();
        },
      ),
      http.put(
        "http://platform.test/api/v1/providers/OpenAI/instances/primary",
        async ({ request }) => {
          expect(await request.json()).toEqual({
            instance_name: "renamed",
            api_key: "replacement-key",
            base_url: "https://api-2.test",
            region: "us",
            verify: true,
          });
          calls.push("update");
          return success();
        },
      ),
      http.delete(
        "http://platform.test/api/v1/providers/OpenAI/instances",
        async ({ request }) => {
          expect(await request.json()).toEqual({ instances: ["renamed"] });
          calls.push("delete");
          return success();
        },
      ),
      http.post(
        "http://platform.test/api/v1/providers/OpenAI/connection",
        async ({ request }) => {
          expect(await request.json()).toEqual({
            api_key: "draft-key",
            base_url: "https://api.test",
            region: "eu",
            model_info: [],
          });
          calls.push("draft-test");
          return success();
        },
      ),
      http.get(
        "http://platform.test/api/v1/providers/OpenAI/instances/renamed/connection",
        () => success(),
      ),
      http.get(
        "http://platform.test/api/v1/providers/OpenAI/instances/renamed/balance",
        () => success({ currency: "USD", balance: 12 }),
      ),
      http.get(
        "http://platform.test/api/v1/providers/OpenAI/instances/renamed/tasks",
        () => success([{ task_id: "task-1", status: "done" }]),
      ),
      http.get(
        "http://platform.test/api/v1/providers/OpenAI/instances/renamed/tasks/task-1",
        () => success({ segments: [{ index: 0, content: "ok" }] }),
      ),
    );

    await addProvider("OpenAI");
    await createProviderInstance("OpenAI", {
      instanceName: "primary",
      apiKey: "ephemeral-key",
      baseUrl: "https://api.test",
      region: "eu",
    });
    await updateProviderInstance("OpenAI", "primary", {
      instanceName: "renamed",
      apiKey: "replacement-key",
      baseUrl: "https://api-2.test",
      region: "us",
    });
    await deleteProviderInstances("OpenAI", ["renamed"]);
    await testProviderConnection("OpenAI", {
      apiKey: "draft-key",
      baseUrl: "https://api.test",
      region: "eu",
    });
    await testProviderInstanceConnection("OpenAI", "renamed");
    await expect(
      getProviderInstanceBalance("OpenAI", "renamed"),
    ).resolves.toEqual({ currency: "USD", balance: 12 });
    await expect(listProviderTasks("OpenAI", "renamed")).resolves.toEqual([
      { id: "task-1", status: "done" },
    ]);
    await expect(
      getProviderTask("OpenAI", "renamed", "task-1"),
    ).resolves.toEqual([{ index: 0, content: "ok" }]);
    expect(calls).toEqual([
      "provider",
      "instance",
      "update",
      "delete",
      "draft-test",
    ]);
    expect(JSON.stringify(localStorage)).not.toContain("ephemeral-key");
    expect(JSON.stringify(localStorage)).not.toContain("replacement-key");
  });

  it("uses the VLLM runtime contract for custom OpenAI-compatible base URLs", async () => {
    const payloads: unknown[] = [];
    platformTestServer.use(
      http.post(
        "http://platform.test/api/v1/providers/VLLM/instances",
        async ({ request }) => {
          payloads.push(await request.json());
          return success();
        },
      ),
      http.put(
        "http://platform.test/api/v1/providers/VLLM/instances/custom",
        async ({ request }) => {
          payloads.push(await request.json());
          return success();
        },
      ),
      http.post(
        "http://platform.test/api/v1/providers/VLLM/connection",
        async ({ request }) => {
          payloads.push(await request.json());
          return success();
        },
      ),
    );

    await createProviderInstance("VLLM", {
      instanceName: "custom",
      apiKey: "temporary-key",
      baseUrl: "https://llm.example.test/",
    });
    await updateProviderInstance("VLLM", "custom", {
      instanceName: "custom",
      baseUrl: "https://llm.example.test/v1/",
    });
    await testProviderConnection("VLLM", {
      apiKey: "temporary-key",
      baseUrl: "https://llm.example.test",
    });

    expect(payloads).toEqual([
      {
        instance_name: "custom",
        api_key: "temporary-key",
        base_url: "https://llm.example.test/v1",
      },
      {
        instance_name: "custom",
        base_url: "https://llm.example.test/v1",
        region: "",
        verify: true,
      },
      {
        api_key: "temporary-key",
        base_url: "https://llm.example.test/v1",
        model_info: [],
      },
    ]);
    expect(JSON.stringify(localStorage)).not.toContain("temporary-key");
  });

  it("covers model CRUD and waits for confirmed default-model responses", async () => {
    const modelDto = {
      model_id: "model-1",
      name: "chat-model",
      provider_name: "OpenAI",
      instance_name: "primary",
      model_type: ["chat", "embedding"],
      status: "active",
    };
    const mutations: unknown[] = [];
    platformTestServer.use(
      http.get("http://platform.test/api/v1/models", () => success([modelDto])),
      http.get("http://platform.test/api/v1/models/default", () =>
        success({
          models: [
            {
              model_provider: "OpenAI",
              model_instance: "primary",
              model_name: "chat-model",
              model_id: "model-1",
              model_type: "chat",
              enable: true,
            },
          ],
        }),
      ),
      http.patch(
        "http://platform.test/api/v1/models/default",
        async ({ request }) => {
          mutations.push(await request.json());
          return success();
        },
      ),
      http.get(
        "http://platform.test/api/v1/providers/OpenAI/instances/primary/models",
        () => success([modelDto]),
      ),
      http.post(
        "http://platform.test/api/v1/providers/OpenAI/instances/primary/models",
        async ({ request }) => {
          mutations.push(await request.json());
          return success();
        },
      ),
      http.patch(
        "http://platform.test/api/v1/providers/OpenAI/instances/primary/models/chat-model",
        async ({ request }) => {
          mutations.push(await request.json());
          return success();
        },
      ),
      http.delete(
        "http://platform.test/api/v1/providers/OpenAI/instances/primary/models",
        async ({ request }) => {
          mutations.push(await request.json());
          return success();
        },
      ),
    );

    const [model] = await listTenantModels();
    expect(model.capabilities).toEqual(["chat", "embedding"]);
    await expect(getDefaultModels()).resolves.toMatchObject([
      { capability: "chat", modelId: "model-1" },
    ]);
    await setDefaultModel(model, "chat");
    await expect(listInstanceModels("OpenAI", "primary")).resolves.toHaveLength(
      1,
    );
    await addInstanceModel("OpenAI", "primary", {
      modelName: "chat-model",
      capabilities: ["chat"],
    });
    await updateInstanceModel("OpenAI", "primary", "chat-model", {
      status: "inactive",
    });
    await deleteInstanceModels("OpenAI", "primary", ["chat-model"]);
    expect(mutations).toEqual([
      {
        model_provider: "OpenAI",
        model_instance: "primary",
        model_name: "chat-model",
        model_id: "model-1",
        model_type: "chat",
      },
      {
        model_name: "chat-model",
        model_type: ["chat"],
        max_tokens: 8192,
        extra: {},
        models: [
          {
            model_name: "chat-model",
            model_types: ["chat"],
            max_tokens: 8192,
            max_dimension: 0,
            dimensions: [],
          },
        ],
      },
      { status: "inactive" },
      { model_name: ["chat-model"], models: ["chat-model"] },
    ]);
  });

  it("discovers remote instance models through the server-held credential", async () => {
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/providers/VLLM/instances/custom/models",
        ({ request }) => {
          expect(new URL(request.url).searchParams.get("supported")).toBe(
            "true",
          );
          return success([
            {
              name: "remote-chat-model",
              model_types: ["chat"],
              max_output: 4096,
            },
          ]);
        },
      ),
    );

    await expect(
      listSupportedInstanceModels("VLLM", "custom"),
    ).resolves.toEqual([
      expect.objectContaining({
        name: "remote-chat-model",
        capabilities: ["chat"],
        providerName: "VLLM",
        instanceName: "custom",
        maxTokens: 4096,
      }),
    ]);
  });

  it("covers every runtime-enabled utility and redacts full embedding vectors from the adapter result consumer", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/chat/to_model", () =>
        HttpResponse.json({
          code: 0,
          answer: "answer",
          reasoning_content: "reason",
          usage: { tokens: 2 },
        }),
      ),
      http.post("http://platform.test/api/v1/embeddings", () =>
        success([{ index: 0, token_count: 2, embedding: [0.1, 0.2, 0.3] }]),
      ),
      http.post("http://platform.test/api/v1/rerank", () =>
        success([{ index: 1, relevance_score: 0.9 }]),
      ),
      http.post("http://platform.test/api/v1/audio/transcriptions", () =>
        success({ text: "transcript" }),
      ),
      http.post("http://platform.test/api/v1/audio/speech", () =>
        success({ audio: btoa("audio") }),
      ),
      http.post("http://platform.test/api/v1/file/ocr", () =>
        success({ text: "ocr text" }),
      ),
      http.post("http://platform.test/api/v1/file/parse", () =>
        success({ task_id: "parse-task" }),
      ),
    );
    const model = { modelId: "model-1" };
    await expect(
      chatToModel(model, [{ role: "user", content: "hi" }]),
    ).resolves.toMatchObject({ answer: "answer", reasoning: "reason" });
    await expect(createEmbeddings(model, ["a"], 0)).resolves.toMatchObject([
      { vector: [0.1, 0.2, 0.3], tokenCount: 2 },
    ]);
    await expect(rerankDocuments(model, "q", ["a", "b"], 2)).resolves.toEqual([
      { index: 1, relevanceScore: 0.9 },
    ]);
    await expect(transcribeAudio(model, "YXVkaW8=", [])).resolves.toBe(
      "transcript",
    );
    await expect(synthesizeSpeech(model, "hello")).resolves.toMatchObject({
      size: 5,
      type: "audio/mpeg",
    });
    await expect(ocrFile(model, "aW1hZ2U=")).resolves.toBe("ocr text");
    await expect(parseFile(model, "ZmlsZQ==")).resolves.toBe("parse-task");
  });

  it("maps the pipeline list/detail contract and exposes abort errors", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/pipelines", ({ request }) => {
        expect(new URL(request.url).searchParams.get("type")).toBe("builtin");
        return success({
          canvas: [
            {
              id: "pipeline-1",
              title: "General",
              description: "Built in",
              filename: "general.json",
            },
          ],
          total: 1,
        });
      }),
      http.get("http://platform.test/api/v1/pipelines/pipeline-1", () =>
        success({ dsl: { components: [] } }),
      ),
      http.get(
        "http://platform.test/api/v1/providers/OpenAI/instances",
        async () => {
          await new Promise((resolve) => setTimeout(resolve, 50));
          return success([]);
        },
      ),
    );
    await expect(listPipelines()).resolves.toEqual([
      {
        id: "pipeline-1",
        title: "General",
        description: "Built in",
        filename: "general.json",
      },
    ]);
    await expect(getPipelineDsl("pipeline-1")).resolves.toEqual({
      components: [],
    });
    const controller = new AbortController();
    const pending = listProviderInstances("OpenAI", controller.signal);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ code: "CLIENT_ABORTED" });
  });

  it("surfaces utility contract errors and supports file-parse cancellation", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/chat/to_model", () =>
        HttpResponse.json({ code: 100, message: "model unavailable" }),
      ),
      http.post("http://platform.test/api/v1/file/ocr", () =>
        HttpResponse.json(
          { code: 500, data: null, message: "partial OCR failure" },
          { status: 500 },
        ),
      ),
      http.post("http://platform.test/api/v1/file/parse", async () => {
        await new Promise((resolve) => setTimeout(resolve, 50));
        return success({ task_id: "too-late" });
      }),
    );
    await expect(
      chatToModel({ modelId: "chat-1" }, [{ role: "user", content: "hi" }]),
    ).rejects.toMatchObject({ code: 100 });
    await expect(
      ocrFile({ modelId: "ocr-1" }, "aW1hZ2U="),
    ).rejects.toMatchObject({
      httpStatus: 500,
    });
    const controller = new AbortController();
    const pending = parseFile(
      { modelId: "parse-1" },
      "ZmlsZQ==",
      controller.signal,
    );
    controller.abort();
    await expect(pending).rejects.toMatchObject({ code: "CLIENT_ABORTED" });
  });
});
