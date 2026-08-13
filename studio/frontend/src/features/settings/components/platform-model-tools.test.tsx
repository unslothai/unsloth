import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { PlatformModel } from "@/integrations/platform-backend";
import { usePlatformSessionStore } from "@/integrations/platform-backend";
import { platformTestServer } from "@/integrations/platform-backend/__tests__/test-server";
import { PlatformModelTools } from "./platform-model-tools";
import { PlatformModelsSettings } from "./platform-models-settings";

const model = (capability: string): PlatformModel => ({
  id: `${capability}-1`,
  name: `${capability}-model`,
  providerId: "provider-1",
  providerName: "Provider",
  instanceId: "instance-1",
  instanceName: "primary",
  capabilities: [capability],
  status: "active",
  maxTokens: null,
});

function chooseTool(name: string) {
  fireEvent.click(screen.getByRole("combobox", { name: "Tool" }));
  fireEvent.click(screen.getByRole("option", { name }));
}

describe("PlatformModelTools", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    Object.defineProperty(Element.prototype, "scrollIntoView", {
      configurable: true,
      value: vi.fn(),
    });
  });
  afterEach(() => vi.unstubAllEnvs());

  it("keeps provider mutations and model tools unavailable to a read-only member", async () => {
    usePlatformSessionStore.getState().setUser({
      active: true,
      avatar: null,
      colorScheme: "",
      createdAt: null,
      email: "member@example.test",
      id: "user-1",
      language: "",
      loginChannel: "password",
      nickname: "Member",
      superuser: false,
      timezone: "",
      updatedAt: null,
    });
    platformTestServer.use(
      http.get("http://platform.test/api/v1/providers", ({ request }) =>
        HttpResponse.json({
          code: 0,
          data: new URL(request.url).searchParams.has("available")
            ? [{ id: "openai", name: "OpenAI" }]
            : [],
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/models", () =>
        HttpResponse.json({ code: 0, data: [], message: "success" }),
      ),
      http.get("http://platform.test/api/v1/models/default", () =>
        HttpResponse.json({
          code: 0,
          data: { models: [] },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/users/me/models", () =>
        HttpResponse.json({
          code: 0,
          data: { tenant_id: "tenant-1", name: "Workspace", role: "member" },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/pipelines", () =>
        HttpResponse.json({
          code: 0,
          data: { canvas: [], total: 0 },
          message: "success",
        }),
      ),
    );
    render(<PlatformModelsSettings mode="create" />);
    await screen.findByRole("combobox", { name: "Connection" });
    expect(screen.getByRole("combobox", { name: "Connection" })).toBeDisabled();
    expect(
      screen.queryByText("Yetkili model araçları"),
    ).not.toBeInTheDocument();
  });

  it("reveals instance credentials directly below the selected provider", async () => {
    usePlatformSessionStore.getState().setUser({
      active: true,
      avatar: null,
      colorScheme: "",
      createdAt: null,
      email: "owner@example.test",
      id: "owner-1",
      language: "",
      loginChannel: "password",
      nickname: "Owner",
      superuser: true,
      timezone: "",
      updatedAt: null,
    });
    platformTestServer.use(
      http.get("http://platform.test/api/v1/providers", ({ request }) =>
        HttpResponse.json({
          code: 0,
          data: new URL(request.url).searchParams.has("available")
            ? [{ id: "openai", name: "OpenAI" }]
            : [],
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/models", () =>
        HttpResponse.json({ code: 0, data: [], message: "success" }),
      ),
      http.get("http://platform.test/api/v1/models/default", () =>
        HttpResponse.json({
          code: 0,
          data: { models: [] },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/users/me/models", () =>
        HttpResponse.json({
          code: 0,
          data: { tenant_id: "tenant-1", name: "Workspace", role: "owner" },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/pipelines", () =>
        HttpResponse.json({
          code: 0,
          data: { canvas: [], total: 0 },
          message: "success",
        }),
      ),
    );

    render(<PlatformModelsSettings mode="create" />);

    const connection = await screen.findByRole("combobox", {
      name: "Connection",
    });
    expect(screen.queryByLabelText("Instance name")).not.toBeInTheDocument();
    fireEvent.click(connection);
    fireEvent.click(await screen.findByRole("option", { name: "OpenAI" }));

    expect(screen.getByLabelText("Instance name")).toBeVisible();
    expect(screen.getByLabelText("API key")).toBeVisible();
    expect(screen.getByLabelText("Base URL")).toBeVisible();
    expect(screen.queryByLabelText("Region")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Models ve varsayılanlar"),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("Pipeline kataloğu")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Yetkili model araçları"),
    ).not.toBeInTheDocument();
  });

  it("routes custom OpenAI-compatible connections through the supported VLLM provider", async () => {
    usePlatformSessionStore.getState().setUser({
      active: true,
      avatar: null,
      colorScheme: "",
      createdAt: null,
      email: "owner@example.test",
      id: "owner-1",
      language: "",
      loginChannel: "password",
      nickname: "Owner",
      superuser: true,
      timezone: "",
      updatedAt: null,
    });
    const connectionProbe = vi.fn();
    platformTestServer.use(
      http.get("http://platform.test/api/v1/providers", ({ request }) =>
        HttpResponse.json({
          code: 0,
          data: new URL(request.url).searchParams.has("available")
            ? [
                { id: "vllm", name: "VLLM" },
                { id: "anthropic", name: "Anthropic" },
                {
                  id: "openai-compatible",
                  name: "OpenAI-API-Compatible",
                },
              ]
            : [],
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/models", () =>
        HttpResponse.json({ code: 0, data: [], message: "success" }),
      ),
      http.get("http://platform.test/api/v1/models/default", () =>
        HttpResponse.json({
          code: 0,
          data: { models: [] },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/users/me/models", () =>
        HttpResponse.json({
          code: 0,
          data: { tenant_id: "tenant-1", name: "Workspace", role: "owner" },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/pipelines", () =>
        HttpResponse.json({ code: 404, message: "not found" }),
      ),
      http.post("http://platform.test/api/v1/providers/VLLM/connection", () => {
        connectionProbe();
        return HttpResponse.json({ code: 0 });
      }),
    );

    render(<PlatformModelsSettings mode="create" />);

    const connection = await screen.findByRole("combobox", {
      name: "Connection",
    });
    fireEvent.click(connection);
    const anthropic = await screen.findByRole("option", { name: "Anthropic" });
    expect(
      anthropic.querySelector('img[src$="provider-logos/anthropic.svg"]'),
    ).not.toBeNull();
    expect(
      screen.queryByRole("option", { name: /OpenAI-API-Compatible/ }),
    ).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("option", {
        name: "OpenAI compatible / Custom (VLLM)",
      }),
    );

    expect(connection).toHaveTextContent("OpenAI compatible / Custom (VLLM)");
    expect(
      connection.querySelector('img[src$="provider-logos/vllm.svg"]'),
    ).not.toBeNull();
    expect(screen.getByText(/Base URL otomatik olarak \/v1/)).toBeVisible();
    expect(screen.getByLabelText("Base URL")).toHaveAttribute(
      "placeholder",
      "https://llm.example.com/v1",
    );

    fireEvent.change(screen.getByLabelText("Instance name"), {
      target: { value: "custom" },
    });
    const apiKey = screen.getByLabelText("API key");
    fireEvent.change(apiKey, { target: { value: "temporary-key" } });
    fireEvent.change(screen.getByLabelText("Base URL"), {
      target: { value: "https://llm.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));

    await waitFor(() => expect(connectionProbe).toHaveBeenCalledTimes(1));
    expect(apiKey).toHaveValue("temporary-key");
    expect(
      screen.getByRole("button", { name: "Add connection" }),
    ).toBeEnabled();
  });

  it("uses service-backed dropdowns for models, capabilities and defaults", async () => {
    usePlatformSessionStore.getState().setUser({
      active: true,
      avatar: null,
      colorScheme: "",
      createdAt: null,
      email: "owner@example.test",
      id: "owner-1",
      language: "",
      loginChannel: "password",
      nickname: "Owner",
      superuser: true,
      timezone: "",
      updatedAt: null,
    });
    const modelDto = {
      model_id: "model-1",
      name: "text-model@revision",
      provider_name: "OpenAI",
      instance_name: "primary",
      model_type: ["chat", "embedding", "rerank"],
      status: "active",
    };
    const saveDefault = vi.fn();
    const changeModelStatus = vi.fn();
    const deleteModel = vi.fn();
    let defaultAssigned = false;
    let storedModelStatus = "active";
    let modelDeleted = false;
    platformTestServer.use(
      http.get("http://platform.test/api/v1/providers", ({ request }) =>
        HttpResponse.json({
          code: 0,
          data: new URL(request.url).searchParams.has("available")
            ? [{ id: "openai", name: "OpenAI" }]
            : [{ id: "openai", name: "OpenAI" }],
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/providers/OpenAI/instances", () =>
        HttpResponse.json({
          code: 0,
          data: [
            {
              id: "instance-1",
              instance_name: "primary",
              provider_id: "openai",
            },
          ],
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/providers/OpenAI/models", () =>
        HttpResponse.json({ code: 0, data: [modelDto], message: "success" }),
      ),
      http.get(
        "http://platform.test/api/v1/providers/OpenAI/models/text-model%40revision",
        () =>
          HttpResponse.json({ code: 0, data: modelDto, message: "success" }),
      ),
      http.get(
        "http://platform.test/api/v1/providers/OpenAI/instances/primary/models",
        ({ request }) => {
          const supported =
            new URL(request.url).searchParams.get("supported") === "true";
          const data = supported
            ? [modelDto]
            : modelDeleted || storedModelStatus === "inactive"
              ? []
              : [
                  {
                    ...modelDto,
                    model_id: "model-chat",
                    model_type: ["chat"],
                    status: storedModelStatus,
                  },
                  {
                    ...modelDto,
                    model_id: "model-embedding",
                    model_type: ["embedding", "rerank"],
                    status: storedModelStatus,
                  },
                ];
          return HttpResponse.json({ code: 0, data, message: "success" });
        },
      ),
      http.get("http://platform.test/api/v1/models", () =>
        HttpResponse.json({
          code: 0,
          data:
            modelDeleted || storedModelStatus === "inactive" ? [] : [modelDto],
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/models/default", () =>
        HttpResponse.json({
          code: 0,
          // A pinned runtime can persist the tenant selector but omit this row.
          data: { models: [] },
          message: "success",
        }),
      ),
      http.patch(
        "http://platform.test/api/v1/models/default",
        async ({ request }) => {
          saveDefault(await request.json());
          defaultAssigned = true;
          return HttpResponse.json({ code: 0, message: "success" });
        },
      ),
      http.patch(
        "http://platform.test/api/v1/providers/OpenAI/instances/primary/models/text-model%40revision",
        async ({ request }) => {
          const body = (await request.json()) as { status: string };
          changeModelStatus(body);
          storedModelStatus = body.status;
          return HttpResponse.json({ code: 0, message: "success" });
        },
      ),
      http.delete(
        "http://platform.test/api/v1/providers/OpenAI/instances/primary/models",
        async ({ request }) => {
          deleteModel(await request.json());
          modelDeleted = true;
          return HttpResponse.json({ code: 0, message: "success" });
        },
      ),
      http.get("http://platform.test/api/v1/users/me/models", () =>
        HttpResponse.json({
          code: 0,
          data: {
            tenant_id: "tenant-1",
            name: "Workspace",
            role: "owner",
            llm_id: defaultAssigned ? "text-model@revision@primary@OpenAI" : "",
          },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/pipelines", () =>
        HttpResponse.json({
          code: 0,
          data: { canvas: [], total: 0 },
          message: "success",
        }),
      ),
    );

    const rendered = render(<PlatformModelsSettings mode="manage" />);

    fireEvent.click(await screen.findByText("primary"));

    expect(
      screen.getByRole("heading", { name: "Model ekle" }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Ekli modeller" }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Varsayılan modeller" }),
    ).toBeVisible();

    const modelSelect = await screen.findByRole("combobox", {
      name: "Model name",
    });
    expect(screen.getByTestId("platform-connection-details")).toHaveClass(
      "space-y-4",
      "py-5",
      "sm:py-6",
    );
    await waitFor(() => expect(modelSelect).toBeEnabled());
    fireEvent.click(modelSelect);
    fireEvent.click(
      await screen.findByRole("option", { name: "text-model@revision" }),
    );

    expect(modelSelect).toHaveTextContent("text-model@revision");
    expect(screen.getAllByText("1 model")).toHaveLength(2);
    expect(screen.queryByText("Rag Platform")).not.toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: "Capability" }),
    ).toHaveTextContent("chat");
    const chatDefault = screen.getByRole("combobox", {
      name: "chat varsayılanı",
    });
    fireEvent.click(chatDefault);
    fireEvent.click(
      await screen.findByRole("option", {
        name: "text-model@revision — OpenAI",
      }),
    );
    await waitFor(() =>
      expect(chatDefault).toHaveTextContent("text-model@revision — OpenAI"),
    );
    expect(saveDefault).toHaveBeenCalledWith({
      model_provider: "OpenAI",
      model_instance: "primary",
      model_name: "text-model@revision",
      model_id: "model-1",
      model_type: "chat",
    });

    const embeddingDefault = screen.getByRole("combobox", {
      name: "embedding varsayılanı",
    });
    fireEvent.click(embeddingDefault);
    fireEvent.click(
      await screen.findByRole("option", {
        name: "text-model@revision — OpenAI",
      }),
    );
    expect(
      await screen.findByRole("alertdialog", {
        name: "Embedding varsayılanını değiştir?",
      }),
    ).toBeVisible();
    expect(
      screen.getByText(
        "Embedding varsayılanını değiştirmek mevcut dataset indeksleriyle uyumsuzluk yaratabilir. Değişiklikten sonra mevcut dataset’leri yeniden indekslemeniz gerekebilir.",
      ),
    ).toBeVisible();
    expect(saveDefault).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Vazgeç" }));
    expect(
      screen.queryByRole("alertdialog", {
        name: "Embedding varsayılanını değiştir?",
      }),
    ).not.toBeInTheDocument();
    expect(saveDefault).toHaveBeenCalledTimes(1);

    fireEvent.click(embeddingDefault);
    fireEvent.click(
      await screen.findByRole("option", {
        name: "text-model@revision — OpenAI",
      }),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Varsayılanı değiştir" }),
    );
    await waitFor(() =>
      expect(saveDefault).toHaveBeenLastCalledWith({
        model_provider: "OpenAI",
        model_instance: "primary",
        model_name: "text-model@revision",
        model_id: "model-1",
        model_type: "embedding",
      }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("alertdialog", {
          name: "Embedding varsayılanını değiştir?",
        }),
      ).not.toBeInTheDocument(),
    );

    // The saved selection must also survive a full component remount even
    // when /models/default remains empty. Model names may contain `@`.
    rendered.unmount();
    render(<PlatformModelsSettings mode="manage" />);
    fireEvent.click(await screen.findByText("primary"));
    expect(
      await screen.findByRole("combobox", { name: "chat varsayılanı" }),
    ).toHaveTextContent("text-model@revision — OpenAI");

    fireEvent.click(screen.getByRole("button", { name: "Devre dışı bırak" }));
    await waitFor(() => expect(screen.getByText("Devre dışı")).toBeVisible());
    // The active hybrid runtime may omit inactive models from the refreshed
    // instance inventory. The row must remain available for reactivation.
    expect(screen.getByText("text-model@revision")).toBeVisible();
    expect(screen.getByRole("button", { name: "Etkinleştir" })).toBeVisible();
    expect(changeModelStatus).toHaveBeenLastCalledWith({ status: "inactive" });

    fireEvent.click(screen.getByRole("button", { name: "Etkinleştir" }));
    await waitFor(() => expect(screen.getByText("Etkin")).toBeVisible());
    expect(changeModelStatus).toHaveBeenLastCalledWith({ status: "active" });

    fireEvent.click(
      screen.getByRole("button", {
        name: "text-model@revision modelini kaldır",
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Modeli kaldır" }));
    expect(await screen.findByText("Henüz model eklenmedi")).toBeVisible();
    expect(deleteModel).toHaveBeenCalledWith({
      model_name: ["text-model@revision"],
      models: ["text-model@revision"],
    });
    expect(
      screen.getByRole("combobox", { name: "embedding varsayılanı" }),
    ).toBeVisible();
    expect(
      screen.getByRole("combobox", { name: "rerank varsayılanı" }),
    ).toBeVisible();
  }, 10_000);

  it("keeps configured instances usable when hybrid provider catalog enrichment fails", async () => {
    usePlatformSessionStore.getState().setUser({
      active: true,
      avatar: null,
      colorScheme: "",
      createdAt: null,
      email: "owner@example.test",
      id: "owner-1",
      language: "",
      loginChannel: "password",
      nickname: "Owner",
      superuser: true,
      timezone: "",
      updatedAt: null,
    });
    const providerName = "OpenAI-API-Compatible";
    const modelDto = {
      model_id: "custom-model-1",
      name: "custom-chat-model",
      provider_name: providerName,
      instance_name: "primary",
      model_type: ["chat"],
      status: "active",
    };
    platformTestServer.use(
      http.get("http://platform.test/api/v1/providers", ({ request }) =>
        HttpResponse.json({
          code: 0,
          data: new URL(request.url).searchParams.has("available")
            ? [{ id: providerName, name: providerName }]
            : [{ id: providerName, name: providerName }],
          message: "success",
        }),
      ),
      http.get(
        `http://platform.test/api/v1/providers/${providerName}/instances`,
        () =>
          HttpResponse.json({
            code: 0,
            data: [
              {
                id: "instance-1",
                instance_name: "primary",
                provider_id: providerName,
              },
            ],
            message: "success",
          }),
      ),
      http.get(
        `http://platform.test/api/v1/providers/${providerName}/models`,
        () =>
          HttpResponse.json({
            code: 404,
            message: `provider '${providerName}' not found`,
          }),
      ),
      http.get(
        `http://platform.test/api/v1/providers/${providerName}/instances/primary/models`,
        () =>
          HttpResponse.json({ code: 0, data: [modelDto], message: "success" }),
      ),
      http.get("http://platform.test/api/v1/models", () =>
        HttpResponse.json({ code: 0, data: [modelDto], message: "success" }),
      ),
      http.get("http://platform.test/api/v1/models/default", () =>
        HttpResponse.json({
          code: 0,
          data: { models: [] },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/users/me/models", () =>
        HttpResponse.json({
          code: 0,
          data: { tenant_id: "tenant-1", name: "Workspace", role: "owner" },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/pipelines", () =>
        HttpResponse.json({ code: 404, message: "not found" }),
      ),
    );

    render(<PlatformModelsSettings mode="manage" />);

    fireEvent.click(await screen.findByText("primary"));
    expect(await screen.findByText("Models ve varsayılanlar")).toBeVisible();
    expect(
      screen.queryByText(`provider '${providerName}' not found`),
    ).not.toBeInTheDocument();
    const modelSelect = screen.getByRole("combobox", { name: "Model name" });
    await waitFor(() => expect(modelSelect).toBeEnabled());
    fireEvent.click(modelSelect);
    expect(
      await screen.findByRole("option", { name: "custom-chat-model" }),
    ).toBeVisible();
  });

  it("tests a saved connection and discovers its remote models end to end", async () => {
    usePlatformSessionStore.getState().setUser({
      active: true,
      avatar: null,
      colorScheme: "",
      createdAt: null,
      email: "owner@example.test",
      id: "owner-1",
      language: "",
      loginChannel: "password",
      nickname: "Owner",
      superuser: true,
      timezone: "",
      updatedAt: null,
    });
    const addModel = vi.fn();
    platformTestServer.use(
      http.get("http://platform.test/api/v1/providers", ({ request }) =>
        HttpResponse.json({
          code: 0,
          data: new URL(request.url).searchParams.has("available")
            ? [{ id: "vllm", name: "VLLM" }]
            : [{ id: "vllm", name: "VLLM" }],
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/providers/VLLM/instances", () =>
        HttpResponse.json({
          code: 0,
          data: [
            {
              id: "instance-1",
              instance_name: "custom",
              provider_id: "vllm",
              api_key: { configured: true },
              base_url: "https://llm.example.test/v1",
            },
          ],
          message: "success",
        }),
      ),
      http.get(
        "http://platform.test/api/v1/providers/VLLM/instances/custom",
        () =>
          HttpResponse.json({
            code: 0,
            data: {
              id: "instance-1",
              instance_name: "custom",
              provider_id: "vllm",
              api_key: { configured: true },
              base_url: "https://llm.example.test/v1",
            },
            message: "success",
          }),
      ),
      http.get("http://platform.test/api/v1/providers/VLLM/models", () =>
        HttpResponse.json({ code: 0, data: [], message: "success" }),
      ),
      http.get(
        "http://platform.test/api/v1/providers/VLLM/instances/custom/models",
        ({ request }) =>
          HttpResponse.json({
            code: 0,
            data: new URL(request.url).searchParams.has("supported")
              ? [
                  {
                    name: "remote-chat-model",
                    model_types: ["chat"],
                  },
                  {
                    name: "remote-embedding-model",
                    model_types: ["embedding"],
                  },
                ]
              : [],
            message: "success",
          }),
      ),
      http.get(
        "http://platform.test/api/v1/providers/VLLM/instances/custom/connection",
        () => HttpResponse.json({ code: 0, message: "success" }),
      ),
      http.post(
        "http://platform.test/api/v1/providers/VLLM/instances/custom/models",
        async ({ request }) => {
          addModel(await request.json());
          return HttpResponse.json({ code: 0, message: "success" });
        },
      ),
      http.get("http://platform.test/api/v1/models", () =>
        HttpResponse.json({ code: 0, data: [], message: "success" }),
      ),
      http.get("http://platform.test/api/v1/models/default", () =>
        HttpResponse.json({
          code: 0,
          data: { models: [] },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/users/me/models", () =>
        HttpResponse.json({
          code: 0,
          data: { tenant_id: "tenant-1", name: "Workspace", role: "owner" },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/pipelines", () =>
        HttpResponse.json({
          code: 0,
          data: { canvas: [], total: 0 },
          message: "success",
        }),
      ),
    );

    render(<PlatformModelsSettings mode="manage" />);

    fireEvent.click(await screen.findByText("custom"));

    const modelSelect = await screen.findByRole("combobox", {
      name: "Model name",
    });
    await waitFor(() => expect(modelSelect).toBeEnabled());
    fireEvent.click(modelSelect);
    fireEvent.click(
      await screen.findByRole("option", { name: "remote-chat-model" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Model ekle" }));
    await waitFor(() =>
      expect(addModel).toHaveBeenCalledWith({
        model_name: "remote-chat-model",
        model_type: ["chat"],
        max_tokens: 8192,
        extra: {},
        models: [
          {
            model_name: "remote-chat-model",
            model_types: ["chat"],
            max_tokens: 8192,
            max_dimension: 0,
            dimensions: [],
          },
        ],
      }),
    );

    expect(screen.queryByText("Bakiye")).not.toBeInTheDocument();
    expect(screen.queryByText("Task’ler")).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "custom bağlantısını test et" }),
    );
    expect(
      await screen.findByText("Bağlantı doğrulandı; 2 model bulundu."),
    ).toBeVisible();

    fireEvent.click(
      screen.getByRole("button", { name: "custom bağlantısını düzenle" }),
    );
    expect(await screen.findByText("Bağlantıyı düzenle")).toBeVisible();
    expect(screen.getByLabelText("Instance name")).toHaveValue("custom");
    expect(screen.getByLabelText("Base URL")).toHaveValue(
      "https://llm.example.test/v1",
    );
    expect(screen.getByLabelText("API key")).toHaveValue("");
  });

  it("disables a utility with an explicit capability reason", () => {
    render(<PlatformModelTools models={[model("chat")]} />);
    chooseTool("OCR");
    expect(screen.getByRole("status")).toHaveTextContent(
      "ocr capability’sine sahip",
    );
    expect(screen.getByRole("button", { name: "Çalıştır" })).toBeDisabled();
  });

  it("shows only embedding dimensions and an eight-value sample", async () => {
    platformTestServer.use(
      http.post("http://platform.test/api/v1/embeddings", () =>
        HttpResponse.json({
          code: 0,
          data: [
            {
              index: 0,
              token_count: 4,
              embedding: [0, 1, 2, 3, 4, 5, 6, 7, 888, 999],
            },
          ],
          message: "success",
        }),
      ),
    );
    render(<PlatformModelTools models={[model("embedding")]} />);
    chooseTool("Embedding");
    fireEvent.change(screen.getByLabelText("Araç girdisi"), {
      target: { value: "hello" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));
    expect(await screen.findByText(/dimension=10/)).toHaveTextContent(
      "sample=[0, 1, 2, 3, 4, 5, 6, 7, …]",
    );
    expect(screen.queryByText(/888/)).not.toBeInTheDocument();
    expect(screen.queryByText(/999/)).not.toBeInTheDocument();
  });

  it("rejects oversize files before network and revokes speech object URLs", async () => {
    const request = vi.fn();
    const createObjectURL = vi.fn(() => "blob:temporary-audio");
    const revokeObjectURL = vi.fn();
    const originalCreateObjectURL = URL.createObjectURL;
    const originalRevokeObjectURL = URL.revokeObjectURL;
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: createObjectURL,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: revokeObjectURL,
    });
    platformTestServer.use(
      http.post("http://platform.test/api/v1/audio/speech", () => {
        request();
        return HttpResponse.json({
          code: 0,
          data: { audio: btoa("audio") },
          message: "success",
        });
      }),
    );

    const rendered = render(
      <PlatformModelTools models={[model("tts"), model("ocr")]} />,
    );
    chooseTool("OCR");
    const oversized = new File(["x"], "large.png", { type: "image/png" });
    Object.defineProperty(oversized, "size", { value: 11 * 1024 * 1024 });
    fireEvent.change(screen.getByLabelText("Araç dosyası"), {
      target: { files: [oversized] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));
    expect(screen.getByRole("alert")).toHaveTextContent("10 MB");
    expect(request).not.toHaveBeenCalled();

    chooseTool("Audio speech");
    fireEvent.change(screen.getByLabelText("Araç girdisi"), {
      target: { value: "hello" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Çalıştır" }));
    await waitFor(() => expect(createObjectURL).toHaveBeenCalledTimes(1));
    rendered.unmount();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:temporary-audio");
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: originalCreateObjectURL,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: originalRevokeObjectURL,
    });
  });
});
