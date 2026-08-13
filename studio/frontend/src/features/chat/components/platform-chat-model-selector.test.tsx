import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { platformTestServer } from "@/integrations/platform-backend/__tests__/test-server";
import { PlatformChatModelSelector } from "./platform-chat-model-selector";

describe("PlatformChatModelSelector", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
  });

  afterEach(() => vi.unstubAllEnvs());

  it("shows only active connected chat models and persists the selected default", async () => {
    let defaultPayload: Record<string, unknown> | undefined;
    platformTestServer.use(
      http.get("http://platform.test/api/v1/models", () =>
        HttpResponse.json({
          code: 0,
          data: [
            {
              model_id: "claude-sonnet",
              model_name: "Claude Sonnet",
              model_type: ["chat"],
              provider_name: "Anthropic",
              instance_name: "production",
              status: "active",
              max_tokens: 200000,
            },
            {
              model_id: "custom-chat",
              model_name: "Custom Chat",
              model_type: ["chat"],
              provider_name: "VLLM",
              instance_name: "medyasoft",
              status: "active",
            },
            {
              model_id: "embedding-1",
              model_name: "Embedding Model",
              model_type: ["embedding"],
              provider_name: "VLLM",
              instance_name: "medyasoft",
              status: "active",
            },
            {
              model_id: "disabled-chat",
              model_name: "Disabled Chat",
              model_type: ["chat"],
              provider_name: "VLLM",
              instance_name: "medyasoft",
              status: "inactive",
            },
          ],
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/models/default", () =>
        HttpResponse.json({
          code: 0,
          data: {
            models: [
              {
                enable: true,
                model_id: "claude-sonnet",
                model_instance: "production",
                model_name: "Claude Sonnet",
                model_provider: "Anthropic",
                model_type: "chat",
              },
            ],
          },
          message: "success",
        }),
      ),
      http.get("http://platform.test/api/v1/users/me/models", () =>
        HttpResponse.json({
          code: 0,
          data: {
            tenant_id: "tenant-1",
            name: "Workspace",
            role: "owner",
          },
          message: "success",
        }),
      ),
      http.patch(
        "http://platform.test/api/v1/models/default",
        async ({ request }) => {
          defaultPayload = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({ code: 0, data: {}, message: "success" });
        },
      ),
    );

    render(<PlatformChatModelSelector />);

    expect(
      await screen.findByRole("button", {
        name: "Chat model: Claude Sonnet — Anthropic",
      }),
    ).toBeVisible();
    fireEvent.click(
      screen.getByRole("button", {
        name: "Chat model: Claude Sonnet — Anthropic",
      }),
    );

    expect(await screen.findByText("Connected models")).toBeVisible();
    expect(screen.getAllByText("Claude Sonnet")).toHaveLength(2);
    expect(screen.getByText("Custom Chat")).toBeVisible();
    expect(screen.queryByText("Embedding Model")).not.toBeInTheDocument();
    expect(screen.queryByText("Disabled Chat")).not.toBeInTheDocument();
    expect(screen.queryByText("Recommended")).not.toBeInTheDocument();
    expect(screen.queryByText("On Device")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", {
        name: "Custom Chat — VLLM — medyasoft",
      }),
    );

    await waitFor(() =>
      expect(defaultPayload).toMatchObject({
        model_id: "custom-chat",
        model_instance: "medyasoft",
        model_name: "Custom Chat",
        model_provider: "VLLM",
        model_type: "chat",
      }),
    );
    expect(
      screen.getByRole("button", {
        name: "Chat model: Custom Chat — VLLM",
      }),
    ).toBeVisible();
  });

  it("links an empty model list back to Connections", async () => {
    platformTestServer.use(
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
          data: {
            tenant_id: "tenant-1",
            name: "Workspace",
            role: "owner",
          },
          message: "success",
        }),
      ),
    );

    render(<PlatformChatModelSelector />);
    const trigger = await screen.findByRole("button", { name: "Select model" });
    fireEvent.click(trigger);

    expect(await screen.findByText("No active chat models")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Open Connections" }),
    ).toBeVisible();
  });
});
