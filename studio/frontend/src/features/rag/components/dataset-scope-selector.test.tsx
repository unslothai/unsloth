import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { platformTestServer } from "@/integrations/platform-backend/__tests__/test-server";
import { DatasetScopeSelector } from "./dataset-scope-selector";

describe("Phase 7 project dataset scope selector", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
  });

  afterEach(() => vi.unstubAllEnvs());

  it("renders loading data and emits the selected backend dataset ids", async () => {
    const onChange = vi.fn();
    platformTestServer.use(
      http.get("http://platform.test/api/v1/datasets", () =>
        HttpResponse.json({
          code: 0,
          data: [
            {
              id: "dataset-1",
              name: "Product docs",
              document_count: 3,
              permission: "me",
            },
          ],
          total_datasets: 1,
        }),
      ),
    );

    render(
      <DatasetScopeSelector selectedIds={[]} onChange={onChange} />,
    );
    expect(screen.getByText("Loading datasets…")).toBeInTheDocument();
    const checkbox = await screen.findByRole("checkbox", {
      name: "Use dataset Product docs",
    });
    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalledWith(["dataset-1"]);
  });

  it("surfaces permission/network failures and retries", async () => {
    let attempts = 0;
    platformTestServer.use(
      http.get("http://platform.test/api/v1/datasets", () => {
        attempts += 1;
        if (attempts === 1) {
          return HttpResponse.json(
            { code: 401, message: "Dataset access denied" },
            { status: 403 },
          );
        }
        return HttpResponse.json({
          code: 0,
          data: [],
          total_datasets: 0,
        });
      }),
    );

    render(
      <DatasetScopeSelector selectedIds={[]} onChange={vi.fn()} />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Dataset access denied",
    );
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => {
      expect(screen.getByText(/No datasets are available/)).toBeInTheDocument();
    });
  });
});
