import { HttpResponse, delay, http } from "msw";
import { describe, expect, it } from "vitest";

import cleanupFixtureText from "../../../../../../docs/rag-platform/fixtures/cleanup.json?raw";
import streamFixtureText from "../../../../../../docs/rag-platform/fixtures/stream.json?raw";
import { platformRequest } from "../client";
import type { PlatformBackendConfig } from "../config";
import { PlatformApiError } from "../errors";
import { platformTestServer } from "./test-server";

const config: PlatformBackendConfig = {
  enabled: true,
  baseUrl: "http://platform.test",
  apiPrefix: "/api/v1",
  proxyTarget: "",
  requestTimeoutMs: 1_000,
};

const cleanupFixture = JSON.parse(cleanupFixtureText) as {
  interactions: Array<{ response: { body: Record<string, unknown> } }>;
};
const streamFixture = JSON.parse(streamFixtureText) as {
  interactions: Array<{ response: { body: Record<string, unknown> } }>;
};

function request<T>(
  endpoint: string,
  options: Parameters<typeof platformRequest<T>>[1] = {},
) {
  return platformRequest<T>(endpoint, { ...options, config });
}

describe("platformRequest", () => {
  it("unwraps a real Faz 0 success envelope", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/success", () =>
        HttpResponse.json(cleanupFixture.interactions[0]?.response.body),
      ),
    );

    await expect(
      request<{ success_count: number }>("/success"),
    ).resolves.toEqual({
      success_count: 1,
    });
  });

  it("treats code !== 0 inside HTTP 200 as a typed error", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/business-error", () =>
        HttpResponse.json(streamFixture.interactions[0]?.response.body),
      ),
    );

    await expect(request("/business-error")).rejects.toMatchObject({
      name: "PlatformApiError",
      httpStatus: 200,
      code: 100,
      endpoint: "/business-error",
    });
  });

  it.each([401, 403, 404, 500])(
    "normalizes HTTP %s responses",
    async (status) => {
      platformTestServer.use(
        http.get("http://platform.test/api/v1/http-error", () =>
          HttpResponse.json(
            { code: status, message: `failure-${status}`, data: null },
            { status, headers: { "x-request-id": "request-123" } },
          ),
        ),
      );

      await expect(
        request("/http-error", { getRetries: 0 }),
      ).rejects.toMatchObject({
        httpStatus: status,
        code: status,
        requestId: "request-123",
      });
    },
  );

  it("does not expose a non-JSON gateway body", async () => {
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/gateway",
        () => new HttpResponse("<html>upstream failed</html>", { status: 502 }),
      ),
    );

    const error = await request("/gateway", { getRetries: 0 }).catch(
      (caught: unknown) => caught,
    );
    expect(error).toBeInstanceOf(PlatformApiError);
    expect(error).toMatchObject({ code: "HTTP_502", httpStatus: 502 });
    expect((error as Error).message).not.toContain("<html>");
  });

  it("handles 204 and empty successful responses", async () => {
    platformTestServer.use(
      http.get(
        "http://platform.test/api/v1/no-content",
        () => new HttpResponse(null, { status: 204 }),
      ),
      http.get(
        "http://platform.test/api/v1/empty",
        () => new HttpResponse(null, { status: 200 }),
      ),
    );

    await expect(request("/no-content")).resolves.toBeUndefined();
    await expect(request("/empty")).resolves.toBeUndefined();
  });

  it("supports external abort and timeout as distinct errors", async () => {
    platformTestServer.use(
      http.get("http://platform.test/api/v1/slow", async () => {
        await delay(100);
        return HttpResponse.json({ code: 0, data: true });
      }),
    );

    const controller = new AbortController();
    const aborted = request("/slow", { signal: controller.signal });
    controller.abort();
    await expect(aborted).rejects.toMatchObject({ code: "CLIENT_ABORTED" });

    await expect(request("/slow", { timeoutMs: 5 })).rejects.toMatchObject({
      code: "CLIENT_TIMEOUT",
    });
  });

  it("lets the browser set multipart Content-Type and supports raw bodies/blobs", async () => {
    let multipartContentType: string | null = null;
    let rawBody = "";
    platformTestServer.use(
      http.post("http://platform.test/api/v1/upload", ({ request }) => {
        multipartContentType = request.headers.get("content-type");
        return HttpResponse.json({ code: 0, data: true });
      }),
      http.post("http://platform.test/api/v1/raw", async ({ request }) => {
        rawBody = await request.text();
        return new HttpResponse("download", {
          headers: { "content-type": "application/octet-stream" },
        });
      }),
    );

    const form = new FormData();
    form.append("file", new Blob(["upload"]), "upload.txt");
    await request("/upload", { method: "POST", body: form });
    expect(multipartContentType).toMatch(/^multipart\/form-data; boundary=/);

    const blob = await request<Blob>("/raw", {
      method: "POST",
      body: "payload",
      responseType: "blob",
    });
    expect(rawBody).toBe("payload");
    await expect(blob.text()).resolves.toBe("download");
  });

  it("adds bearer/query values without retrying mutations", async () => {
    let mutationCount = 0;
    platformTestServer.use(
      http.post("http://platform.test/api/v1/mutate", ({ request }) => {
        mutationCount += 1;
        const url = new URL(request.url);
        expect(url.searchParams.getAll("tag")).toEqual(["a", "b"]);
        expect(request.headers.get("authorization")).toBe(
          "Bearer opaque-token",
        );
        return HttpResponse.json(
          { code: 503, data: null, message: "busy" },
          { status: 503 },
        );
      }),
    );

    await expect(
      request("/mutate", {
        method: "POST",
        token: "opaque-token",
        query: { tag: ["a", "b"] },
        json: { value: true },
        getRetries: 2,
      }),
    ).rejects.toMatchObject({ httpStatus: 503 });
    expect(mutationCount).toBe(1);
  });

  it("retries a retryable GET only within the configured bound", async () => {
    let requestCount = 0;
    platformTestServer.use(
      http.get("http://platform.test/api/v1/retry", () => {
        requestCount += 1;
        if (requestCount === 1) {
          return HttpResponse.json(
            { code: 503, data: null, message: "busy" },
            { status: 503 },
          );
        }
        return HttpResponse.json({ code: 0, data: "ok", message: "success" });
      }),
    );

    await expect(request<string>("/retry", { getRetries: 1 })).resolves.toBe(
      "ok",
    );
    expect(requestCount).toBe(2);
  });
});
