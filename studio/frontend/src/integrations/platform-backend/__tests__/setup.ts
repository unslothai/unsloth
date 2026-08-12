import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterAll, afterEach, beforeAll } from "vitest";

import { platformTestServer } from "./test-server";

beforeAll(() => platformTestServer.listen({ onUnhandledRequest: "error" }));
afterEach(() => {
  cleanup();
  platformTestServer.resetHandlers();
});
afterAll(() => platformTestServer.close());
