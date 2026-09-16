// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness page for tests/studio/playwright_link_definition_probe.py: three settled replies
// through the real Thread, selected by ?case=.

/* eslint-disable no-restricted-imports -- a measurement entry point, not app code. */
import "@/features/chat/stores/sidebar-organization-store";
/* eslint-enable no-restricted-imports */

import { Thread } from "@/components/assistant-ui/thread";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  AssistantRuntimeProvider,
  type ChatModelAdapter,
  ExportedMessageRepository,
  type ThreadMessageLike,
  useAui,
  useLocalRuntime,
} from "@assistant-ui/react";
import {
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRouter,
} from "@tanstack/react-router";
import { type ReactElement, useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url =
    typeof input === "string"
      ? input
      : ((input as Request).url ?? String(input));
  if (url.includes("/api/")) {
    return Promise.resolve(
      new Response("{}", {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
  }
  return realFetch(input, init);
};

const CODE_REPLY = [
  "Here is the shape you asked for.",
  "",
  "```ts",
  "interface Grid {",
  "  [key: string]: number[][];",
  "}",
  "",
  "const cell = grid[row][col];",
  "```",
  "",
  "And the helper that reads it.",
  "",
  "```python",
  "def read(grid, row, col):",
  "    return grid[row][col]",
  "```",
  "",
  "That is all.",
].join("\n");

const LINK_REPLY = [
  "See the [handbook][hb] and the [spec][sp] for details.",
  "",
  "```ts",
  "const x = 1;",
  "```",
  "",
  "[hb]: https://example.com/handbook",
  "[sp]: https://example.com/spec",
].join("\n");

const PLAIN_REPLY = [
  "A plain fence with no brackets.",
  "",
  "```python",
  "print('hello')",
  "```",
].join("\n");

const CASES: Record<string, string> = {
  code: CODE_REPLY,
  link: LINK_REPLY,
  plain: PLAIN_REPLY,
};

function reply(text: string): ThreadMessageLike[] {
  return [
    { role: "user", content: [{ type: "text", text: "show me" }] },
    { role: "assistant", content: [{ type: "text", text }] },
  ];
}

const adapter: ChatModelAdapter = {
  async *run() {
    yield { content: [{ type: "text", text: "" }] };
  },
};

function ProbeApi(): null {
  const aui = useAui();

  useEffect(() => {
    const which = new URLSearchParams(window.location.search).get("case") ?? "code";
    const text = CASES[which] ?? CODE_REPLY;
    void aui
      .thread()
      .import(ExportedMessageRepository.fromArray(reply(text)));

    const api = {
      ready(): boolean {
        return document.querySelectorAll("[data-role='assistant']").length > 0;
      },
      counts(): Record<string, number | string> {
        return {
          case: which,
          assistantMessages: document.querySelectorAll("[data-role='assistant']").length,
          codeBlocks: document.querySelectorAll('[data-streamdown="code-block"]').length,
          preElements: document.querySelectorAll("pre").length,
          copyButtons: document.querySelectorAll('button[title="Copy code"]').length,
          downloadButtons: document.querySelectorAll('button[title="Download file"]').length,
          renderedLinks: document.querySelectorAll("[data-role='assistant'] a[href]").length,
          text: (document.querySelector("[data-role='assistant']")?.textContent ?? "").slice(0, 120),
        };
      },
    };
    (window as unknown as { __probe: typeof api }).__probe = api;
  }, [aui]);

  return null;
}

function Harness(): ReactElement {
  const runtime = useLocalRuntime(adapter);
  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <ProbeApi />
        <div
          data-smoke="link-definition-probe"
          style={{ display: "flex", flexDirection: "column", height: "100vh" }}
        >
          <Thread hideWelcome={true} />
        </div>
      </AssistantRuntimeProvider>
    </TooltipProvider>
  );
}

const rootRoute = createRootRoute({ component: Harness });
const router = createRouter({
  routeTree: rootRoute,
  history: createMemoryHistory({ initialEntries: ["/"] }),
});

const root = document.getElementById("root");
if (!root) {
  throw new Error("missing #root");
}
createRoot(root).render(<RouterProvider router={router as unknown as never} />);
