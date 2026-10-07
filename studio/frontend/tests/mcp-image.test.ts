// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  imageFieldCandidates,
  mcpImageMappingsEnabled,
  modelVisibleMessage,
  toolOnlyImages,
} from "../src/features/chat/api/mcp-image.ts";

const PRIVATE = "data:image/png;base64,UFJJVkFURQ==";

const message = {
  role: "user",
  content: [{ type: "text", text: "which anime is this?" }],
  attachments: [
    {
      id: "a1",
      mcpToolOnly: true,
      content: [{ type: "image", image: PRIVATE, mcpToolOnly: true }],
    },
    {
      id: "a2",
      content: [{ type: "image", image: "data:image/png;base64,T0s=" }],
    },
  ],
};

test("tool-only images never reach the model but become mcp_image", () => {
  const visible = modelVisibleMessage(message);
  assert.deepEqual(
    visible.attachments?.map((a) => a.id),
    ["a2"],
  );
  assert.ok(!JSON.stringify(visible).includes(PRIVATE));
  assert.deepEqual(toolOnlyImages(message), [PRIVATE]);

  const reloaded = {
    content: [
      { type: "text", text: "hi" },
      { type: "image", image: PRIVATE, mcpToolOnly: true },
    ],
  };
  assert.deepEqual(modelVisibleMessage(reloaded).content, [
    { type: "text", text: "hi" },
  ]);
  assert.deepEqual(toolOnlyImages(reloaded), [PRIVATE]);

  const plain = { content: [{ type: "text", text: "hi" }] };
  assert.equal(modelVisibleMessage(plain), plain);
  assert.deepEqual(toolOnlyImages(plain), []);
});

test("only enabled servers with a mapping make images tool-only", () => {
  const server = { is_enabled: true, image_input_mappings: [] };
  const mapped = { tool: "lookup", field: "image", encoding: "base64" };
  type Servers = Parameters<typeof mcpImageMappingsEnabled>[0];
  assert.equal(mcpImageMappingsEnabled([server] as unknown as Servers), false);
  assert.equal(
    mcpImageMappingsEnabled([
      { ...server, image_input_mappings: [mapped] },
    ] as unknown as Servers),
    true,
  );
  assert.equal(
    mcpImageMappingsEnabled([
      { ...server, is_enabled: false, image_input_mappings: [mapped] },
    ] as unknown as Servers),
    false,
  );
  assert.equal(
    mcpImageMappingsEnabled([
      {
        ...server,
        image_input_mappings: [mapped],
        image_mappings_active: false,
      },
    ] as unknown as Servers),
    false,
  );
});

test("mapping candidates are the top-level string fields", () => {
  assert.deepEqual(
    imageFieldCandidates({
      type: "object",
      properties: {
        image: { type: "string" },
        url: { type: "string", format: "uri" },
        cut_borders: { type: "boolean" },
      },
    }),
    ["image", "url"],
  );
  assert.deepEqual(imageFieldCandidates(undefined), []);
});
