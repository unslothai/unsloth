// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  changedImageMappings,
  imageFieldCandidates,
  mcpImageMappingsEnabled,
  modelVisibleMessage,
  toolOnlyImages,
  unmappedImageFields,
  withImageField,
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

  // A reloaded thread can carry the flagged image as message content.
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

test("the add list leaves out pairs that are already mapped", () => {
  const options = [
    { tool: "search_by_file", field: "filePath" },
    { tool: "search_by_file", field: "imageBase64" },
    { tool: "search_by_url", field: "url" },
  ];
  assert.deepEqual(
    unmappedImageFields(options, [
      { tool: "search_by_file", field: "imageBase64", encoding: "data_url" },
    ]),
    [options[0], options[2]],
  );
  assert.deepEqual(unmappedImageFields(options, []), options);
});

test("adding a field keeps one row per tool, in place", () => {
  const mapped = [
    { tool: "search_by_file", field: "imageBase64", encoding: "data_url" },
    { tool: "search_by_url", field: "url", encoding: "base64" },
  ] as const;
  assert.deepEqual(
    withImageField(mapped, { tool: "search_by_file", field: "filePath" }),
    [
      { tool: "search_by_file", field: "filePath", encoding: "base64" },
      mapped[1],
    ],
  );
  assert.deepEqual(withImageField(mapped, { tool: "lookup", field: "image" }), [
    ...mapped,
    { tool: "lookup", field: "image", encoding: "base64" },
  ]);
});

test("a tool saved with two rows keeps one after picking a field", () => {
  const saved = [
    { tool: "search_by_file", field: "imageBase64", encoding: "data_url" },
    { tool: "search_by_url", field: "url", encoding: "base64" },
    { tool: "search_by_file", field: "filePath", encoding: "base64" },
  ] as const;
  assert.deepEqual(
    withImageField(saved, { tool: "search_by_file", field: "file" }),
    [{ tool: "search_by_file", field: "file", encoding: "base64" }, saved[1]],
  );
});

test("changing a row's encoding keeps that field and its tool's other rows go", () => {
  const saved = [
    { tool: "search_by_file", field: "imageBase64", encoding: "base64" },
    { tool: "search_by_file", field: "filePath", encoding: "base64" },
  ] as const;
  assert.deepEqual(withImageField(saved, saved[1], "data_url"), [
    { tool: "search_by_file", field: "filePath", encoding: "data_url" },
  ]);
});

test("an edit leaves unchanged mappings out of the update", () => {
  const saved = [
    { tool: "search_by_file", field: "imageBase64", encoding: "base64" },
    { tool: "search_by_file", field: "filePath", encoding: "base64" },
  ] as const;
  assert.equal(changedImageMappings(saved, [...saved]), undefined);
  const edited = [{ ...saved[0], encoding: "data_url" as const }];
  assert.deepEqual(changedImageMappings(saved, edited), edited);
  assert.deepEqual(changedImageMappings([], []), undefined);
});
