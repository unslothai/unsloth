import assert from "node:assert/strict";
import { test } from "node:test";
import {
  mcpImageAttachmentForTokenCount,
  modelVisibleMessage,
} from "../src/features/chat/api/mcp-image-privacy.ts";

test("tool-only bytes and extracted aliases stay private through serialized reload and replay", () => {
  const secret = "data:image/png;base64,PRIVATE_CANARY";
  const image = { type: "image", image: secret, mcpToolOnly: true };
  const extracted = { type: "text", text: "PRIVATE_OCR", mcpToolOnly: true };
  const message = {
    content: [image, extracted, { type: "text", text: "Inspect it" }],
    attachments: [
      { mcpToolOnly: true, content: [image, extracted] },
      { content: [{ type: "image", image: "data:image/png;base64,VISION" }] },
    ],
  };
  for (const input of [message, JSON.parse(JSON.stringify(message))]) {
    const visible = modelVisibleMessage(input);
    const wire = JSON.stringify(visible);
    assert.ok(!wire.includes("PRIVATE"));
    assert.ok(wire.includes("VISION"));
    assert.ok(wire.includes("Inspect it"));
    assert.deepEqual(modelVisibleMessage(visible), visible);
  }
  assert.equal(message.attachments[0]?.mcpToolOnly, true);
  assert.ok(JSON.stringify(message).includes(secret));
});

test("ordinary parts survive when their values equal private attachment parts", () => {
  const secret = "data:image/png;base64,SHARED_VALUE";
  const text = "shared extracted text";
  const message = {
    content: [
      { type: "image", image: secret },
      { type: "text", text },
    ],
    attachments: [
      {
        mcpToolOnly: true,
        content: [
          { type: "image", image: secret },
          { type: "text", text },
        ],
      },
    ],
  };

  const visible = modelVisibleMessage(message);
  assert.deepEqual(visible.content, message.content);
  assert.deepEqual(visible.attachments, []);
});

test("token counts reference one persisted tool-only image without serializing its bytes", () => {
  const secret = "data:image/png;base64,PRIVATE_COUNT_CANARY";
  const messages = [
    {
      id: "message-1",
      role: "user",
      content: [{ type: "text", text: "Inspect it" }],
      attachments: [
        {
          id: "image-1",
          mcpToolOnly: true,
          content: [{ type: "image", image: secret }],
        },
      ],
    },
  ];
  const persisted = [
    {
      id: "message-1",
      threadId: "thread-1",
      attachments: [{ id: "image-1", mcpToolOnly: true }],
    },
  ];

  const selection = mcpImageAttachmentForTokenCount(
    messages,
    "thread-1",
    persisted,
  );
  assert.deepEqual(selection, {
    message_id: "message-1",
    attachment_id: "image-1",
  });
  assert.ok(!JSON.stringify(selection).includes(secret));

  assert.equal(
    mcpImageAttachmentForTokenCount(messages, "other-thread", persisted),
    undefined,
  );
  assert.equal(
    mcpImageAttachmentForTokenCount(
      [
        {
          ...messages[0],
          attachments: [
            ...messages[0].attachments,
            { id: "image-2", mcpToolOnly: true, content: [] },
          ],
        },
      ],
      "thread-1",
      persisted,
    ),
    undefined,
  );
});
