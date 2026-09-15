import assert from "node:assert/strict";
import { test } from "node:test";
import {
  disclosureExpired,
  markImageDisclosureReceived,
  mayAutoApproveTool,
  mcpImageAttachmentForTokenCount,
  modelVisibleMessage,
} from "../src/features/chat/api/mcp-image-privacy.ts";

const privateImage = (id: string, image?: string) => ({
  id,
  mcpToolOnly: true,
  content: image ? [{ type: "image", image }] : [],
});
const userMessage = (attachments: ReturnType<typeof privateImage>[]) => ({
  id: "message-1",
  role: "user",
  content: [],
  attachments,
});

test("tool-only bytes and extracted aliases stay private through serialized reload and replay", () => {
  const secret = "data:image/png;base64,PRIVATE_CANARY";
  const privateParts = [
    { type: "image", image: secret, mcpToolOnly: true },
    { type: "text", text: "PRIVATE_OCR", mcpToolOnly: true },
  ];
  const message = {
    content: [...privateParts, { type: "text", text: "Inspect it" }],
    attachments: [
      { mcpToolOnly: true, content: privateParts },
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
  const content = [
    { type: "image", image: "data:image/png;base64,SHARED_VALUE" },
    { type: "text", text: "shared extracted text" },
  ];
  const visible = modelVisibleMessage({
    content,
    attachments: [
      { mcpToolOnly: true, content: content.map((part) => ({ ...part })) },
    ],
  });
  assert.deepEqual(visible.content, content);
  assert.deepEqual(visible.attachments, []);
});

test("token counts reference one persisted tool-only image without serializing its bytes", () => {
  const secret = "data:image/png;base64,PRIVATE_COUNT_CANARY";
  const messages = [userMessage([privateImage("image-1", secret)])];
  const persisted = [
    {
      id: "message-1",
      threadId: "thread-1",
      attachments: [{ id: "image-1", mcpToolOnly: true }],
    },
  ];
  const select = (candidate: typeof messages = messages, thread = "thread-1") =>
    mcpImageAttachmentForTokenCount(candidate, thread, persisted);

  const selection = select();
  assert.deepEqual(selection, {
    message_id: "message-1",
    attachment_id: "image-1",
  });
  assert.equal(select(messages, "other-thread"), undefined);

  const ambiguous = [
    userMessage([...messages[0].attachments, privateImage("image-2")]),
  ];
  assert.equal(select(ambiguous), undefined);
});

const card = {
  purpose: "mcp_image_disclosure",
  sizeBytes: 1,
  serverName: "server",
  toolName: "inspect",
  destination: "stdio",
  field: "blob",
  encoding: "base64",
  expiresAt: 1,
  expiresInMs: 100,
} as const;

test("expiry is relative to receipt instead of the server or client wall clock", () => {
  const received = markImageDisclosureReceived(card, 1_000);
  if (!received) throw new Error("disclosure was not marked received");
  assert.equal(disclosureExpired(received, 1_099), false);
  assert.equal(disclosureExpired(received, 1_100), true);
  assert.equal(disclosureExpired({ ...card, status: "cancelled" }, 0), true);
  assert.equal(disclosureExpired({ ...card, status: "expired" }, 0), true);
});

test("Always allow never auto-resolves image disclosure, retaining ordinary approvals", () => {
  assert.equal(mayAutoApproveTool(card, true), false);
  assert.equal(mayAutoApproveTool(card, false), false);
  assert.equal(mayAutoApproveTool(undefined, true), true);
  assert.equal(mayAutoApproveTool(undefined, false), false);
});
