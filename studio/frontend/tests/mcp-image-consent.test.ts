import assert from "node:assert/strict";
import { test } from "node:test";
import {
  type ImageDisclosure,
  disclosureExpired,
  markImageDisclosureReceived,
  mayAutoApproveTool,
} from "../src/features/chat/api/mcp-image-privacy.ts";

const card: ImageDisclosure = {
  purpose: "mcp_image_disclosure",
  sizeBytes: 1,
  serverName: "server",
  toolName: "inspect",
  destination: "stdio",
  field: "blob",
  encoding: "base64",
  expiresAt: 1,
  expiresInMs: 100,
};
test("expiry is relative to receipt instead of the server or client wall clock", () => {
  const received = markImageDisclosureReceived(card, 1_000);
  assert.ok(received);
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
