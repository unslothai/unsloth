import assert from "node:assert/strict";
import { test } from "node:test";
import { eligibleImageFields } from "../src/features/chat/api/mcp-image-mapping-options.ts";
import {
  beginMcpImageSettingRefresh,
  beginMcpImageSettingSave,
  canApplyMcpImageSettingRefresh,
  canApplyMcpImageSettingSave,
  finishMcpImageSettingSave,
} from "../src/features/chat/api/mcp-image-setting-order.ts";

test("mapping options come from exact top-level schema strings without name inference", () => {
  assert.deepEqual(
    eligibleImageFields({
      type: "object",
      properties: {
        // biome-ignore lint/style/useNamingConvention: MCP schemas can use arbitrary field names.
        unrelated_name: { type: "string" },
        image: { type: "number" },
        nested: { type: "object" },
      },
    }),
    ["unrelated_name"],
  );
});
test("ambiguous schemas and restricted payload values are unavailable", () => {
  assert.deepEqual(
    eligibleImageFields({
      type: "object",
      oneOf: [],
      properties: { x: { type: "string" } },
    }),
    [],
  );
  assert.deepEqual(
    eligibleImageFields({
      type: "object",
      properties: {
        x: { type: ["string", "null"] },
        y: { type: "string", enum: ["private"] },
        z: { $ref: "#/$defs/image" },
      },
    }),
    [],
  );
});

test("refresh ordering survives a settings control remount", () => {
  const save = beginMcpImageSettingSave();
  assert.equal(beginMcpImageSettingRefresh(), null);
  assert.equal(canApplyMcpImageSettingSave(save), true);
  assert.equal(finishMcpImageSettingSave(save), true);

  const staleRefresh = beginMcpImageSettingRefresh();
  assert.notEqual(staleRefresh, null);
  const nextSave = beginMcpImageSettingSave();
  assert.equal(canApplyMcpImageSettingRefresh(staleRefresh!), false);
  assert.equal(canApplyMcpImageSettingSave(nextSave), true);
  assert.equal(finishMcpImageSettingSave(nextSave), true);
});
