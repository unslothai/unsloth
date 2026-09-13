import assert from "node:assert/strict";
import { test } from "node:test";
import { eligibleImageFields } from "../src/features/chat/api/mcp-image-mapping-options.ts";
import { readSrc } from "./helpers/kit.ts";

const settingsSource = readSrc(
  "features/chat/api/mcp-image-settings-controls.tsx",
);

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

test("focus refreshes cannot overwrite a pending image-sharing save", () => {
  assert.match(settingsSource, /const operationId = useRef\(0\)/);
  assert.match(settingsSource, /const savePending = useRef\(false\)/);
  assert.match(settingsSource, /if \(savePending\.current\) return;/);
  assert.match(
    settingsSource,
    /!savePending\.current[\s\S]*requestId === operationId\.current/,
  );
  assert.match(settingsSource, /savePending\.current = true/);
});
