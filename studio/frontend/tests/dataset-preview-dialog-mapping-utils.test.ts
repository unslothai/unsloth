import assert from "node:assert/strict";
import test from "node:test";

import {
  deriveDefaultMapping,
  isMappingComplete,
} from "../src/features/studio/sections/dataset-preview-dialog-mapping-utils.ts";

const base = {
  requires_manual_mapping: true,
  detected_format: "simple_image_text",
  columns: ["image", "text"],
  detected_image_column: "image",
  detected_text_column: "text",
};

test("VLM instruction is mapped dynamically without replacing image and text", () => {
  const mapping = deriveDefaultMapping(
    {
      ...base,
      columns: [...base.columns, "instruction"],
      detected_instruction_column: "instruction",
    },
    true,
  );
  assert.deepEqual(mapping, {
    image: "image",
    text: "text",
    instruction: "user",
  });
  assert.equal(isMappingComplete(mapping, true), true);
});

test("VLM with an empty first-row instruction detection keeps required mappings", () => {
  const mapping = deriveDefaultMapping(
    {
      ...base,
      columns: [...base.columns, "instruction"],
      detected_instruction_column: null,
    },
    true,
  );
  assert.deepEqual(mapping, { image: "image", text: "text" });
});

test("VLM without an instruction column keeps required mappings", () => {
  assert.deepEqual(deriveDefaultMapping(base, true), {
    image: "image",
    text: "text",
  });
});
