import assert from "node:assert/strict";
import test from "node:test";

import {
  deriveDefaultMapping,
  getUnselectedInstructionColumns,
  getViewerColumnSelections,
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

test("viewer summary exposes every auto-selected VLM column", () => {
  const selections = getViewerColumnSelections(
    {
      ...base,
      columns: [...base.columns, "instruction"],
      detected_instruction_column: "instruction",
    },
    true,
    {},
  );
  assert.deepEqual(selections, [
    { column: "image", label: "Image", source: "auto" },
    { column: "text", label: "Assistant response", source: "auto" },
    { column: "instruction", label: "User instruction", source: "auto" },
  ]);
});

test("viewer summary updates and identifies a manual mapping", () => {
  const selections = getViewerColumnSelections(base, true, {
    caption: "text",
  });
  assert.deepEqual(selections, [
    { column: "image", label: "Image", source: "auto" },
    { column: "caption", label: "Assistant response", source: "manual" },
  ]);
});

test("viewer summary warns only for a present unselected instruction column", () => {
  const data = { ...base, columns: [...base.columns, "instruction"] };
  const selections = getViewerColumnSelections(data, true, {});
  assert.deepEqual(getUnselectedInstructionColumns(data, selections), [
    "instruction",
  ]);
  assert.deepEqual(getUnselectedInstructionColumns(base, selections), []);
});
