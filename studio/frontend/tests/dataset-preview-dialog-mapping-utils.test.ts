import assert from "node:assert/strict";
import test from "node:test";

import {
  deriveDefaultMapping,
  getAvailableRoles,
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
    { column: "instruction", reason: "empty_first_value" },
  ]);
  assert.deepEqual(getUnselectedInstructionColumns(base, selections), []);
});

test("audio VLM mapping selects audio, instruction, and text together", () => {
  const data = {
    ...base,
    is_audio: true,
    columns: ["audio", "instruction", "text"],
    detected_audio_column: "audio",
    detected_instruction_column: "instruction",
    detected_text_column: "text",
    preview_samples: [
      {
        audio: "sample.wav",
        instruction: "Transcribe this audio.",
        text: "Hello",
      },
    ],
  };
  const mapping = deriveDefaultMapping(data, false, undefined, true, true);

  assert.deepEqual(mapping, {
    audio: "audio",
    instruction: "user",
    text: "text",
  });
  assert.deepEqual(getAvailableRoles(false, undefined, true, true), [
    "audio",
    "user",
    "text",
    "speaker_id",
  ]);
  assert.equal(isMappingComplete(mapping, false, undefined, true), true);
});

test("Whisper ASR maps only audio and text", () => {
  const mapping = deriveDefaultMapping(
    {
      ...base,
      is_audio: true,
      columns: ["audio", "instruction", "text"],
      detected_audio_column: "audio",
      detected_instruction_column: "instruction",
      detected_text_column: "text",
    },
    false,
    undefined,
    true,
    false,
  );

  assert.deepEqual(mapping, { audio: "audio", text: "text" });
  assert.deepEqual(getAvailableRoles(false, undefined, true, false), [
    "audio",
    "text",
    "speaker_id",
  ]);
});

test("Gemma audio VLM maps the instruction column to user", () => {
  const data = {
    ...base,
    is_audio: true,
    columns: ["audio", "instruction", "text"],
    detected_audio_column: "audio",
    detected_instruction_column: "instruction",
    detected_text_column: "text",
  };

  assert.deepEqual(deriveDefaultMapping(data, false, undefined, true, true), {
    audio: "audio",
    instruction: "user",
    text: "text",
  });
  assert.deepEqual(getViewerColumnSelections(data, false, {}, true, true), [
    { column: "audio", label: "Audio", source: "auto" },
    { column: "instruction", label: "User instruction", source: "auto" },
    { column: "text", label: "Assistant response", source: "auto" },
  ]);
});

test("blank audio VLM instructions retain the mapping for transcription fallback", () => {
  const data = {
    ...base,
    is_audio: true,
    columns: ["audio", "instruction", "text"],
    detected_audio_column: "audio",
    detected_instruction_column: "instruction",
    detected_text_column: "text",
    preview_samples: [
      { audio: "sample.wav", instruction: "   ", text: "Hello" },
    ],
  };

  const mapping = deriveDefaultMapping(data, false, undefined, true, true);
  assert.deepEqual(mapping, {
    audio: "audio",
    instruction: "user",
    text: "text",
  });
  assert.equal(isMappingComplete(mapping, false, undefined, true), true);
});

test("audio VLM preview does not warn about an auto-selected instruction", () => {
  const data = {
    ...base,
    is_audio: true,
    columns: ["audio", "instruction", "text"],
    detected_audio_column: "audio",
    detected_instruction_column: "instruction",
    detected_text_column: "text",
    preview_samples: [
      { audio: "sample.wav", instruction: "", text: "Hello" },
      { audio: "other.wav", instruction: "Translate this audio.", text: "Hi" },
    ],
  };
  const selections = getViewerColumnSelections(data, false, {}, true, true);

  assert.deepEqual(getUnselectedInstructionColumns(data, selections, true), []);
});
