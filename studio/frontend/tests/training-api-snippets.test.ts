// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Training API card's examples carry the Train page's settings, never its secrets or host paths.

import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import test from "node:test";
import {
  EXAMPLE_TRAINING_API_REQUEST,
  buildTrainingApiSnippet,
  buildTrainingCancelSnippet,
  trainingApiJobBody,
  trainingApiRequestFromPayload,
} from "../src/features/settings/components/training-api-snippets.ts";
import type { TrainingStartRequest } from "../src/features/training/types/api.ts";
import { readSrc } from "./helpers/kit.ts";

const BASE = "http://127.0.0.1:8888";
const KEY = "sk-unsloth-YOUR_KEY";
const HOSTILE = "$(calc) it's";

function payload(overrides: Partial<TrainingStartRequest> = {}): TrainingStartRequest {
  return {
    model_name: "unsloth/Qwen3-0.6B",
    project_name: "my run",
    training_type: "LoRA/QLoRA",
    hf_token: "hf_SECRET",
    model_known_cached: true,
    model_local_path: "/home/me/.cache/models/qwen",
    model_format: "safetensors",
    load_in_4bit: false,
    max_seq_length: 4096,
    vision_image_size: null,
    trust_remote_code: false,
    approved_remote_code_fingerprint: "abc",
    hf_dataset: null,
    dataset_known_cached: false,
    dataset_local_path: null,
    subset: null,
    train_split: null,
    eval_split: null,
    dataset_streaming: false,
    dataset_slice_start: null,
    dataset_slice_end: null,
    local_datasets: ["/home/me/.unsloth/uploads/0123456789abcdef0123456789abcdef_chats.jsonl"],
    local_eval_datasets: [],
    s3_config: null,
    format_type: "auto",
    custom_format_mapping: undefined,
    num_epochs: 3,
    learning_rate: "0.0002",
    batch_size: 4,
    gradient_accumulation_steps: 2,
    warmup_steps: 5,
    warmup_ratio: null,
    max_steps: 0,
    save_steps: 100,
    eval_steps: 0,
    weight_decay: 0.001,
    random_seed: 3407,
    packing: false,
    optim: "adamw_8bit",
    lr_scheduler_type: "linear",
    use_lora: true,
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0,
    target_modules: ["q_proj", "v_proj"],
    gradient_checkpointing: "unsloth",
    use_rslora: false,
    use_loftq: false,
    use_dora: true,
    train_on_completions: true,
    finetune_vision_layers: false,
    finetune_language_layers: true,
    finetune_attention_modules: true,
    finetune_mlp_modules: true,
    is_dataset_image: false,
    is_dataset_audio: false,
    is_embedding: false,
    is_decision: false,
    enable_wandb: true,
    wandb_token: "wandb_SECRET",
    enable_tensorboard: true,
    tensorboard_dir: "/home/me/tb",
    ...overrides,
  } as TrainingStartRequest;
}

const ALLOWED = new Set([
  "learning_rate",
  "max_steps",
  "training_type",
  "load_in_4bit",
  "max_seq_length",
  "gradient_accumulation_steps",
  "warmup_steps",
  "warmup_ratio",
  "weight_decay",
  "optim",
  "lr_scheduler_type",
  "packing",
  "lora_r",
  "lora_alpha",
  "lora_dropout",
  "target_modules",
  "use_rslora",
  "use_dora",
  "use_loftq",
  "train_on_completions",
  "is_dataset_image",
  "is_dataset_audio",
  "is_embedding",
  "is_decision",
  "finetune_vision_layers",
  "finetune_language_layers",
  "finetune_attention_modules",
  "finetune_mlp_modules",
  "format_type",
  "custom_format_mapping",
]);

const SECRETS = ["hf_SECRET", "wandb_SECRET", "/home/me", "abc\"", "tensorboard", "offload", "drafter"];

const LANGS = [
  ["python", "unix"],
  ["javascript", "unix"],
  ["curl", "unix"],
  ["curl", "windows"],
] as const;

test("the snapshot keeps only the allowlist and maps the OpenAI fields", () => {
  const request = trainingApiRequestFromPayload(payload(), "chats.jsonl");
  assert.equal(request.model, "unsloth/Qwen3-0.6B");
  assert.equal(request.datasetName, "chats.jsonl");
  assert.equal(request.datasetFromHub, false);
  assert.equal(request.nEpochs, 3);
  assert.equal(request.batchSize, 4);
  assert.equal(request.seed, 3407);
  assert.equal(request.suffix, "my run");
  for (const key of Object.keys(request.extras)) assert.ok(ALLOWED.has(key), key);
  assert.equal(request.extras.learning_rate, "0.0002");
  assert.equal(request.extras.use_dora, true);
  assert.ok(!("use_rslora" in request.extras));
  assert.ok(!("max_steps" in request.extras));
  assert.ok(!("training_type" in request.extras));
  assert.ok(!("format_type" in request.extras));
  assert.deepEqual(request.extras.target_modules, ["q_proj", "v_proj"]);
});

test("full finetuning names its training type and drops the LoRA settings", () => {
  const request = trainingApiRequestFromPayload(
    payload({ training_type: "Full Finetuning", use_lora: false, target_modules: [], max_steps: 60 }),
    null,
  );
  assert.equal(request.extras.training_type, "Full Finetuning");
  assert.equal(request.extras.max_steps, 60);
  for (const key of ["lora_r", "lora_alpha", "lora_dropout", "target_modules", "use_dora"]) {
    assert.ok(!(key in request.extras), key);
  }
});

test("a vision, audio, embedding or decision run keeps its run type and layer choices", () => {
  const plain = trainingApiRequestFromPayload(payload(), "chats.jsonl");
  for (const key of ["is_dataset_image", "finetune_vision_layers", "finetune_mlp_modules"]) {
    assert.ok(!(key in plain.extras), key);
  }
  const vision = trainingApiRequestFromPayload(
    payload({ is_dataset_image: true, finetune_vision_layers: true, finetune_mlp_modules: false }),
    "images.zip",
  );
  assert.equal(vision.extras.is_dataset_image, true);
  assert.equal(vision.extras.finetune_vision_layers, true);
  assert.equal(vision.extras.finetune_mlp_modules, false);
  assert.ok(!("finetune_language_layers" in vision.extras));
  for (const flag of ["is_dataset_audio", "is_embedding", "is_decision"] as const) {
    const request = trainingApiRequestFromPayload(payload({ [flag]: true }), null);
    assert.equal(request.extras[flag], true, flag);
  }
});

test("no dataset picked yet is not mistaken for a Hub dataset", () => {
  const request = trainingApiRequestFromPayload(payload({ local_datasets: [] }), null);
  assert.equal(request.datasetFromHub, false);
  assert.equal(request.datasetName, null);
  const s3 = trainingApiRequestFromPayload(
    payload({ local_datasets: [], s3_config: { bucket: "b" } as TrainingStartRequest["s3_config"] }),
    null,
  );
  assert.equal(s3.datasetFromHub, true);
});

test("a Hub or S3 dataset is flagged, and its fields never reach the example", () => {
  const request = trainingApiRequestFromPayload(
    payload({ hf_dataset: "org/data", subset: "x", local_datasets: [], format_type: "sharegpt" }),
    null,
  );
  assert.equal(request.datasetFromHub, true);
  assert.equal(request.datasetName, null);
  assert.equal(request.extras.format_type, "sharegpt");
  for (const [lang, os] of LANGS) {
    const snippet = buildTrainingApiSnippet(request, { base: BASE, apiKey: KEY, lang, os });
    assert.ok(snippet.includes("train.jsonl"), `${lang} ${os}`);
    assert.ok(!snippet.includes("org/data"), `${lang} ${os}`);
  }
});

for (const [lang, os] of LANGS) {
  const label = `${lang} ${os}`;
  const request = trainingApiRequestFromPayload(
    payload({ custom_format_mapping: { instruction: "user", __system_prompt: "Be brief" } }),
    "chats.jsonl",
  );
  const snippet = buildTrainingApiSnippet(request, { base: BASE, apiKey: KEY, lang, os });

  test(`${label}: uploads, creates, follows events, then chats with the trained model`, () => {
    assert.ok(snippet.includes("chats.jsonl"));
    assert.ok(snippet.includes("fine-tune"));
    assert.ok(snippet.includes("unsloth/Qwen3-0.6B"));
    assert.ok(snippet.includes("my run"));
    assert.ok(snippet.includes("0.0002"));
    assert.match(snippet, /fine_tuned_model/);
    assert.match(snippet, /succeeded/);
    assert.match(snippet, /cancelled/);
    assert.ok(!snippet.includes("cancel("), "cancel is a separate example");
    assert.ok(!snippet.includes("/cancel"), "cancel is a separate example");
    for (const secret of SECRETS) assert.ok(!snippet.includes(secret), secret);
  });

  test(`${label}: the extras go on the wire flat, never inside an extra_body key`, () => {
    if (lang === "python") {
      assert.match(snippet, /extra_body=\{\n {8}"learning_rate": "0\.0002",/);
      assert.match(snippet, /"use_dora": True,/);
      assert.match(snippet, /"train_on_completions": True,/);
      assert.match(snippet, /"packing": False,/);
      assert.match(snippet, /hyperparameters=\{"n_epochs": 3, "batch_size": 4\}/);
      assert.match(snippet, /"__system_prompt": "Be brief"/);
      return;
    }
    assert.ok(!snippet.includes("extra_body"));
    if (lang === "javascript") {
      assert.match(snippet, /\n {2}learning_rate: "0\.0002",\n/);
      assert.match(snippet, /\n {2}use_dora: true,\n/);
      assert.match(snippet, /hyperparameters: \{ n_epochs: 3, batch_size: 4 \}/);
      return;
    }
    assert.match(snippet, /\n {2}"learning_rate": "0\.0002"/);
    assert.match(snippet, /\n {2}"hyperparameters": \{"n_epochs":3,"batch_size":4\}/);
    assert.match(snippet, /\n {2}"use_dora": true/);
  });
}

test("events are read newest first, de-duplicated and printed oldest first", () => {
  const request = EXAMPLE_TRAINING_API_REQUEST;
  const python = buildTrainingApiSnippet(request, { base: BASE, apiKey: KEY, lang: "python", os: "unix" });
  assert.match(python, /for event in reversed\(events\):\n\s+if event\.id not in seen:/);
  const js = buildTrainingApiSnippet(request, { base: BASE, apiKey: KEY, lang: "javascript", os: "unix" });
  assert.match(js, /\[\.\.\.events\.data\]\.reverse\(\)/);
  assert.match(js, /if \(seen\.has\(event\.id\)\) continue;/);
  const bash = buildTrainingApiSnippet(request, { base: BASE, apiKey: KEY, lang: "curl", os: "unix" });
  assert.match(bash, /\.data \| reverse\[\]/);
  assert.match(bash, /\*" \$EVENT_ID "\*\) ;;/);
  const ps = buildTrainingApiSnippet(request, { base: BASE, apiKey: KEY, lang: "curl", os: "windows" });
  assert.match(ps, /\[array\]::Reverse\(\$events\)/);
  assert.match(ps, /if \(-not \$seen\.ContainsKey\(\$item\.id\)\)/);
});

test("the PowerShell body is a literal here-string, so $(...) in a setting stays text", () => {
  const request = trainingApiRequestFromPayload(
    payload({ model_name: HOSTILE, project_name: HOSTILE }),
    HOSTILE,
  );
  const snippet = buildTrainingApiSnippet(request, { base: BASE, apiKey: KEY, lang: "curl", os: "windows" });
  assert.match(snippet, /\$settings = @'\n {2}"model": "\$\(calc\) it's",\n/);
  assert.match(snippet, /\n'@\n/);
  assert.ok(snippet.includes(`-F 'file=@$(calc) it''s'`));
  assert.ok(snippet.includes('"{`n  `"training_file`": `"$($file.id)`",`n$settings`n}"'));
});

test("the bash body stays inside single quotes and splices only the file id", () => {
  const request = trainingApiRequestFromPayload(
    payload({ model_name: HOSTILE, project_name: HOSTILE }),
    HOSTILE,
  );
  const snippet = buildTrainingApiSnippet(request, { base: BASE, apiKey: KEY, lang: "curl", os: "unix" });
  assert.ok(snippet.includes(`  "model": "$(calc) it'\\''s",`));
  assert.ok(snippet.includes(`  "suffix": "$(calc) it'\\''s",`));
  assert.ok(snippet.includes(`  "training_file": "'"$FILE_ID"'",`));
  assert.ok(snippet.includes(`-F 'file=@$(calc) it'\\''s'`));
});

test("python and javascript strings are escaped literals", () => {
  const request = trainingApiRequestFromPayload(
    payload({ model_name: 'a"b\\c', project_name: null }),
    null,
  );
  for (const lang of ["python", "javascript"] as const) {
    const snippet = buildTrainingApiSnippet(request, { base: BASE, apiKey: KEY, lang, os: "unix" });
    assert.ok(snippet.includes('"a\\"b\\\\c"'), lang);
    assert.ok(!snippet.includes("suffix"), lang);
  }
});

test("the placeholder example and the cancel example render for every language", () => {
  for (const [lang, os] of LANGS) {
    const snippet = buildTrainingApiSnippet(EXAMPLE_TRAINING_API_REQUEST, { base: BASE, apiKey: KEY, lang, os });
    assert.ok(snippet.includes("unsloth/Qwen3-0.6B"));
    assert.ok(snippet.includes("train.jsonl"));
    const cancel = buildTrainingCancelSnippet({ base: BASE, apiKey: KEY, lang, os });
    assert.match(cancel, /cancel/);
    assert.ok(cancel.includes(KEY));
  }
});

test("the Train header leads its page links with the API button, and the API tab has a Training kind", () => {
  const page = readSrc("features/studio/studio-page.tsx");
  assert.match(page, /<div className="ml-auto flex items-center gap-2">\s*<TrainingApiButton \/>\s*<MediaPageLink\b[^]*?to="\/library"/);
  const button = readSrc("features/studio/training-api-button.tsx");
  assert.match(button, /className=\{HEADER_ICON_BUTTON\}/);
  assert.match(button, /buildTrainingStartPayload\(config, null\)/);
  assert.match(button, /\.openTrainingApi\(/);
  const tabs = readSrc("features/settings/components/api-example-tabs.tsx");
  assert.match(tabs, /"api-keys-training-api": "training"/);
  assert.match(tabs, /<TrainingApiExamples[^>]*showHeading=\{false\}/);
  const monitor = readSrc("features/api-monitor/api-monitor-page.tsx");
  assert.match(monitor, /kind === "training" \? \(\s*<TrainingApiExamples/);
});

// studio/backend/tests/test_training_api_snippet_contract.py sends these bodies through the fine-tuning
// route's validation; UPDATE_TRAINING_API_BODIES=1 rewrites them after an intended change.
const BODIES = new URL("../../backend/tests/fixtures/training_api_job_bodies.json", import.meta.url);

test("the backend contract fixture is what the examples send", () => {
  const bodies = {
    lora: trainingApiJobBody(trainingApiRequestFromPayload(payload(), "chats.jsonl")),
    fullFinetuning: trainingApiJobBody(
      trainingApiRequestFromPayload(
        payload({
          training_type: "Full Finetuning",
          use_lora: false,
          target_modules: [],
          use_dora: false,
          max_steps: 60,
          warmup_steps: null,
          warmup_ratio: 0.03,
          format_type: "alpaca",
          project_name: null,
        }),
        null,
      ),
    ),
    qloraMapping: trainingApiJobBody(
      trainingApiRequestFromPayload(
        payload({
          load_in_4bit: true,
          use_dora: false,
          use_rslora: true,
          learning_rate: "1e-5",
          train_on_completions: false,
          custom_format_mapping: {
            question: "user",
            answer: "assistant",
            __system_prompt: "Answer in one word.",
            __label_mapping: { answer: { "0": "no", "1": "yes" } },
          },
        }),
        null,
      ),
    ),
    continuedPretraining: trainingApiJobBody(
      trainingApiRequestFromPayload(
        payload({ training_type: "Continued Pretraining", use_dora: false, use_loftq: true }),
        null,
      ),
    ),
    vision: trainingApiJobBody(
      trainingApiRequestFromPayload(
        payload({ is_dataset_image: true, finetune_vision_layers: true, use_dora: false }),
        null,
      ),
    ),
    example: trainingApiJobBody(EXAMPLE_TRAINING_API_REQUEST),
  };
  if (process.env.UPDATE_TRAINING_API_BODIES === "1") {
    writeFileSync(BODIES, `${JSON.stringify(bodies, null, 2)}\n`);
  }
  assert.deepEqual(bodies, JSON.parse(readFileSync(BODIES, "utf8")));
});
