// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingStartRequest } from "@/features/training/types/api";
import type { TrainingApiRequest } from "../stores/settings-dialog-store";
import { psSingle, shSingle } from "./agent-command.ts";
import type { AudioApiLang, AudioApiOs } from "./audio-api-snippets";

export const TRAINING_API_PLACEHOLDER_MODEL = "unsloth/Qwen3-0.6B";
export const TRAINING_API_DEFAULT_DATASET = "train.jsonl";

export interface TrainingApiSnippetInput {
  base: string;
  apiKey: string;
  lang: AudioApiLang;
  os: AudioApiOs;
}

const LORA = "LoRA/QLoRA";

/** Run type and layer choices, sent only where they differ from the route's defaults. */
const FLAG_DEFAULTS = {
  "is_dataset_image": false,
  "is_dataset_audio": false,
  "is_embedding": false,
  "is_decision": false,
  "finetune_vision_layers": false,
  "finetune_language_layers": true,
  "finetune_attention_modules": true,
  "finetune_mlp_modules": true,
} as const;

/**
 * The Studio settings a fine-tuning job takes beyond the OpenAI fields. Everything else Start sends
 * (tokens, cache and host paths, offload, drafter, the Hub / S3 dataset) stays out of the example.
 */
function trainingExtras(payload: TrainingStartRequest): Record<string, unknown> {
  const extras: Record<string, unknown> = {
    "learning_rate": payload.learning_rate,
  };
  if (payload.max_steps != null && payload.max_steps > 0) {
    extras.max_steps = payload.max_steps;
  }
  const lora = payload.training_type === LORA;
  if (!lora) extras.training_type = payload.training_type;
  extras.load_in_4bit = payload.load_in_4bit;
  extras.max_seq_length = payload.max_seq_length;
  extras.gradient_accumulation_steps = payload.gradient_accumulation_steps;
  if (payload.warmup_steps != null) {
    extras.warmup_steps = payload.warmup_steps;
  } else if (payload.warmup_ratio != null) {
    extras.warmup_ratio = payload.warmup_ratio;
  }
  extras.weight_decay = payload.weight_decay;
  extras.optim = payload.optim;
  extras.lr_scheduler_type = payload.lr_scheduler_type;
  extras.packing = payload.packing;
  if (payload.use_lora) {
    extras.lora_r = payload.lora_r;
    extras.lora_alpha = payload.lora_alpha;
    extras.lora_dropout = payload.lora_dropout;
    if (payload.target_modules.length > 0) {
      extras.target_modules = [...payload.target_modules];
    }
    for (const flag of ["use_rslora", "use_dora", "use_loftq"] as const) {
      if (payload[flag]) extras[flag] = true;
    }
  }
  // Decision checkpoints share one repo and differ only by subfolder.
  if (payload.model_subfolder) extras.model_subfolder = payload.model_subfolder;
  for (const [flag, fallback] of Object.entries(FLAG_DEFAULTS)) {
    const value = payload[flag as keyof typeof FLAG_DEFAULTS];
    if (value != null && value !== fallback) extras[flag] = value;
  }
  extras.train_on_completions = payload.train_on_completions;
  if (payload.format_type && payload.format_type !== "auto") {
    extras.format_type = payload.format_type;
  }
  if (
    payload.custom_format_mapping &&
    Object.keys(payload.custom_format_mapping).length > 0
  ) {
    extras.custom_format_mapping = payload.custom_format_mapping;
  }
  return extras;
}

/** `payload` is what Start would send; `datasetName` the uploaded file's display name. */
export function trainingApiRequestFromPayload(
  payload: TrainingStartRequest,
  datasetName: string | null,
): TrainingApiRequest {
  return {
    model: payload.model_name || null,
    datasetName: payload.local_datasets.length > 0 ? datasetName : null,
    datasetFromHub:
      payload.local_datasets.length === 0 && (Boolean(payload.hf_dataset) || payload.s3_config != null),
    nEpochs: payload.num_epochs,
    batchSize: payload.batch_size,
    seed: payload.random_seed,
    suffix: payload.project_name?.trim() || null,
    extras: trainingExtras(payload),
  };
}

export const EXAMPLE_TRAINING_API_REQUEST: TrainingApiRequest = {
  model: null,
  datasetName: null,
  datasetFromHub: false,
  nEpochs: 1,
  batchSize: 2,
  seed: 3407,
  suffix: null,
  extras: {
    "learning_rate": "2e-4",
    "load_in_4bit": true,
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 16,
  },
};

const j = (value: unknown) => JSON.stringify(value);

function py(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (Array.isArray(value)) return `[${value.map(py).join(", ")}]`;
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return `{${entries.map(([key, item]) => `${j(key)}: ${py(item)}`).join(", ")}}`;
  }
  return j(value);
}

const IDENTIFIER = /^[A-Za-z_$][\w$]*$/;
const jsKey = (key: string) => (IDENTIFIER.test(key) ? key : j(key));

/** The create-job fields in wire order, the training file excluded. */
function jobFields(request: TrainingApiRequest) {
  return {
    model: request.model ?? TRAINING_API_PLACEHOLDER_MODEL,
    hyperparameters: { "n_epochs": request.nEpochs, "batch_size": request.batchSize },
    seed: request.seed,
    suffix: request.suffix,
  };
}

function datasetFile(request: TrainingApiRequest): string {
  return request.datasetName || TRAINING_API_DEFAULT_DATASET;
}

function pythonSnippet(request: TrainingApiRequest, base: string, apiKey: string) {
  const { model, hyperparameters, seed, suffix } = jobFields(request);
  const args = [
    `    model=${j(model)},`,
    "    training_file=training_file.id,",
    `    hyperparameters=${py(hyperparameters)},`,
    `    seed=${seed},`,
  ];
  if (suffix) args.push(`    suffix=${j(suffix)},`);
  const extras = Object.entries(request.extras);
  if (extras.length > 0) {
    args.push(
      "    extra_body={",
      ...extras.map(([key, value]) => `        ${j(key)}: ${py(value)},`),
      "    },",
    );
  }
  return `import time

from openai import OpenAI

client = OpenAI(base_url=${j(`${base}/v1`)}, api_key=${j(apiKey)})

with open(${j(datasetFile(request))}, "rb") as dataset:
    training_file = client.files.create(file=dataset, purpose="fine-tune")

job = client.fine_tuning.jobs.create(
${args.join("\n")}
)
print(job.id, job.status)

seen = set()
while True:
    job = client.fine_tuning.jobs.retrieve(job.id)
    events = client.fine_tuning.jobs.list_events(job.id, limit=100).data
    for event in reversed(events):
        if event.id not in seen:
            seen.add(event.id)
            print(event.message)
    if job.status in ("succeeded", "failed", "cancelled"):
        break
    time.sleep(10)
if job.status != "succeeded":
    raise RuntimeError(job.error)

reply = client.chat.completions.create(
    model=job.fine_tuned_model,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)`;
}

function javascriptSnippet(
  request: TrainingApiRequest,
  base: string,
  apiKey: string,
) {
  const { model, hyperparameters, seed, suffix } = jobFields(request);
  const fields = [
    `  model: ${j(model)},`,
    "  training_file: trainingFile.id,",
    `  hyperparameters: { n_epochs: ${hyperparameters.n_epochs}, batch_size: ${hyperparameters.batch_size} },`,
    `  seed: ${seed},`,
  ];
  if (suffix) fields.push(`  suffix: ${j(suffix)},`);
  for (const [key, value] of Object.entries(request.extras)) {
    fields.push(`  ${jsKey(key)}: ${j(value)},`);
  }
  return `import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({ baseURL: ${j(`${base}/v1`)}, apiKey: ${j(apiKey)} });

const trainingFile = await client.files.create({
  file: fs.createReadStream(${j(datasetFile(request))}),
  purpose: "fine-tune",
});

let job = await client.fineTuning.jobs.create({
${fields.join("\n")}
});
console.log(job.id, job.status);

const seen = new Set();
while (true) {
  job = await client.fineTuning.jobs.retrieve(job.id);
  const events = await client.fineTuning.jobs.listEvents(job.id, { limit: 100 });
  for (const event of [...events.data].reverse()) {
    if (seen.has(event.id)) continue;
    seen.add(event.id);
    console.log(event.message);
  }
  if (["succeeded", "failed", "cancelled"].includes(job.status)) break;
  await new Promise((resolve) => setTimeout(resolve, 10000));
}
if (job.status !== "succeeded") throw new Error(JSON.stringify(job.error));

const reply = await client.chat.completions.create({
  model: job.fine_tuned_model,
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(reply.choices[0].message.content);`;
}

/** The create-job body without `training_file`, as every language sends it: extras flat, no `extra_body` key. */
export function trainingApiJobBody(
  request: TrainingApiRequest,
): Record<string, unknown> {
  const { model, hyperparameters, seed, suffix } = jobFields(request);
  const body: Record<string, unknown> = { model, hyperparameters, seed };
  if (suffix) body.suffix = suffix;
  return { ...body, ...request.extras };
}

function bodyLines(request: TrainingApiRequest): string[] {
  return Object.entries(trainingApiJobBody(request)).map(
    ([key, value]) => `  ${j(key)}: ${j(value)}`,
  );
}

function bashSnippet(request: TrainingApiRequest, base: string, apiKey: string) {
  const lines = bodyLines(request).map(shSingle);
  return `BASE='${shSingle(base)}'
KEY='${shSingle(apiKey)}'
FILE_ID=$(curl -sSf "$BASE/v1/files" \\
  -H "Authorization: Bearer $KEY" \\
  -F purpose=fine-tune \\
  -F 'file=@${shSingle(datasetFile(request))}' | jq -r .id)
JOB=$(curl -sSf "$BASE/v1/fine_tuning/jobs" \\
  -H "Authorization: Bearer $KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
  "training_file": "'"$FILE_ID"'",
${lines.join(",\n")}
}')
JOB_ID=$(jq -r .id <<<"$JOB")
STATUS=$(jq -r .status <<<"$JOB")
echo "$JOB_ID $STATUS"
SEEN=" "
while :; do
  JOB=$(curl -sSf "$BASE/v1/fine_tuning/jobs/$JOB_ID" -H "Authorization: Bearer $KEY")
  STATUS=$(jq -r .status <<<"$JOB")
  while read -r EVENT_ID MESSAGE; do
    case "$SEEN" in
      *" $EVENT_ID "*) ;;
      *) SEEN="$SEEN$EVENT_ID "; echo "$MESSAGE" ;;
    esac
  done < <(curl -sSf "$BASE/v1/fine_tuning/jobs/$JOB_ID/events?limit=100" \\
    -H "Authorization: Bearer $KEY" \\
    | jq -r '.data | reverse[] | "\\(.id) \\(.message | gsub("\\n"; " "))"')
  case "$STATUS" in succeeded|failed|cancelled|"") break ;; esac
  sleep 10
done
if [ "$STATUS" = succeeded ]; then
  MODEL=$(jq -r .fine_tuned_model <<<"$JOB")
  curl -sSf "$BASE/v1/chat/completions" \\
    -H "Authorization: Bearer $KEY" \\
    -H "Content-Type: application/json" \\
    -d "$(jq -n --arg model "$MODEL" '{model: $model, messages: [{role: "user", content: "Hello!"}]}')" \\
    | jq -r '.choices[0].message.content'
else
  jq .error <<<"$JOB"
fi`;
}

function powershellSnippet(
  request: TrainingApiRequest,
  base: string,
  apiKey: string,
) {
  return `$base = '${psSingle(base)}'
$key = '${psSingle(apiKey)}'
$file = curl.exe -sSf "$base/v1/files" \`
  -H "Authorization: Bearer $key" \`
  -F purpose=fine-tune \`
  -F 'file=@${psSingle(datasetFile(request))}' | ConvertFrom-Json
$settings = @'
${bodyLines(request).join(",\n")}
'@
$body = "{\`n  \`"training_file\`": \`"$($file.id)\`",\`n$settings\`n}"
[IO.File]::WriteAllText("$PWD\\body.json", $body)
$job = curl.exe -sSf "$base/v1/fine_tuning/jobs" \`
  -H "Authorization: Bearer $key" \`
  -H "Content-Type: application/json" \`
  -d "@body.json" | ConvertFrom-Json
"$($job.id) $($job.status)"
$seen = @{}
while ($true) {
  $job = curl.exe -sSf "$base/v1/fine_tuning/jobs/$($job.id)" -H "Authorization: Bearer $key" | ConvertFrom-Json
  if (-not $job) { break }
  $events = @((curl.exe -sSf "$base/v1/fine_tuning/jobs/$($job.id)/events?limit=100" -H "Authorization: Bearer $key" | ConvertFrom-Json).data)
  [array]::Reverse($events)
  foreach ($item in $events) {
    if (-not $seen.ContainsKey($item.id)) {
      $seen[$item.id] = $true
      $item.message
    }
  }
  if ($job.status -in "succeeded", "failed", "cancelled") { break }
  Start-Sleep -Seconds 10
}
if ($job.status -eq "succeeded") {
  $chat = @{ model = $job.fine_tuned_model; messages = @(@{ role = "user"; content = "Hello!" }) } | ConvertTo-Json -Depth 5
  [IO.File]::WriteAllText("$PWD\\chat.json", $chat)
  $reply = curl.exe -sSf "$base/v1/chat/completions" \`
    -H "Authorization: Bearer $key" \`
    -H "Content-Type: application/json" \`
    -d "@chat.json" | ConvertFrom-Json
  $reply.choices[0].message.content
} else {
  $job.error
}`;
}

/** Upload, create, follow the job's events until it ends, then chat with the trained model. */
export function buildTrainingApiSnippet(
  request: TrainingApiRequest,
  { base, apiKey, lang, os }: TrainingApiSnippetInput,
): string {
  if (lang === "python") return pythonSnippet(request, base, apiKey);
  if (lang === "javascript") return javascriptSnippet(request, base, apiKey);
  return os === "windows"
    ? powershellSnippet(request, base, apiKey)
    : bashSnippet(request, base, apiKey);
}

export function buildTrainingCancelSnippet({
  base,
  apiKey,
  lang,
  os,
}: TrainingApiSnippetInput): string {
  const jobId = "ftjob-YOUR_JOB_ID";
  if (lang === "python") {
    return `from openai import OpenAI

client = OpenAI(base_url=${j(`${base}/v1`)}, api_key=${j(apiKey)})
print(client.fine_tuning.jobs.cancel(${j(jobId)}).status)`;
  }
  if (lang === "javascript") {
    return `import OpenAI from "openai";

const client = new OpenAI({ baseURL: ${j(`${base}/v1`)}, apiKey: ${j(apiKey)} });
console.log((await client.fineTuning.jobs.cancel(${j(jobId)})).status);`;
  }
  if (os === "windows") {
    return `$base = '${psSingle(base)}'
$key = '${psSingle(apiKey)}'
(curl.exe -sSf -X POST "$base/v1/fine_tuning/jobs/${jobId}/cancel" -H "Authorization: Bearer $key" | ConvertFrom-Json).status`;
  }
  return `BASE='${shSingle(base)}'
KEY='${shSingle(apiKey)}'
curl -sSf -X POST "$BASE/v1/fine_tuning/jobs/${jobId}/cancel" -H "Authorization: Bearer $KEY" | jq -r .status`;
}
