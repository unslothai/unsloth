// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { psSingle, shSingle } from "../settings/components/agent-command.ts";

export type ApiSamplingDefaults = Partial<{
  temperature: number;
  top_p: number;
  top_k: number;
  min_p: number;
  repetition_penalty: number;
  presence_penalty: number;
}>;

export type ApiExampleLanguage = "curl" | "python" | "javascript";
export type ApiExampleOs = "unix" | "windows";

export const API_EXAMPLE_LANGUAGES: {
  id: ApiExampleLanguage;
  label: string;
}[] = [
  { id: "curl", label: "curl" },
  { id: "python", label: "Python" },
  { id: "javascript", label: "JavaScript" },
];

export const API_EXAMPLE_KEY = "sk-unsloth-YOUR_KEY";
const PROMPT = "Hello!";

// The OpenAI client takes these as kwargs; the Unsloth extensions only go through extra_body.
const CLIENT_FIELDS = ["temperature", "top_p", "presence_penalty"] as const;
const EXTRA_FIELDS = ["top_k", "min_p", "repetition_penalty"] as const;
const ALL_FIELDS = [...CLIENT_FIELDS, ...EXTRA_FIELDS];

export function apiExampleModelName(
  id: string,
  ggufVariant: string | null,
): string {
  return ggufVariant && !id.includes(":") ? `${id}:${ggufVariant}` : id;
}

/** The language prefix of the API tab's stored pick ("pythonTools" -> "python"). */
export const LANGUAGE_PREFIX_RE = /^(curl|python|javascript)/;

export function apiExampleLanguage(
  stored: string | null | undefined,
  fallback: ApiExampleLanguage,
): ApiExampleLanguage {
  const match = stored?.match(LANGUAGE_PREFIX_RE);
  return match ? (match[1] as ApiExampleLanguage) : fallback;
}

interface ApiExampleInput {
  /** Ends in /v1, as the OpenAI clients take it. */
  baseUrl: string;
  model: string;
  sampling: ApiSamplingDefaults;
  apiKey?: string;
}

function samplingBody(sampling: ApiSamplingDefaults): Record<string, number> {
  const body: Record<string, number> = {};
  for (const field of ALL_FIELDS) {
    const value = sampling[field];
    if (value != null) body[field] = value;
  }
  return body;
}

export function buildApiSamplingExample({
  baseUrl,
  model,
  sampling,
  apiKey = API_EXAMPLE_KEY,
}: ApiExampleInput): string {
  const args = CLIENT_FIELDS.filter((field) => sampling[field] != null).map(
    (field) => `    ${field}=${sampling[field]},`,
  );
  const extras = EXTRA_FIELDS.filter((field) => sampling[field] != null).map(
    (field) => `        "${field}": ${sampling[field]},`,
  );
  return [
    "from openai import OpenAI",
    "",
    "client = OpenAI(",
    `    base_url=${JSON.stringify(baseUrl)},`,
    `    api_key=${JSON.stringify(apiKey)},`,
    ")",
    "",
    "response = client.chat.completions.create(",
    `    model=${JSON.stringify(model)},`,
    `    messages=[{"role": "user", "content": ${JSON.stringify(PROMPT)}}],`,
    ...args,
    ...(extras.length > 0 ? ["    extra_body={", ...extras, "    },"] : []),
    ")",
    "print(response.choices[0].message.content)",
  ].join("\n");
}

function javascriptExample({
  baseUrl,
  model,
  sampling,
  apiKey = API_EXAMPLE_KEY,
}: ApiExampleInput): string {
  // The JS SDK forwards unknown options into the body, so the Unsloth fields sit at the top level.
  const options = Object.entries(samplingBody(sampling)).map(
    ([field, value]) => `  ${field}: ${value},`,
  );
  return [
    'import OpenAI from "openai";',
    "",
    "const client = new OpenAI({",
    `  baseURL: ${JSON.stringify(baseUrl)},`,
    `  apiKey: ${JSON.stringify(apiKey)},`,
    "});",
    "",
    "const response = await client.chat.completions.create({",
    `  model: ${JSON.stringify(model)},`,
    `  messages: [{ role: "user", content: ${JSON.stringify(PROMPT)} }],`,
    ...options,
    "});",
    "console.log(response.choices[0].message.content);",
  ].join("\n");
}

/** One field per line, the messages kept on one, as the API tab's curl prints them. */
function requestBody(
  model: string,
  sampling: ApiSamplingDefaults,
  indent: string,
): string {
  const fields = [
    `"model": ${JSON.stringify(model)}`,
    `"messages": [{"role": "user", "content": ${JSON.stringify(PROMPT)}}]`,
    ...Object.entries(samplingBody(sampling)).map(
      ([field, value]) => `"${field}": ${value}`,
    ),
  ];
  return `{\n${fields.map((f) => `${indent}  ${f}`).join(",\n")}\n${indent}}`;
}

function curlExample(
  { baseUrl, model, sampling, apiKey = API_EXAMPLE_KEY }: ApiExampleInput,
  os: ApiExampleOs,
): string {
  if (os === "windows") {
    // PowerShell mangles inline JSON quotes on the way to curl.exe, so the body goes through a file.
    return `$body = '${psSingle(requestBody(model, sampling, ""))}'
Set-Content -Path body.json -Value $body -Encoding ascii
curl.exe ${baseUrl}/chat/completions \`
  -H "Authorization: Bearer ${apiKey}" \`
  -H "Content-Type: application/json" \`
  -d "@body.json"`;
  }
  return `curl ${baseUrl}/chat/completions \\
  -H "Authorization: Bearer ${apiKey}" \\
  -H "Content-Type: application/json" \\
  -d '${shSingle(requestBody(model, sampling, "  "))}'`;
}

export function buildApiExample(
  input: ApiExampleInput & { language: ApiExampleLanguage; os: ApiExampleOs },
): string {
  if (input.language === "python") return buildApiSamplingExample(input);
  if (input.language === "javascript") return javascriptExample(input);
  return curlExample(input, input.os);
}

/** Shiki's name for each example language. */
export function apiExampleShikiLanguage(
  language: ApiExampleLanguage,
  os: ApiExampleOs,
): string {
  if (language === "curl") return os === "windows" ? "powershell" : "bash";
  return language;
}
