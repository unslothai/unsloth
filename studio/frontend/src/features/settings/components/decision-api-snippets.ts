// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ApiExampleLanguage,
  ApiExampleOs,
} from "../../api-monitor/api-sampling-example.ts";
import { DEFAULT_MODEL, requestText } from "../lib/decision-request.ts";
import type {
  KeylessApiAccessExposure,
  KeylessApiAccessScope,
} from "../api/keyless-api-access";
import { psSingle, shSingle } from "./agent-command.ts";
import { keylessBaseEligible } from "./keyless-example-eligibility.ts";

const KEY_PLACEHOLDER = "sk-unsloth-YOUR_KEY";
const KEYLESS_KEY_PLACEHOLDER = "not-needed";
const STATE = "I was charged twice. Please fix this ASAP.";
const INSTRUCTIONS = "What is this ticket about?";
const CRITERIA = {
  billing: "Charges, refunds or invoices",
  technical: "Bugs or errors in the product",
  other: "Anything else",
};

export function decisionApiKey(
  apiKey: string | null | undefined,
  server: {
    base: string;
    tunnel: boolean;
    scope: KeylessApiAccessScope;
    exposure: KeylessApiAccessExposure | null;
  },
): string {
  if (apiKey) return apiKey;
  const keyless =
    !server.tunnel &&
    keylessBaseEligible(server.base, server.scope, server.exposure);
  return keyless ? KEYLESS_KEY_PLACEHOLDER : KEY_PLACEHOLDER;
}

// mcp_url always names 127.0.0.1, which is the wrong machine over LAN or a tunnel.
export function decisionMcpUrl(base: string, mcpUrl: string): string {
  return `${base}${new URL(mcpUrl).pathname}`;
}

export function buildDecisionCurl({
  base,
  apiKey,
  body,
  os,
}: {
  base: string;
  apiKey: string;
  body: string;
  os: ApiExampleOs;
}): string {
  if (os === "windows") {
    // Set-Content writes ascii: \u escapes survive it, raw non-English text does not.
    const ascii = body.replace(
      /[\u0080-\uffff]/g,
      (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`,
    );
    return `$body = '${psSingle(ascii)}'
Set-Content -Path body.json -Value $body -Encoding ascii
curl.exe ${base}/v1/systemone \`
  -H "Authorization: Bearer ${apiKey}" \`
  -H "Content-Type: application/json" \`
  -d "@body.json"`;
  }
  return `curl ${base}/v1/systemone \\
  -H "Authorization: Bearer ${apiKey}" \\
  -H "Content-Type: application/json" \\
  -d '${shSingle(body)}'`;
}

function pythonExample(base: string, apiKey: string): string {
  return `# pip install typesafe-sdk
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(base_url=${JSON.stringify(base)}, api_key=${JSON.stringify(apiKey)}) as client:
    response = client.system_one(
        state=${JSON.stringify(STATE)},
        questions={
            "category": Choice(
                instructions=${JSON.stringify(INSTRUCTIONS)},
                criteria=${JSON.stringify(CRITERIA, null, 4).replace(/\n/g, `\n${" ".repeat(16)}`)},
            ),
        },
    )

print(response.choices["category"].choice)`;
}

function javascriptExample(base: string, apiKey: string): string {
  return `// npm install @typesafe-ai/sdk
import { choice, TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({
  baseURL: ${JSON.stringify(base)},
  apiKey: ${JSON.stringify(apiKey)},
});
const response = await client.systemOne({
  state: ${JSON.stringify(STATE)},
  questions: {
    category: choice(${JSON.stringify(INSTRUCTIONS)}, ${JSON.stringify(CRITERIA, null, 2).replace(/\n/g, "\n    ")}),
  },
});

console.log(response.answers.category.choice);`;
}

export function buildDecisionApiSnippet({
  base,
  apiKey,
  language,
  os,
}: {
  base: string;
  apiKey: string;
  language: ApiExampleLanguage;
  os: ApiExampleOs;
}): string {
  if (language === "python") return pythonExample(base, apiKey);
  if (language === "javascript") return javascriptExample(base, apiKey);
  return buildDecisionCurl({
    base,
    apiKey,
    os,
    body: requestText({
      state: STATE,
      model: DEFAULT_MODEL,
      questions: {
        category: {
          type: "choice",
          instructions: INSTRUCTIONS,
          criteria: CRITERIA,
        },
      },
    }),
  });
}
