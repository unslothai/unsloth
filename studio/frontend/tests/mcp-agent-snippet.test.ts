// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings > API shows how to point each coding agent at Studio's MCP endpoint. Pinned
// here: every agent's format as its own docs give it, the URL always ends in /mcp/, the
// PowerShell variant reads the environment, and a key appears only when one is passed.

import assert from "node:assert/strict";
import test from "node:test";

import yaml from "js-yaml";

import { SUPPORTED_AGENTS } from "../src/features/settings/components/coding-agent-list.ts";
import {
  MCP_SHELL_AGENT_IDS,
  buildMcpSnippet,
  mcpEndpointUrl,
} from "../src/features/settings/components/mcp-agent-snippet.ts";

const LOCAL = "http://127.0.0.1:8888";
const TUNNEL = "https://shed-topic-remark-arguments.trycloudflare.com/";
const LAN = "http://192.168.1.20:8888";
const KEY = "sk-unsloth-0123456789abcdef0123456789abcdef";

// dsh evaluates `!!js` scalars; the test only needs to read them back.
const DSH_SCHEMA = yaml.DEFAULT_SCHEMA.extend([
  new yaml.Type("tag:yaml.org,2002:js", {
    kind: "scalar",
    construct: (source: string) => ({ js: source }),
  }),
]);

function snippet(
  agent: string,
  os: "unix" | "windows" = "unix",
  key: string | null = null,
  base: string = LOCAL,
) {
  const result = buildMcpSnippet(agent, base, os, key);
  assert.ok(result, `${agent} has a snippet`);
  return result;
}

test("every listed agent has a snippet and only the CLIs get an OS choice", () => {
  for (const agent of SUPPORTED_AGENTS) {
    assert.ok(buildMcpSnippet(agent.id, LOCAL, "unix"), agent.id);
  }
  assert.deepEqual([...MCP_SHELL_AGENT_IDS], ["claude", "codex"]);
  assert.equal(buildMcpSnippet("pi", LOCAL, "unix"), null);
});

test("the endpoint is always the origin plus /mcp/", () => {
  assert.equal(mcpEndpointUrl(LOCAL), "http://127.0.0.1:8888/mcp/");
  assert.equal(
    mcpEndpointUrl(TUNNEL),
    "https://shed-topic-remark-arguments.trycloudflare.com/mcp/",
  );
  assert.equal(mcpEndpointUrl(LAN), "http://192.168.1.20:8888/mcp/");
  assert.equal(
    mcpEndpointUrl("http://127.0.0.1:8888/mcp"),
    "http://127.0.0.1:8888/mcp/",
  );
  assert.equal(
    mcpEndpointUrl("https://studio.example/some/path?x=1#y"),
    "https://studio.example/mcp/",
  );
  assert.equal(mcpEndpointUrl(""), null);
  assert.equal(mcpEndpointUrl("not a url"), null);
  assert.equal(mcpEndpointUrl("file:///etc/passwd"), null);
  assert.equal(buildMcpSnippet("claude", "not a url", "unix"), null);
});

test("Claude Code reads the key from the shell it runs in", () => {
  assert.equal(
    snippet("claude", "unix").text,
    'claude mcp add --transport http unsloth-studio http://127.0.0.1:8888/mcp/ --header "Authorization: Bearer $UNSLOTH_API_KEY"',
  );
  assert.equal(
    snippet("claude", "windows").text,
    'claude mcp add --transport http unsloth-studio http://127.0.0.1:8888/mcp/ --header "Authorization: Bearer $env:UNSLOTH_API_KEY"',
  );
  assert.equal(snippet("claude").readsKeyEnv, true);
});

test("Claude Code quotes a real key for each shell", () => {
  const unix = snippet("claude", "unix", KEY);
  assert.equal(
    unix.text,
    `claude mcp add --transport http unsloth-studio http://127.0.0.1:8888/mcp/ --header 'Authorization: Bearer ${KEY}'`,
  );
  assert.equal(unix.readsKeyEnv, false);
  assert.ok(
    snippet("claude", "windows", KEY).text.endsWith(
      `--header 'Authorization: Bearer ${KEY}'`,
    ),
  );
  assert.ok(
    snippet("claude", "unix", "it's").text.endsWith(
      `--header 'Authorization: Bearer it'\\''s'`,
    ),
  );
  assert.ok(
    snippet("claude", "windows", "it's").text.endsWith(
      `--header 'Authorization: Bearer it''s'`,
    ),
  );
});

test("Codex names the variable and offers a longer tool timeout", () => {
  const text = snippet("codex").text;
  assert.equal(
    text.split("\n")[0],
    "codex mcp add unsloth-studio --url http://127.0.0.1:8888/mcp/ --bearer-token-env-var UNSLOTH_API_KEY",
  );
  assert.match(text, /^# tool_timeout_sec = 300$/m);
  assert.equal(snippet("codex", "windows").text, text);
  // Codex cannot take the key itself.
  assert.equal(snippet("codex", "unix", KEY).text, text);
  assert.equal(snippet("codex", "unix", KEY).readsKeyEnv, true);
});

test("OpenCode gets valid JSON with OAuth off and an env reference", () => {
  const parsed = JSON.parse(snippet("opencode").text);
  assert.deepEqual(parsed, {
    mcp: {
      "unsloth-studio": {
        type: "remote",
        url: "http://127.0.0.1:8888/mcp/",
        oauth: false,
        headers: { Authorization: "Bearer {env:UNSLOTH_API_KEY}" },
      },
    },
  });
  const withKey = JSON.parse(snippet("opencode", "unix", 'a"b').text);
  assert.equal(
    withKey.mcp["unsloth-studio"].headers.Authorization,
    'Bearer a"b',
  );
});

test("OpenClaw gets valid JSON with the transport set explicitly", () => {
  const parsed = JSON.parse(snippet("openclaw").text);
  assert.deepEqual(parsed, {
    mcp: {
      servers: {
        "unsloth-studio": {
          url: "http://127.0.0.1:8888/mcp/",
          transport: "streamable-http",
          headers: { Authorization: "Bearer ${UNSLOTH_API_KEY}" },
        },
      },
    },
  });
  assert.equal(
    JSON.parse(snippet("openclaw", "unix", KEY).text).mcp.servers[
      "unsloth-studio"
    ].headers.Authorization,
    `Bearer ${KEY}`,
  );
});

test("Hermes gets YAML with ${UNSLOTH_API_KEY} and the reload step", () => {
  const text = snippet("hermes").text;
  assert.ok(text.includes('Authorization: "Bearer ${UNSLOTH_API_KEY}"'));
  assert.match(text, /\/reload-mcp/);
  assert.deepEqual(yaml.load(text), {
    mcp_servers: {
      unsloth_studio: {
        url: "http://127.0.0.1:8888/mcp/",
        headers: { Authorization: "Bearer ${UNSLOTH_API_KEY}" },
      },
    },
  });
  const tricky = 'k"e\\y';
  assert.equal(
    (
      yaml.load(snippet("hermes", "unix", tricky).text) as {
        mcp_servers: {
          unsloth_studio: { headers: { Authorization: string } };
        };
      }
    ).mcp_servers.unsloth_studio.headers.Authorization,
    `Bearer ${tricky}`,
  );
});

test("Mistral Vibe uses the static auth block", () => {
  assert.equal(
    snippet("vibe").text,
    [
      "[[mcp_servers]]",
      'name = "unsloth_studio"',
      'transport = "streamable-http"',
      'url = "http://127.0.0.1:8888/mcp/"',
      "",
      "[mcp_servers.auth]",
      'type = "static"',
      'api_key_env = "UNSLOTH_API_KEY"',
      'api_key_header = "Authorization"',
      'api_key_format = "Bearer {token}"',
    ].join("\n"),
  );
  // Legacy top-level keys mixed with [auth] are an error in Vibe.
  const text = snippet("vibe").text;
  assert.doesNotMatch(text, /^headers/m);
  assert.ok(text.indexOf("api_key_env") > text.indexOf("[mcp_servers.auth]"));
  assert.equal(snippet("vibe", "unix", KEY).text, snippet("vibe").text);
});

test("DeepSeek Harness evaluates the key with !!js, or takes a plain string", () => {
  const text = snippet("dsh").text;
  assert.ok(
    text.includes(
      "Authorization: !!js '`Bearer ${process.env.UNSLOTH_API_KEY}`'",
    ),
  );
  assert.deepEqual(yaml.load(text, { schema: DSH_SCHEMA }), [
    {
      insert: [
        {
          id: "mcp-unsloth-studio",
          name: "@deepseek-ai/dsh-mcp-client",
          config: {
            serverName: "unsloth-studio",
            transport: "streamable-http",
            url: "http://127.0.0.1:8888/mcp/",
            headers: {
              Authorization: {
                js: "`Bearer ${process.env.UNSLOTH_API_KEY}`",
              },
            },
          },
        },
      ],
    },
  ]);
  const withKey = snippet("dsh", "unix", KEY).text;
  assert.doesNotMatch(withKey, /!!js/);
  assert.ok(withKey.includes(`Authorization: "Bearer ${KEY}"`));
});

test("no snippet carries a key unless one is passed", () => {
  for (const agent of SUPPORTED_AGENTS) {
    for (const os of ["unix", "windows"] as const) {
      for (const base of [LOCAL, TUNNEL, LAN]) {
        for (const key of [null, ""]) {
          const result = snippet(agent.id, os, key, base);
          assert.doesNotMatch(result.text, /sk-/, `${agent.id} ${os}`);
          assert.equal(result.readsKeyEnv, true, `${agent.id} ${os}`);
          assert.match(result.text, /UNSLOTH_API_KEY/);
          assert.ok(
            result.text.includes(mcpEndpointUrl(base) as string),
            `${agent.id} ${os} ${base}`,
          );
        }
      }
    }
  }
});

test("config-file agents say which file to edit", () => {
  assert.equal(snippet("claude").configPath, null);
  assert.equal(snippet("codex").configPath, null);
  assert.equal(
    snippet("opencode").configPath,
    "~/.config/opencode/opencode.json",
  );
  assert.equal(snippet("hermes").configPath, "~/.hermes/config.yaml");
  assert.equal(snippet("openclaw").configPath, "~/.openclaw/openclaw.json");
  assert.equal(snippet("vibe").configPath, "~/.vibe/config.toml");
  assert.equal(snippet("dsh").configPath, "~/.dsh/cordis.patch.yml");
});
