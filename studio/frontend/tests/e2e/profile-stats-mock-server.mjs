// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import http from "node:http";

const PORT = Number(process.env.MOCK_API_PORT ?? 8888);

function isoDay(date) {
  const year = date.getFullYear();
  const month = `${date.getMonth() + 1}`.padStart(2, "0");
  const day = `${date.getDate()}`.padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function buildDailySeries() {
  const today = new Date();
  const days = 366;
  const series = [];
  for (let offset = days - 1; offset >= 0; offset -= 1) {
    const date = new Date(today);
    date.setDate(today.getDate() - offset);
    const key = isoDay(date);
    let tokens = 0;
    let messages = 0;
    if (offset <= 1) {
      tokens = 100;
      messages = 4;
    } else if (offset <= 8) {
      tokens = 10;
      messages = 1;
    }
    series.push({ date: key, tokens, messages, chats: messages > 0 ? 1 : 0 });
  }
  return series;
}

const daily = buildDailySeries();
const totalTokens = daily.reduce((sum, day) => sum + day.tokens, 0);

const profileStats = {
  generatedAt: Date.now(),
  days: 366,
  totals: {
    threads: 3,
    messages: daily.reduce((sum, day) => sum + day.messages, 0),
    userMessages: 20,
    assistantMessages: 25,
    promptTokens: Math.round(totalTokens * 0.4),
    completionTokens: Math.round(totalTokens * 0.6),
    totalTokens,
    cachedTokens: 0,
    toolCalls: 0,
    attachments: 0,
    activeDays: daily.filter((day) => day.tokens > 0).length,
    chatSeconds: 524,
  },
  streak: { current: 1, longest: 2, lastActiveDay: daily.at(-1)?.date ?? null },
  peakDay: { date: daily.at(-1)?.date ?? "2026-08-20", tokens: 100 },
  longestChat: {
    threadId: "thread-1",
    title: "Test chat",
    seconds: 524,
    messages: 8,
  },
  daily,
  models: [
    {
      id: "unsloth/Qwen3-8B-GGUF",
      label: "Qwen3 8B",
      messages: 20,
      tokens: totalTokens,
    },
  ],
  speed: {
    averageTokensPerSecond: 42.5,
    bestTokensPerSecond: 88.0,
    bestTokensPerSecondModel: "unsloth/Qwen3-8B-GGUF",
    averageResponseMs: 1200,
    averageFirstTokenMs: 180,
    samples: 10,
  },
  training: {
    runs: 0,
    completed: 0,
    steps: 0,
    tokens: 0,
    seconds: 0,
    models: 0,
    datasets: 0,
    bestLoss: null,
    recent: [],
  },
};

function json(res, status, body) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
  });
  res.end(JSON.stringify(body));
}

const nowSeconds = () => Date.now() / 1000;

function mockPayload(path) {
  if (path === "/api/inference/monitor") {
    return { entries: [], server_time: nowSeconds() };
  }
  if (path === "/api/inference/status") {
    return {
      loaded: false,
      model: null,
      device_type: "cuda",
      parallel_slots: 1,
    };
  }
  if (path === "/api/inference/active-generations") {
    return { generations: [] };
  }
  if (path === "/api/inference/load-progress") {
    return { active: false };
  }
  if (path === "/api/chat/projects") {
    return [];
  }
  if (path === "/api/chat/settings") {
    return {};
  }
  if (path === "/api/chat/count") {
    return { threads: 0, messages: 0 };
  }
  if (path === "/api/chat/export") {
    return { threads: [] };
  }
  if (path === "/api/chat/import-ledger") {
    return { entries: [] };
  }
  if (path === "/api/models/list") {
    return { models: [], default_models: [] };
  }
  if (path === "/api/models/local") {
    return { models: [] };
  }
  if (path === "/api/models/scan-folders") {
    return { folders: [] };
  }
  if (path === "/api/models/recommended-folders") {
    return { folders: [] };
  }
  if (path === "/api/hub/cached-gguf") {
    return { variants: [] };
  }
  if (path === "/api/hub/cached-models") {
    return { models: [] };
  }
  if (path === "/api/providers/") {
    return [];
  }
  if (path === "/api/providers/registry") {
    return { providers: [] };
  }
  if (path.startsWith("/api/personalization")) {
    return {};
  }
  return {};
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url ?? "/", `http://127.0.0.1:${PORT}`);
  const path = url.pathname;

  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "*",
    });
    res.end();
    return;
  }

  if (path === "/api/auth/status") {
    return json(res, 200, {
      initialized: true,
      requires_password_change: false,
    });
  }

  if (path === "/api/auth/refresh" && req.method === "POST") {
    return json(res, 200, {
      access_token: "test-access-token",
      refresh_token: "test-refresh-token",
      must_change_password: false,
    });
  }

  if (path === "/api/profile/stats") {
    return json(res, 200, profileStats);
  }

  if (path === "/api/health") {
    return json(res, 200, {
      status: "ok",
      device_type: "cuda",
      chat_only: false,
      video_supported: true,
      images_supported: true,
    });
  }

  if (path.startsWith("/api/")) {
    return json(res, 200, mockPayload(path));
  }

  res.writeHead(404);
  res.end("not found");
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`profile-stats mock API on http://127.0.0.1:${PORT}`);
});
