// SPDX-License-Identifier: AGPL-3.0-only
import { authFetch, getAuthSessionEpoch } from "@/features/auth";
// eslint-disable-next-line no-restricted-imports -- API leaf avoids pulling the Images page into chat.
import {
  chatImageSwapSelection,
  getDiffusionStatus,
  parseJson,
  type DiffusionGenerateResponse,
} from "@/features/images/api";
import { createNavigationNonce } from "@/lib/navigation-nonce";
import { fileToBase64 } from "@/lib/audio-utils";
import type {
  OpenAIChatChunk,
  OpenAIChatCompletionsRequest,
  OpenAIChatMessage,
} from "../types/api";
import {
  getInferenceStatus,
  parseJsonOrThrow,
  streamChatCompletions,
} from "./chat-api";

type Selection = NonNullable<ReturnType<typeof chatImageSwapSelection>> & {
  textId: string;
};
type ToolChunk = OpenAIChatChunk & { _toolEvent: Record<string, unknown> };

export async function imageSwapForRequest(
  body: OpenAIChatCompletionsRequest,
  signal: AbortSignal,
) {
  if (
    body.enabled_tools?.length !== 1 ||
    body.enabled_tools[0] !== "generate_image" ||
    body.mcp_enabled ||
    body.deep_research_armed ||
    body.video_base64 ||
    body.audio_base64 ||
    body.extra_audio_base64?.length ||
    body.provider_id ||
    body.external_model
  )
    return null;
  const selection = chatImageSwapSelection();
  if (!selection) return null;
  try {
    const state = await getDiffusionStatus(signal);
    if (state.loaded || state.load_id) return null;
    const chat = await getInferenceStatus(signal);
    if (
      chat.loaded.length !== 1 ||
      chat.loading.length ||
      chat.is_audio ||
      chat.is_diffusion ||
      chat.is_npu ||
      chat.is_mlx ||
      (chat.engine && chat.engine !== "auto")
    )
      return null;
    const textId = chat.model_identifier ?? chat.active_model;
    return textId ? { ...selection, textId } : null;
  } catch {
    signal.throwIfAborted();
    return null;
  }
}

function sameSession(selection: Selection) {
  if (getAuthSessionEpoch() !== selection.epoch)
    throw new Error("The chat session changed.");
}

async function render(
  selection: Selection,
  args: { prompt: string; width?: number; height?: number; seed?: number },
  signal: AbortSignal,
) {
  sameSession(selection);
  const swapId = createNavigationNonce();
  const stop = () => {
    if (getAuthSessionEpoch() !== selection.epoch) return;
    void authFetch(
      `/api/inference/images/chat-swap/${swapId}/cancel`,
      { method: "POST" },
      { retryNetworkErrors: false },
    ).catch(() => {});
  };
  signal.throwIfAborted();
  signal.addEventListener("abort", stop, { once: true });
  try {
    const response = await authFetch(
      "/api/inference/images/chat-swap",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal,
        body: JSON.stringify({
          swap_id: swapId,
          text_id: selection.textId,
          image_load: selection.image,
          prompt: args.prompt,
          width: args.width,
          height: args.height,
          seed: args.seed,
        }),
      },
      { retryNetworkErrors: false },
    );
    const { images } = await parseJsonOrThrow<DiffusionGenerateResponse>(
      response,
      "Chat image generation",
    );
    if (!images?.length) throw new Error("The generator returned no image.");
    return images[0];
  } finally {
    signal.removeEventListener("abort", stop);
  }
}

export async function* streamImageSwap(
  body: OpenAIChatCompletionsRequest,
  selection: Selection,
  signal: AbortSignal,
  contextLength: number | null,
) {
  const tool = {
    type: "function",
    function: {
      name: "generate_image",
      description: "Generate one image from a complete visual prompt.",
      parameters: {
        type: "object",
        properties: {
          prompt: { type: "string" },
          width: { type: "integer" },
          height: { type: "integer" },
          seed: { type: "integer" },
        },
        required: ["prompt"],
      },
    },
  };
  const response = await authFetch(
    "/v1/chat/completions",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        ...body,
        stream: false,
        stream_options: undefined,
        enable_tools: false,
        enabled_tools: undefined,
        // This request only proposes a call. The client confirms before rendering.
        confirm_tool_calls: false,
        mcp_enabled: false,
        tools: [tool],
        tool_choice: "auto",
        parallel_tool_calls: false,
      }),
    },
    { retryNetworkErrors: false },
  );
  const first = await parseJson<{
    choices: {
      message: OpenAIChatMessage & { reasoning_content?: string };
      finish_reason: string;
    }[];
    usage?: OpenAIChatChunk["usage"];
  }>(response);
  const choice = first.choices[0];
  const calls = choice.message.tool_calls;
  if (!calls?.length) {
    yield {
      choices: [
        { delta: choice.message, finish_reason: choice.finish_reason },
      ],
      usage: first.usage,
    } as OpenAIChatChunk;
    return;
  }
  const delta = {
    content: choice.message.content,
    reasoning_content: choice.message.reasoning_content,
  };
  if (delta.content || delta.reasoning_content)
    yield { choices: [{ delta, finish_reason: null }] } as OpenAIChatChunk;
  const [call] = calls;
  if (
    calls.length !== 1 ||
    !call.id ||
    call.function?.name !== "generate_image"
  )
    throw new Error("Expected a single image tool call.");
  const args = JSON.parse(call.function.arguments ?? "{}");
  const prompt: unknown = args?.prompt;
  if (typeof prompt !== "string" || !prompt.trim())
    throw new Error("The image tool needs a prompt.");
  const event = {
    tool_name: "generate_image",
    tool_call_id: call.id,
    arguments: { prompt },
  };
  yield { _toolEvent: { ...event, type: "tool_start" } } as ToolChunk;
  let result = "Image generation was declined.";
  const ask =
    body.permission_mode === "ask" || body.confirm_tool_calls === true;
  let imageBase64: string | undefined;
  if (
    !ask ||
    window.confirm(
      `Generate an image with ${selection.image.model_path} and restore ${selection.textId}?\n\n${prompt}`,
    )
  ) {
    try {
      const image = await render(selection, args, signal);
      result = `The image is displayed in chat and saved to the gallery. Image ID: ${image.id}. File: ${image.url}. Size: ${image.width}x${image.height}. Seed: ${image.seed}.`;
      sameSession(selection);
      const file = await authFetch(
        `/api/inference/images/gallery/${encodeURIComponent(image.id)}/file`,
        { signal },
      );
      if (file.ok)
        imageBase64 = await fileToBase64(
          new File([await file.blob()], "image.png", { type: "image/png" }),
        );
    } catch (error) {
      signal.throwIfAborted();
      sameSession(selection);
      const chat = await getInferenceStatus(signal);
      if (![chat.model_identifier, chat.active_model].includes(selection.textId))
        throw error;
      result = `Image tool error: ${error instanceof Error ? error.message : String(error)}`;
    }
  }
  yield {
    _toolEvent: {
      ...event,
      type: "tool_end",
      result,
      image_b64: imageBase64,
      image_mime: "image/png",
    },
  } as ToolChunk;
  signal.throwIfAborted();
  yield* streamChatCompletions(
    {
      ...body,
      enabled_tools: [],
      enable_tools: false,
      messages: [
        ...body.messages,
        choice.message,
        {
          role: "tool",
          tool_call_id: call.id,
          content: result,
        },
      ],
    },
    signal,
    contextLength,
  );
}
