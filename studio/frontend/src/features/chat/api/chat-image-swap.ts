// SPDX-License-Identifier: AGPL-3.0-only
import { authFetch, getAuthSessionEpoch } from "@/features/auth";
// eslint-disable-next-line no-restricted-imports -- API leaf avoids pulling the Images page into chat.
import {
  chatImageSwapSelection,
  getDiffusionStatus,
  type DiffusionGenerateResponse,
} from "@/features/images/api";
import { createNavigationNonce } from "@/lib/navigation-nonce";
import { fileToBase64 } from "@/lib/audio-utils";
import { streamedToolCallArguments } from "../tool-call-arguments";
import { extractDeltaText } from "../utils/parse-assistant-content";
import type {
  OpenAIChatChunk,
  OpenAIChatCompletionsRequest,
  OpenAIChatMessage,
  OpenAIToolCallPart,
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
  const proposal = {
    ...body,
    stream: true,
    enable_tools: false,
    enabled_tools: undefined,
    // The stream only proposes a call. The client confirms before rendering.
    confirm_tool_calls: false,
    mcp_enabled: false,
    tools: [tool],
    tool_choice: "auto",
    parallel_tool_calls: false,
  };
  let call: OpenAIToolCallPart | undefined;
  let content = "";
  let reasoning = "";
  let finishReason: string | null | undefined;
  for await (const chunk of streamChatCompletions(
    proposal,
    signal,
    contextLength,
    false,
    { waitForClose: true },
  )) {
    const choice = chunk.choices?.[0];
    const delta = choice?.delta;
    content += extractDeltaText(delta?.content).text;
    const thought = (delta as { reasoning_content?: unknown } | undefined)
      ?.reasoning_content;
    if (typeof thought === "string") reasoning += thought;
    finishReason = choice?.finish_reason ?? finishReason;
    for (const fragment of delta?.tool_calls ?? []) {
      if (
        ("index" in fragment &&
          typeof fragment.index === "number" &&
          fragment.index !== 0) ||
        (call?.id && fragment.id && call.id !== fragment.id)
      ) {
        throw new Error("Expected a single image tool call.");
      }
      call = {
        id: fragment.id ?? call?.id,
        type: "function",
        function: {
          name: fragment.function?.name ?? call?.function?.name,
          arguments:
            (call?.function?.arguments ?? "") +
            streamedToolCallArguments(fragment.function?.arguments),
        },
      };
    }
    if (
      choice &&
      (delta?.tool_calls?.length || choice.finish_reason === "tool_calls")
    ) {
      // The image card is emitted after the complete call, as one start/end pair.
      yield {
        ...chunk,
        choices: [
          {
            ...choice,
            delta: { ...delta, tool_calls: undefined },
            finish_reason: null,
          },
        ],
      };
    } else {
      yield chunk;
    }
  }
  if (!call) return;
  if (
    finishReason === "length" ||
    !call.id ||
    call.function?.name !== "generate_image"
  ) {
    throw new Error("Expected a complete image tool call.");
  }
  const assistant: OpenAIChatMessage & { reasoning_content?: string } = {
    role: "assistant",
    content: content || null,
    ...(reasoning ? { reasoning_content: reasoning } : {}),
    tool_calls: [call],
  };
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
        assistant,
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
