// src/tools/octto/questions.ts
import { tool } from "@opencode-ai/plugin/tool";

import type { SessionStore } from "../../octto/session";
import type { ConfirmConfig, PickManyConfig, PickOneConfig, RankConfig, RateConfig } from "../../octto/types";
import { createQuestionToolFactory } from "./factory";
import type { OcttoTools } from "./types";

const optionsSchema = tool.schema
  .array(
    tool.schema.object({
      id: tool.schema.string().describe("Unique option identifier"),
      label: tool.schema.string().describe("Display label"),
      description: tool.schema.string().optional().describe("Optional description"),
    }),
  )
  .describe("Available options");

function requireOptions(args: { options?: unknown[] }): string | null {
  if (!args.options || args.options.length === 0) return "options array must not be empty";
  return null;
}

export function createQuestionTools(sessions: SessionStore): OcttoTools {
  const createTool = createQuestionToolFactory(sessions);

  const pick_one = createTool<PickOneConfig & { session_id: string }>({
    type: "pick_one",
    description: `Ask user to select ONE option from a list.
Response format: { selected: string } where selected is the chosen option id.`,
    args: {
      question: tool.schema.string().describe("Question to display"),
      options: optionsSchema,
      recommended: tool.schema.string().optional().describe("Recommended option id (highlighted)"),
      allowOther: tool.schema.boolean().optional().describe("Allow custom 'other' input"),
    },
    validate: requireOptions,
    toConfig: (args) => ({
      question: args.question,
      options: args.options,
      recommended: args.recommended,
      allowOther: args.allowOther,
    }),
  });

  const pick_many = createTool<PickManyConfig & { session_id: string }>({
    type: "pick_many",
    description: `Ask user to select MULTIPLE options from a list.
Response format: { selected: string[] } where selected is array of chosen option ids.`,
    args: {
      question: tool.schema.string().describe("Question to display"),
      options: optionsSchema,
      recommended: tool.schema.array(tool.schema.string()).optional().describe("Recommended option ids"),
      min: tool.schema.number().optional().describe("Minimum selections required"),
      max: tool.schema.number().optional().describe("Maximum selections allowed"),
      allowOther: tool.schema.boolean().optional().describe("Allow custom 'other' input"),
    },
    validate: (args) => {
      if (!args.options || args.options.length === 0) return "options array must not be empty";
      if (args.min !== undefined && args.max !== undefined && args.min > args.max) {
        return `min (${args.min}) cannot be greater than max (${args.max})`;
      }
      return null;
    },
    toConfig: (args) => ({
      question: args.question,
      options: args.options,
      recommended: args.recommended,
      min: args.min,
      max: args.max,
      allowOther: args.allowOther,
    }),
  });

  const confirm = createTool<ConfirmConfig & { session_id: string }>({
    type: "confirm",
    description: `Ask user for Yes/No confirmation.
Response format: { choice: "yes" | "no" | "cancel" }`,
    args: {
      question: tool.schema.string().describe("Question to display"),
      context: tool.schema.string().optional().describe("Additional context/details"),
      yesLabel: tool.schema.string().optional().describe("Custom label for yes button"),
      noLabel: tool.schema.string().optional().describe("Custom label for no button"),
      allowCancel: tool.schema.boolean().optional().describe("Show cancel option"),
    },
    toConfig: (args) => ({
      question: args.question,
      context: args.context,
      yesLabel: args.yesLabel,
      noLabel: args.noLabel,
      allowCancel: args.allowCancel,
    }),
  });

  const rank = createTool<RankConfig & { session_id: string }>({
    type: "rank",
    description: `Ask user to rank/order items by dragging.
Response format: { ranked: string[] } where ranked is array of option ids in user's order (first = highest).`,
    args: {
      question: tool.schema.string().describe("Question to display"),
      options: optionsSchema.describe("Items to rank"),
      context: tool.schema.string().optional().describe("Instructions/context"),
    },
    validate: requireOptions,
    toConfig: (args) => ({
      question: args.question,
      options: args.options,
      context: args.context,
    }),
  });

  const rate = createTool<RateConfig & { session_id: string }>({
    type: "rate",
    description: `Ask user to rate items on a numeric scale.
Response format: { ratings: Record<string, number> } where key is option id, value is rating.`,
    args: {
      question: tool.schema.string().describe("Question to display"),
      options: optionsSchema.describe("Items to rate"),
      min: tool.schema.number().optional().describe("Minimum rating value (default: 1)"),
      max: tool.schema.number().optional().describe("Maximum rating value (default: 5)"),
      step: tool.schema.number().optional().describe("Rating step (default: 1)"),
      labels: tool.schema
        .object({
          min: tool.schema.string().optional().describe("Label for minimum value"),
          max: tool.schema.string().optional().describe("Label for maximum value"),
        })
        .optional()
        .describe("Optional labels for min/max"),
    },
    validate: (args) => {
      if (!args.options || args.options.length === 0) return "options array must not be empty";
      const min = args.min ?? 1;
      const max = args.max ?? 5;
      if (min >= max) return `min (${min}) must be less than max (${max})`;
      return null;
    },
    toConfig: (args) => ({
      question: args.question,
      options: args.options,
      min: args.min ?? 1,
      max: args.max ?? 5,
      step: args.step,
      labels: args.labels,
    }),
  });

  // Import remaining tools from other files
  const inputTools = createInputTools(sessions);
  const presentationTools = createPresentationTools(sessions);
  const quickTools = createQuickTools(sessions);

  return {
    pick_one,
    pick_many,
    confirm,
    rank,
    rate,
    ...inputTools,
    ...presentationTools,
    ...quickTools,
  };
}

// Input tools using factory
function createInputTools(sessions: SessionStore): OcttoTools {
  const createTool = createQuestionToolFactory(sessions);

  interface TextConfig {
    session_id: string;
    question: string;
    placeholder?: string;
    context?: string;
    multiline?: boolean;
    minLength?: number;
    maxLength?: number;
  }

  const ask_text = createTool<TextConfig>({
    type: "ask_text",
    description: `Ask user for text input (single or multi-line).
Response format: { text: string }`,
    args: {
      question: tool.schema.string().describe("Question to display"),
      placeholder: tool.schema.string().optional().describe("Placeholder text"),
      context: tool.schema.string().optional().describe("Instructions/context"),
      multiline: tool.schema.boolean().optional().describe("Multi-line input (default: false)"),
      minLength: tool.schema.number().optional().describe("Minimum text length"),
      maxLength: tool.schema.number().optional().describe("Maximum text length"),
    },
    toConfig: (args) => ({
      question: args.question,
      placeholder: args.placeholder,
      context: args.context,
      multiline: args.multiline,
      minLength: args.minLength,
      maxLength: args.maxLength,
    }),
  });

  interface ImageConfig {
    session_id: string;
    question: string;
    context?: string;
    multiple?: boolean;
    maxImages?: number;
    accept?: string[];
  }

  const ask_image = createTool<ImageConfig>({
    type: "ask_image",
    description: "Ask user to upload/paste image(s).",
    args: {
      question: tool.schema.string().describe("Question to display"),
      context: tool.schema.string().optional().describe("Instructions/context"),
      multiple: tool.schema.boolean().optional().describe("Allow multiple images"),
      maxImages: tool.schema.number().optional().describe("Maximum number of images"),
      accept: tool.schema.array(tool.schema.string()).optional().describe("Allowed image types"),
    },
    toConfig: (args) => ({
      question: args.question,
      context: args.context,
      multiple: args.multiple,
      maxImages: args.maxImages,
      accept: args.accept,
    }),
  });

  interface FileConfig {
    session_id: string;
    question: string;
    context?: string;
    multiple?: boolean;
    maxFiles?: number;
    accept?: string[];
    maxSize?: number;
  }

  const ask_file = createTool<FileConfig>({
    type: "ask_file",
    description: "Ask user to upload file(s).",
    args: {
      question: tool.schema.string().describe("Question to display"),
      context: tool.schema.string().optional().describe("Instructions/context"),
      multiple: tool.schema.boolean().optional().describe("Allow multiple files"),
      maxFiles: tool.schema.number().optional().describe("Maximum number of files"),
      accept: tool.schema.array(tool.schema.string()).optional().describe("Allowed file types"),
      maxSize: tool.schema.number().optional().describe("Maximum file size in bytes"),
    },
    toConfig: (args) => ({
      question: args.question,
      context: args.context,
      multiple: args.multiple,
      maxFiles: args.maxFiles,
      accept: args.accept,
      maxSize: args.maxSize,
    }),
  });

  interface CodeConfig {
    session_id: string;
    question: string;
    context?: string;
    language?: string;
    placeholder?: string;
  }

  const ask_code = createTool<CodeConfig>({
    type: "ask_code",
    description: "Ask user for code input with syntax highlighting.",
    args: {
      question: tool.schema.string().describe("Question to display"),
      context: tool.schema.string().optional().describe("Instructions/context"),
      language: tool.schema.string().optional().describe("Programming language for highlighting"),
      placeholder: tool.schema.string().optional().describe("Placeholder code"),
    },
    toConfig: (args) => ({
      question: args.question,
      context: args.context,
      language: args.language,
      placeholder: args.placeholder,
    }),
  });

  return { ask_text, ask_image, ask_file, ask_code };
}

// Presentation tools using factory
function createPresentationTools(sessions: SessionStore): OcttoTools {
  const createTool = createQuestionToolFactory(sessions);

  interface DiffConfig {
    session_id: string;
    question: string;
    before: string;
    after: string;
    filePath?: string;
    language?: string;
  }

  const show_diff = createTool<DiffConfig>({
    type: "show_diff",
    description: "Show a diff and ask user to approve/reject/edit.",
    args: {
      question: tool.schema.string().describe("Title/description of the change"),
      before: tool.schema.string().describe("Original content"),
      after: tool.schema.string().describe("Modified content"),
      filePath: tool.schema.string().optional().describe("File path for context"),
      language: tool.schema.string().optional().describe("Language for syntax highlighting"),
    },
    toConfig: (args) => ({
      question: args.question,
      before: args.before,
      after: args.after,
      filePath: args.filePath,
      language: args.language,
    }),
  });

  const sectionSchema = tool.schema.array(
    tool.schema.object({
      id: tool.schema.string().describe("Section identifier"),
      title: tool.schema.string().describe("Section title"),
      content: tool.schema.string().describe("Section content (markdown)"),
    }),
  );

  interface PlanConfig {
    session_id: string;
    question: string;
    sections?: Array<{ id: string; title: string; content: string }>;
    markdown?: string;
  }

  const show_plan = createTool<PlanConfig>({
    type: "show_plan",
    description: `Show a plan/document for user review with annotations.
Response format: { approved: boolean, annotations?: Record<sectionId, string> }`,
    args: {
      question: tool.schema.string().describe("Plan title"),
      sections: sectionSchema.optional().describe("Plan sections"),
      markdown: tool.schema.string().optional().describe("Full markdown (alternative to sections)"),
    },
    toConfig: (args) => ({
      question: args.question,
      sections: args.sections,
      markdown: args.markdown,
    }),
  });

  const prosConsOptionSchema = tool.schema.array(
    tool.schema.object({
      id: tool.schema.string().describe("Unique option identifier"),
      label: tool.schema.string().describe("Display label"),
      description: tool.schema.string().optional().describe("Optional description"),
      pros: tool.schema.array(tool.schema.string()).optional().describe("Advantages"),
      cons: tool.schema.array(tool.schema.string()).optional().describe("Disadvantages"),
    }),
  );

  interface ShowOptionsConfig {
    session_id: string;
    question: string;
    options: Array<{ id: string; label: string; description?: string; pros?: string[]; cons?: string[] }>;
    recommended?: string;
    allowFeedback?: boolean;
  }

  const show_options = createTool<ShowOptionsConfig>({
    type: "show_options",
    description: `Show options with pros/cons for user to select.
Response format: { selected: string, feedback?: string } where selected is the chosen option id.`,
    args: {
      question: tool.schema.string().describe("Question to display"),
      options: prosConsOptionSchema.describe("Options with pros/cons"),
      recommended: tool.schema.string().optional().describe("Recommended option id"),
      allowFeedback: tool.schema.boolean().optional().describe("Allow text feedback with selection"),
    },
    validate: (args) => {
      if (!args.options || args.options.length === 0) return "options array must not be empty";
      return null;
    },
    toConfig: (args) => ({
      question: args.question,
      options: args.options,
      recommended: args.recommended,
      allowFeedback: args.allowFeedback,
    }),
  });

  interface ReviewConfig {
    session_id: string;
    question: string;
    content: string;
    context?: string;
  }

  const review_section = createTool<ReviewConfig>({
    type: "review_section",
    description: "Show content section for user review with inline feedback.",
    args: {
      question: tool.schema.string().describe("Section title"),
      content: tool.schema.string().describe("Section content (markdown)"),
      context: tool.schema.string().optional().describe("Context about what to review"),
    },
    toConfig: (args) => ({
      question: args.question,
      content: args.content,
      context: args.context,
    }),
  });

  return { show_diff, show_plan, show_options, review_section };
}

// Quick tools using factory
function createQuickTools(sessions: SessionStore): OcttoTools {
  const createTool = createQuestionToolFactory(sessions);

  interface ThumbsConfig {
    session_id: string;
    question: string;
    context?: string;
  }

  const thumbs = createTool<ThumbsConfig>({
    type: "thumbs",
    description: `Ask user for quick thumbs up/down feedback.
Response format: { choice: "up" | "down" }`,
    args: {
      question: tool.schema.string().describe("Question to display"),
      context: tool.schema.string().optional().describe("Context to show"),
    },
    toConfig: (args) => ({
      question: args.question,
      context: args.context,
    }),
  });

  interface EmojiConfig {
    session_id: string;
    question: string;
    context?: string;
    emojis?: string[];
  }

  const emoji_react = createTool<EmojiConfig>({
    type: "emoji_react",
    description: "Ask user to react with an emoji.",
    args: {
      question: tool.schema.string().describe("Question to display"),
      context: tool.schema.string().optional().describe("Context to show"),
      emojis: tool.schema.array(tool.schema.string()).optional().describe("Available emoji options"),
    },
    toConfig: (args) => ({
      question: args.question,
      context: args.context,
      emojis: args.emojis,
    }),
  });

  interface SliderConfig {
    session_id: string;
    question: string;
    min: number;
    max: number;
    step?: number;
    defaultValue?: number;
    context?: string;
    labels?: { min?: string; max?: string; mid?: string };
  }

  const slider = createTool<SliderConfig>({
    type: "slider",
    description: `Ask user to select a value on a numeric slider.
Response format: { value: number }`,
    args: {
      question: tool.schema.string().describe("Question to display"),
      min: tool.schema.number().describe("Minimum value"),
      max: tool.schema.number().describe("Maximum value"),
      step: tool.schema.number().optional().describe("Step size (default: 1)"),
      defaultValue: tool.schema.number().optional().describe("Default value"),
      context: tool.schema.string().optional().describe("Instructions/context"),
      labels: tool.schema
        .object({
          min: tool.schema.string().optional().describe("Label for minimum value"),
          max: tool.schema.string().optional().describe("Label for maximum value"),
          mid: tool.schema.string().optional().describe("Label for middle value"),
        })
        .optional()
        .describe("Optional labels for the slider"),
    },
    validate: (args) => {
      if (args.min >= args.max) return `min (${args.min}) must be less than max (${args.max})`;
      return null;
    },
    toConfig: (args) => ({
      question: args.question,
      min: args.min,
      max: args.max,
      step: args.step,
      defaultValue: args.defaultValue,
      context: args.context,
      labels: args.labels,
    }),
  });

  return { thumbs, emoji_react, slider };
}
