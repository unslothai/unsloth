// src/octto/types.ts
// Common types for all interactive tools

export interface BaseConfig {
  /** Window title */
  title?: string;
  /** Timeout in seconds (0 = no timeout) */
  timeout?: number;
  /** Theme preference */
  theme?: "light" | "dark" | "auto";
}

export interface Option {
  /** Unique identifier */
  id: string;
  /** Display label */
  label: string;
  /** Optional description */
  description?: string;
}

export interface OptionWithPros extends Option {
  /** Pros/advantages */
  pros?: string[];
  /** Cons/disadvantages */
  cons?: string[];
}

export interface RatedOption extends Option {
  /** User's rating (filled after response) */
  rating?: number;
}

export interface RankedOption extends Option {
  /** User's rank position (filled after response) */
  rank?: number;
}

// Tool-specific configs

export interface PickOneConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Available options */
  options: Option[];
  /** Recommended option id (highlighted) */
  recommended?: string;
  /** Allow custom "other" input */
  allowOther?: boolean;
}

export interface PickManyConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Available options */
  options: Option[];
  /** Recommended option ids (highlighted) */
  recommended?: string[];
  /** Minimum selections required */
  min?: number;
  /** Maximum selections allowed */
  max?: number;
  /** Allow custom "other" input */
  allowOther?: boolean;
}

export interface ConfirmConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Context/details to show */
  context?: string;
  /** Custom label for yes button */
  yesLabel?: string;
  /** Custom label for no button */
  noLabel?: string;
  /** Show cancel option */
  allowCancel?: boolean;
}

export interface RankConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Items to rank */
  options: Option[];
  /** Context/instructions */
  context?: string;
}

export interface RateConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Items to rate */
  options: Option[];
  /** Minimum rating value */
  min?: number;
  /** Maximum rating value */
  max?: number;
  /** Rating step (default 1) */
  step?: number;
  /** Labels for min/max */
  labels?: { min?: string; max?: string };
}

export interface AskTextConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Placeholder text */
  placeholder?: string;
  /** Context/instructions */
  context?: string;
  /** Multi-line input */
  multiline?: boolean;
  /** Minimum length */
  minLength?: number;
  /** Maximum length */
  maxLength?: number;
}

export interface AskImageConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Context/instructions */
  context?: string;
  /** Allow multiple images */
  multiple?: boolean;
  /** Maximum number of images */
  maxImages?: number;
  /** Allowed mime types */
  accept?: string[];
}

export interface AskFileConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Context/instructions */
  context?: string;
  /** Allow multiple files */
  multiple?: boolean;
  /** Maximum number of files */
  maxFiles?: number;
  /** Allowed file extensions or mime types */
  accept?: string[];
  /** Maximum file size in bytes */
  maxSize?: number;
}

export interface AskCodeConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Context/instructions */
  context?: string;
  /** Programming language for syntax highlighting */
  language?: string;
  /** Placeholder code */
  placeholder?: string;
}

export interface ShowDiffConfig extends BaseConfig {
  /** Title/description of the change */
  question: string;
  /** Original content */
  before: string;
  /** Modified content */
  after: string;
  /** File path (for context) */
  filePath?: string;
  /** Language for syntax highlighting */
  language?: string;
}

export interface PlanSection {
  /** Section identifier */
  id: string;
  /** Section title */
  title: string;
  /** Section content (markdown) */
  content: string;
}

export interface ShowPlanConfig extends BaseConfig {
  /** Plan title */
  question: string;
  /** Plan sections */
  sections?: PlanSection[];
  /** Full markdown (alternative to sections) */
  markdown?: string;
}

export interface ShowOptionsConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Options with pros/cons */
  options: OptionWithPros[];
  /** Recommended option id */
  recommended?: string;
  /** Allow text feedback with selection */
  allowFeedback?: boolean;
}

export interface ReviewSectionConfig extends BaseConfig {
  /** Section title */
  question: string;
  /** Section content (markdown) */
  content: string;
  /** Context about what to review */
  context?: string;
}

export interface ThumbsConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Context to show */
  context?: string;
}

export interface EmojiReactConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Context to show */
  context?: string;
  /** Available emoji options (default: common set) */
  emojis?: string[];
}

export interface SliderConfig extends BaseConfig {
  /** Question/prompt to display */
  question: string;
  /** Context/instructions */
  context?: string;
  /** Minimum value */
  min: number;
  /** Maximum value */
  max: number;
  /** Step size */
  step?: number;
  /** Default value */
  defaultValue?: number;
  /** Labels for values */
  labels?: { min?: string; max?: string; mid?: string };
}

// Response types

export interface BaseResponse {
  /** Whether the interaction completed (false if cancelled/timeout) */
  completed: boolean;
  /** Cancellation reason if not completed */
  cancelReason?: "timeout" | "cancelled" | "closed";
}

export interface PickOneResponse extends BaseResponse {
  /** Selected option id */
  selected?: string;
  /** Custom "other" value if provided */
  other?: string;
}

export interface PickManyResponse extends BaseResponse {
  /** Selected option ids */
  selected: string[];
  /** Custom "other" values if provided */
  other?: string[];
}

export interface ConfirmResponse extends BaseResponse {
  /** User's choice */
  choice?: "yes" | "no" | "cancel";
}

export interface RankResponse extends BaseResponse {
  /** Option ids in ranked order (first = highest) */
  ranking: string[];
}

export interface RateResponse extends BaseResponse {
  /** Ratings by option id */
  ratings: Record<string, number>;
}

export interface AskTextResponse extends BaseResponse {
  /** User's text input */
  text?: string;
}

export interface AskImageResponse extends BaseResponse {
  /** Image data */
  images: Array<{
    /** Original filename */
    filename: string;
    /** Mime type */
    mimeType: string;
    /** Base64 encoded data */
    data: string;
  }>;
  /** File paths (if provided instead of upload) */
  paths?: string[];
}

export interface AskFileResponse extends BaseResponse {
  /** File data */
  files: Array<{
    /** Original filename */
    filename: string;
    /** Mime type */
    mimeType: string;
    /** Base64 encoded data */
    data: string;
  }>;
  /** File paths (if provided instead of upload) */
  paths?: string[];
}

export interface AskCodeResponse extends BaseResponse {
  /** User's code input */
  code?: string;
  /** Detected/selected language */
  language?: string;
}

export interface ShowDiffResponse extends BaseResponse {
  /** User's decision */
  decision?: "approve" | "reject" | "edit";
  /** User's edited version (if decision is "edit") */
  edited?: string;
  /** Optional feedback */
  feedback?: string;
}

export interface Annotation {
  /** Annotation id */
  id: string;
  /** Section id or line range */
  target: string;
  /** Annotation type */
  type: "comment" | "suggest" | "delete" | "approve";
  /** Annotation content */
  content?: string;
}

export interface ShowPlanResponse extends BaseResponse {
  /** User's decision */
  decision?: "approve" | "reject" | "revise";
  /** User annotations */
  annotations: Annotation[];
  /** Overall feedback */
  feedback?: string;
}

export interface ShowOptionsResponse extends BaseResponse {
  /** Selected option id */
  selected?: string;
  /** Optional feedback text */
  feedback?: string;
}

export interface ReviewSectionResponse extends BaseResponse {
  /** User's decision */
  decision?: "approve" | "revise";
  /** Inline feedback/suggestions */
  feedback?: string;
}

export interface ThumbsResponse extends BaseResponse {
  /** User's choice */
  choice?: "up" | "down";
}

export interface EmojiReactResponse extends BaseResponse {
  /** Selected emoji */
  emoji?: string;
}

export interface SliderResponse extends BaseResponse {
  /** Selected value */
  value?: number;
}
