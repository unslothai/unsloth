export type CheckFormatResponse = {
  requires_manual_mapping: boolean;
  detected_format: string;
  columns: string[];
  suggested_mapping?: Record<string, string> | null;
  detected_image_column?: string | null;
  detected_audio_column?: string | null;
  detected_text_column?: string | null;
  detected_speaker_column?: string | null;
  preview_samples?: Record<string, unknown>[] | null;
  total_rows?: number | null;
  is_multimodal?: boolean;
  is_audio?: boolean;
  multimodal_columns?: string[] | null;
};

