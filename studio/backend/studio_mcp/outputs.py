# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Typed tool outputs. Several Unsloth Studio routes still return host paths to API-key callers, so a tool never passes route JSON through: it copies named fields into a ToolOutput, which refuses anything undeclared. Free text that came from a route (a status message, an error) is declared ``RouteText`` and scrubbed of paths as a backstop; model-written text and Unsloth Studio URLs are plain ``str`` and left alone."""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypeVar, Union, get_args, get_origin

from pydantic import AfterValidator, BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt

from hub.utils.host_paths import redact_paths_in_text

RouteText = Annotated[str, AfterValidator(redact_paths_in_text)]


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    @classmethod
    def from_route(cls: type[_Out], payload: Any, **overrides: Any) -> _Out:
        """Copy each declared field from route JSON, coerced by its type: text must be non-empty, numbers must not be bools, a bool must be ``true``, a list of text keeps its strings. A field the route left out or sent malformed keeps its default; nested models, dicts and anything needing more care come in through ``overrides``."""
        row = payload if isinstance(payload, dict) else {}
        fields = {}
        for name, info in cls.model_fields.items():
            value = None if name in overrides else _route_value(info.annotation, row.get(name))
            if value is not None:
                fields[name] = value
        return cls(**fields, **overrides)


_Out = TypeVar("_Out", bound = ToolOutput)


def _unwrap(annotation: Any) -> Any:
    # The type under Annotated and Optional; None for a union of several types.
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) is Union:
        kinds = [kind for kind in get_args(annotation) if kind is not type(None)]
        return _unwrap(kinds[0]) if len(kinds) == 1 else None
    return annotation


def _route_value(annotation: Any, value: Any) -> Any:
    from studio_mcp.tools import integer, number, opt_text

    kind = _unwrap(annotation)
    if get_origin(kind) is list:
        if _unwrap(get_args(kind)[0]) is not str or not isinstance(value, list):
            return None
        return [item for item in value if isinstance(item, str)]
    if kind is bool:
        return value is True
    rule = {str: opt_text, int: integer, float: number}.get(kind)
    return rule(value) if rule else None


class AudioOption(ToolOutput):
    name: RouteText
    # bool, int, float, string or enum.
    type: Optional[RouteText] = None
    description: Optional[RouteText] = None
    default: Optional[Union[StrictBool, StrictInt, StrictFloat, RouteText]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    # An enum's accepted values, such as the model's built-in voices.
    values: list[RouteText] = []


def audio_options(rows: Any) -> Optional[list[AudioOption]]:
    """The options an audio model reports, or None when it reports none."""
    options = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and isinstance(row.get("name"), str) and row["name"]:
            default = row.get("default")
            scalar = isinstance(default, (bool, int, float, str))
            options.append(AudioOption.from_route(row, default = default if scalar else None))
    return options or None


class Placement(ToolOutput):
    """Where a model ended up, as Unsloth Studio reports it after the load."""

    # "cpu" or "gpu"; for a speech model that asked for the GPU and fell back, "cpu".
    device: Optional[RouteText] = None
    memory_mode: Optional[RouteText] = None
    # How much of an image or video model is moved off the GPU: none, model, group, streaming or sequential.
    offload_policy: Optional[RouteText] = None
    transformer_quant: Optional[RouteText] = None
    text_encoder_quant: Optional[RouteText] = None


class MusicModeBounds(ToolOutput):
    # A music mode's duration_s range and default, in seconds.
    id: RouteText
    min_s: Optional[float] = None
    max_s: Optional[float] = None
    default_s: Optional[float] = None


class ChatModel(ToolOutput):
    id: RouteText
    display_name: Optional[RouteText] = None
    is_gguf: Optional[bool] = None
    # The active audio model's options for generate_audio, built-in voices included.
    audio_options: Optional[list[AudioOption]] = None
    # The active music model's duration_s bounds per mode; a longer duration is clamped.
    audio_music: Optional[list[MusicModeBounds]] = None

    @staticmethod
    def music_modes(raw: Any) -> Optional[list[MusicModeBounds]]:
        """The bounds of each mode in a status ``audio_music``, or None when it lists none."""
        from studio_mcp.tools import number

        modes = raw.get("modes") if isinstance(raw, dict) else None
        bounds = []
        for mode in modes if isinstance(modes, list) else []:
            duration = mode.get("duration") if isinstance(mode, dict) else None
            if not isinstance(duration, dict) or not isinstance(mode.get("id"), str):
                continue
            bounds.append(
                MusicModeBounds(
                    id = mode["id"],
                    min_s = number(duration.get("min")),
                    max_s = number(duration.get("max")),
                    default_s = number(duration.get("default")),
                )
            )
        return bounds or None


class ChatSlot(ToolOutput):
    loaded: list[ChatModel] = []
    loading: list[RouteText] = []
    # Keep multiple models loaded: off, fit or always. None from a server that predates it.
    multi_model_mode: Optional[RouteText] = None


class ImageSlot(ToolOutput):
    loaded: bool = False
    model: Optional[RouteText] = None
    family: Optional[RouteText] = None
    # What the loaded model runs, e.g. txt2img, img2img, inpaint, edit, reference.
    workflows: list[RouteText] = []
    # The steps and guidance an omitted value resolves to.
    generation_defaults: Optional[dict[str, Union[StrictInt, StrictFloat]]] = None
    # Width and height must be multiples of this.
    dimension_multiple: Optional[int] = None
    # init_image plus reference_images together hold at most this many images.
    max_condition_images: Optional[int] = None
    # How the loaded model reads prompts, in one or two sentences.
    prompt_hint: Optional[RouteText] = None
    generating: bool = False
    placement: Optional[Placement] = None


class VideoSlot(ToolOutput):
    loaded: bool = False
    model: Optional[RouteText] = None
    # How the loaded model reads prompts, in one or two sentences.
    prompt_hint: Optional[RouteText] = None
    supports_last_frame: bool = False
    # Whether the loaded model takes reference_images and reference_audios.
    references: bool = False
    # "WIDTHxHEIGHT" sizes the loaded model renders.
    size_presets: list[RouteText] = []
    generating: bool = False
    placement: Optional[Placement] = None


class SttDownload(ToolOutput):
    model: RouteText
    fraction: Optional[float] = None


class SttSlot(ToolOutput):
    engine: Optional[RouteText] = None
    model: Optional[RouteText] = None
    loading: bool = False
    downloading: list[SttDownload] = []


class EmbedderSlot(ToolOutput):
    # False when this key may not read the owner's embedder setting.
    available: bool = False
    loaded: bool = False
    # A Hub repo id, or "custom" for a model the owner picked from disk.
    model: Optional[RouteText] = None


class TrainingSlot(ToolOutput):
    job_id: Optional[RouteText] = None
    phase: RouteText = "idle"
    is_training_running: bool = False
    message: RouteText = ""
    error: Optional[RouteText] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    progress_percent: Optional[float] = None
    eta_seconds: Optional[float] = None
    elapsed_seconds: Optional[float] = None


class TrainingJob(TrainingSlot):
    # Once completed: the run folder and its checkpoints by the names export_model and load_model take.
    run_folder: Optional[RouteText] = None
    checkpoints: list[RouteText] = []
    # The call that returns the run's full loss curve.
    loss_history: Optional[str] = None


class ExportSlot(ToolOutput):
    active: bool = False
    op_kind: Optional[RouteText] = None
    last_op_status: Optional[RouteText] = None
    # The output folder's name, never its path.
    last_output: Optional[RouteText] = None


class GpuDevice(ToolOutput):
    index: Optional[int] = None
    gpu_utilization_pct: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    temperature_c: Optional[float] = None


class HardwareSlot(ToolOutput):
    available: bool = False
    backend: Optional[RouteText] = None
    devices: list[GpuDevice] = []


class StudioStatus(ToolOutput):
    chat: ChatSlot = ChatSlot()
    image: ImageSlot = ImageSlot()
    video: VideoSlot = VideoSlot()
    stt: SttSlot = SttSlot()
    embedder: EmbedderSlot = EmbedderSlot()
    training: TrainingSlot = TrainingSlot()
    diffusion_training: TrainingSlot = TrainingSlot()
    export: ExportSlot = ExportSlot()
    hardware: HardwareSlot = HardwareSlot()
    # Slots whose route failed, with the reason; the other slots are still reported.
    unavailable: dict[str, RouteText] = {}


class ModelEntry(ToolOutput):
    id: RouteText
    # From the route's task; "llm" when there is none.
    kind: RouteText
    loaded: bool = False
    display_name: Optional[RouteText] = None
    quant: Optional[RouteText] = None
    context_length: Optional[int] = None
    # Catalog rows only: decimal bytes, a reduced-memory artifact, whether it fits one GPU's free memory now.
    size_bytes: Optional[int] = None
    low_vram: Optional[bool] = None
    fits_now: Optional[bool] = None
    # "catalog" for list_models(source="catalog") rows.
    source: Optional[RouteText] = None
    audio_workflows: Optional[list[RouteText]] = None


class ModelList(ToolOutput):
    models: list[ModelEntry] = []
    # Studio's training defaults for the model named in the call.
    training_defaults: Optional[dict[str, Union[StrictBool, StrictInt, StrictFloat, RouteText]]] = (
        None
    )


class LoadResult(ToolOutput):
    kind: RouteText
    model: RouteText
    loaded: bool = True
    # "loading" when wait_s ran out first: the load goes on, and get_job(kind="load", id=job_id) follows it.
    status: RouteText = "loaded"
    fraction: Optional[float] = None
    job_id: Optional[RouteText] = None
    poll_after_s: Optional[float] = None
    display_name: Optional[RouteText] = None
    placement: Optional[Placement] = None
    # True when the chat model that was serving before the load still serves; None when none was.
    alongside_applied: Optional[bool] = None
    evicted: list[RouteText] = []
    evicted_slots: list[EvictedSlot] = []


class EvictedSlot(ToolOutput):
    # llm, audio, image or video.
    kind: RouteText
    model: RouteText


class UnloadResult(ToolOutput):
    kind: RouteText
    model: Optional[RouteText] = None
    unloaded: bool


class SlotModel(ToolOutput):
    # llm (chat, TTS and audio models share it), image, video or stt.
    kind: RouteText
    model: RouteText


class UnloadAllResult(ToolOutput):
    kind: RouteText = "all"
    unloaded: list[SlotModel] = []


class PlanVariant(ToolOutput):
    variant: RouteText
    size_bytes: Optional[int] = None
    downloaded: bool = False
    # Some files are on disk but not all: a download still running, or one that stopped.
    partial: bool = False


class PlanLoadResult(ToolOutput):
    model: RouteText
    kind: RouteText
    variants: list[PlanVariant] = []
    # Bytes still to download for this load; None when Unsloth Studio cannot tell before downloading.
    download_bytes: Optional[int] = None
    # The GPU memory the load is estimated to take; only known for a downloaded GGUF chat model.
    est_gpu_bytes: Optional[int] = None
    residents: list[SlotModel] = []
    # off, fit or always (Keep multiple models loaded); None when this Unsloth Studio does not report it.
    multi_model_mode: Optional[RouteText] = None
    would_evict: list[SlotModel] = []
    # False when would_evict is an upper bound: the load may keep some of them.
    would_evict_exact: bool = True
    # True when Keep multiple models loaded is Always and the load may be refused instead of unloading a model.
    would_refuse: bool = False


class Usage(ToolOutput):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatResult(ToolOutput):
    # Model-written, so never rewritten.
    text: str
    model: Optional[RouteText] = None
    finish_reason: Optional[RouteText] = None
    usage: Optional[Usage] = None
    note: Optional[RouteText] = None
    # Pass to cancel(kind="chat", id=...) from another call to stop a long reply.
    cancel_id: Optional[str] = None


class EmbedMatch(ToolOutput):
    index: int
    score: float
    text: str


class EmbedResult(ToolOutput):
    model: Optional[RouteText] = None
    dimensions: int
    embeddings: Optional[list[list[float]]] = None
    # With a query: the best matches, highest score first, instead of vectors.
    matches: Optional[list[EmbedMatch]] = None


class DecisionAnswer(ToolOutput):
    type: RouteText
    # noul: probability of yes. choice: the picked option. score: the level, 1 being lowest.
    noul: Optional[float] = None
    choice: Optional[RouteText] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict[str, float]] = None
    legend: Optional[dict[str, RouteText]] = None


class SystemOneResult(ToolOutput):
    model: Optional[RouteText] = None
    answers: dict[str, DecisionAnswer]
    request_id: Optional[RouteText] = None


class ImageItem(ToolOutput):
    id: RouteText
    url: str
    # Works without the API key for about an hour; None when Unsloth Studio would not mint it.
    download_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    # Set on images from a seeds or prompts batch: the image's own seed and its place in the batch.
    batch_seed: Optional[int] = None
    batch_index: Optional[int] = None


class ImageResult(ToolOutput):
    images: list[ImageItem]


class ImageRow(ToolOutput):
    id: RouteText
    url: str
    download_url: Optional[str] = None
    prompt: Optional[RouteText] = None
    model: Optional[RouteText] = None
    workflow: Optional[RouteText] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    created_at: Optional[float] = None
    pinned: bool = False


class AudioClip(ToolOutput):
    id: RouteText
    role: RouteText = "output"
    workflow: Optional[RouteText] = None
    voice_id: Optional[RouteText] = None
    source_clip_id: Optional[RouteText] = None
    seed: Optional[int] = None
    url: str
    download_url: Optional[str] = None
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None


class AudioResult(ToolOutput):
    model: Optional[RouteText] = None
    group_id: Optional[RouteText] = None
    clips: list[AudioClip] = []
    # Options sent that the model does not take; they were left out.
    ignored_options: list[RouteText] = []
    # Values Unsloth Studio used instead of the ones asked for, such as a clamped duration_s.
    applied: dict[str, float] = {}
    warnings: list[RouteText] = []
    # Pass to cancel(kind="audio", id=...) from another call to stop the run.
    cancel_id: Optional[str] = None
    # False when Studio could not save the clip to Audio history; the audio then comes back inline only.
    saved: bool = True

    @classmethod
    def from_route(cls, payload: Any, **overrides: Any) -> "AudioResult":
        applied = payload.get("applied") if isinstance(payload, dict) else None
        overrides.setdefault(
            "applied",
            {
                name: float(value)
                for name, value in (applied.items() if isinstance(applied, dict) else ())
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
        )
        return super().from_route(payload, **overrides)


class AudioRow(ToolOutput):
    id: RouteText
    url: str
    download_url: Optional[str] = None
    prompt: Optional[RouteText] = None
    model: Optional[RouteText] = None
    workflow: Optional[RouteText] = None
    role: Optional[RouteText] = None
    group_id: Optional[RouteText] = None
    voice_id: Optional[RouteText] = None
    source_clip_id: Optional[RouteText] = None
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None
    created_at: Optional[RouteText] = None
    pinned: bool = False


class VoiceRow(ToolOutput):
    id: RouteText
    name: RouteText
    # Needs the API key: voices have no key-free link.
    url: str
    transcript: Optional[RouteText] = None
    language: Optional[RouteText] = None
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None
    created_at: Optional[RouteText] = None


class VoicesResult(ToolOutput):
    action: RouteText
    voices: Optional[list[VoiceRow]] = None
    voice: Optional[VoiceRow] = None
    removed: Optional[bool] = None


class TranscriptSegment(ToolOutput):
    start: Optional[float] = None
    end: Optional[float] = None
    text: str = ""
    # The speaker's label, such as "Speaker 1", when transcribe was asked for speakers.
    speaker: Optional[RouteText] = None


class TranscriptResult(ToolOutput):
    text: str
    transcript_id: Optional[RouteText] = None
    language: Optional[RouteText] = None
    model: Optional[RouteText] = None
    segments: Optional[list[TranscriptSegment]] = None
    # 1 - word error rate against expected_text, from 0 to 1; None without expected_text or when
    # the texts are too long to align.
    match_score: Optional[float] = None
    saved_to_history: bool = False


class TranscriptRow(ToolOutput):
    id: RouteText
    title: Optional[RouteText] = None
    text: str = ""
    model: Optional[RouteText] = None
    language: Optional[RouteText] = None
    duration: Optional[float] = None
    segment_count: Optional[int] = None
    created_at: Optional[RouteText] = None


class AnalysisResult(ToolOutput):
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    # BS.1770 loudness; None for silence.
    integrated_lufs: Optional[float] = None
    true_peak_dbtp: Optional[float] = None
    clipped_samples: Optional[int] = None
    clipping_ratio: Optional[float] = None
    # Where the sound starts and ends; None for silence.
    speech_start_s: Optional[float] = None
    speech_end_s: Optional[float] = None
    # Model-written, so never rewritten; only with expected_text.
    transcript: Optional[str] = None
    # None when the texts are too long to align.
    match_score: Optional[float] = None
    stt_model: Optional[RouteText] = None


class VideoJobRef(ToolOutput):
    id: RouteText
    status: RouteText
    # Set while the video waits in this MCP server's queue behind another video; 1 is next.
    queue_position: Optional[int] = None
    # Seconds until the queued video is expected to start, from the running video's progress.
    eta_s: Optional[float] = None
    progress: Optional[int] = None
    progress_percent: Optional[float] = None

    def model_post_init(self, context: Any) -> None:
        if self.progress_percent is None and self.progress is not None:
            self.progress_percent = float(self.progress)

    model: Optional[RouteText] = None
    seconds: Optional[RouteText] = None
    size: Optional[RouteText] = None
    # Pass it again with the same settings to repeat this take.
    seed: Optional[int] = None


class VideoInfo(ToolOutput):
    # The MP4, for the agent or its user to download; the tool never fetches it.
    url: str
    download_url: Optional[str] = None
    thumbnail_inline: bool = False


class VideoRow(ToolOutput):
    id: RouteText
    url: str
    download_url: Optional[str] = None
    prompt: Optional[RouteText] = None
    model: Optional[RouteText] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    duration_s: Optional[float] = None
    seed: Optional[int] = None
    has_audio: bool = False
    created_at: Optional[RouteText] = None
    pinned: bool = False


class MediaPage(ToolOutput):
    kind: RouteText
    images: Optional[list[ImageRow]] = None
    audio: Optional[list[AudioRow]] = None
    videos: Optional[list[VideoRow]] = None
    transcripts: Optional[list[TranscriptRow]] = None
    voices: Optional[list[VoiceRow]] = None
    # Pass it back as cursor for the next page; None on the last page.
    next_cursor: Optional[str] = None


class JobSummary(ToolOutput):
    id: RouteText
    status: RouteText
    progress_percent: Optional[float] = None


class ExportJobSummary(JobSummary):
    format: RouteText
    phase: RouteText
    # Relative to Unsloth Studio's exports folder, or only a name; never a host path.
    output: Optional[RouteText] = None
    error: Optional[RouteText] = None


class RecipeError(ToolOutput):
    message: RouteText
    path: Optional[RouteText] = None


class RecipeResult(ToolOutput):
    mode: RouteText
    valid: Optional[bool] = None
    errors: Optional[list[RecipeError]] = None
    job_id: Optional[RouteText] = None


class RecipeInfo(ToolOutput):
    stage: Optional[RouteText] = None
    rows: Optional[int] = None
    # The saved dataset's name under Studio's recipes folder, never its path.
    dataset: Optional[RouteText] = None
    total_rows: Optional[int] = None
    data_rows: Optional[list[dict[str, Any]]] = None


class DownloadRef(ToolOutput):
    # What get_job(kind="download") and cancel(kind="model_download") take; None for a speech-to-text download.
    id: Optional[RouteText] = None
    model: Optional[RouteText] = None
    kind: Optional[RouteText] = None
    variant: Optional[RouteText] = None
    status: RouteText
    attached: bool = False
    error: Optional[RouteText] = None


class ExportInfo(ToolOutput):
    format: RouteText
    # Relative to Studio's exports folder, or only a name; never a host path.
    output: Optional[RouteText] = None
    phase: RouteText


class JobStatus(ToolOutput):
    kind: RouteText
    id: Optional[RouteText] = None
    # A video job's seed.
    seed: Optional[int] = None
    status: Optional[RouteText] = None
    progress_percent: Optional[float] = None
    # While the job runs: seconds to wait before the next get_job call without wait_s.
    poll_after_s: Optional[float] = None
    training: Optional[TrainingJob] = None
    load: Optional[LoadResult] = None
    queue_position: Optional[int] = None
    eta_s: Optional[float] = None
    error: Optional[RouteText] = None
    error_code: Optional[RouteText] = None
    video: Optional[VideoInfo] = None
    recipe: Optional[RecipeInfo] = None
    download: Optional[DownloadRef] = None
    export: Optional[ExportInfo] = None
    jobs: Optional[list[Union[ExportJobSummary, JobSummary]]] = None


class LocalDataset(ToolOutput):
    id: RouteText
    label: RouteText
    source: RouteText = "local"
    rows: Optional[int] = None
    # What config.local_datasets takes for any account; ``id`` alone works for the owner only.
    ref: Optional[RouteText] = None


class CachedDataset(ToolOutput):
    repo_id: RouteText
    size_bytes: Optional[int] = None


class DatasetFormat(ToolOutput):
    detected_format: Optional[RouteText] = None
    requires_manual_mapping: Optional[bool] = None
    columns: list[RouteText] = []
    suggested_mapping: Optional[dict[str, RouteText]] = None
    is_image: Optional[bool] = None
    is_audio: Optional[bool] = None
    total_rows: Optional[int] = None
    warning: Optional[RouteText] = None


class DatasetDownload(ToolOutput):
    repo_id: RouteText
    state: RouteText
    error: Optional[RouteText] = None


class DatasetUpload(ToolOutput):
    id: RouteText
    filename: RouteText
    # The handle config.local_datasets takes; never the stored path.
    ref: RouteText


class ImageDatasetUpload(ToolOutput):
    # What start_training kind "diffusion" takes as config.data_dir.
    name: RouteText
    image_count: Optional[int] = None
    caption_count: Optional[int] = None
    uploaded: Optional[int] = None


class DatasetsResult(ToolOutput):
    local: Optional[list[LocalDataset]] = None
    cached: Optional[list[CachedDataset]] = None
    format: Optional[DatasetFormat] = None
    download: Optional[DatasetDownload] = None
    uploaded: Optional[DatasetUpload] = None
    image_dataset: Optional[ImageDatasetUpload] = None


ConfigScalar = Union[StrictBool, StrictInt, StrictFloat, RouteText]
ConfigValue = Union[ConfigScalar, list[ConfigScalar], dict[str, ConfigScalar]]


class TrainingStarted(ToolOutput):
    # None when validate_only checked the config without starting anything.
    job_id: Optional[RouteText] = None
    status: RouteText
    message: Optional[RouteText] = None
    # validate_only: the config as the route will read it, without paths or secrets.
    config: Optional[dict[str, ConfigValue]] = None
    # What cancel(kind="training_start") takes while the job has not begun.
    start_request_id: Optional[RouteText] = None


TrainingScalar = Union[StrictBool, StrictInt, StrictFloat, RouteText]


class TrainingRun(ToolOutput):
    id: RouteText
    status: RouteText
    model_name: Optional[RouteText] = None
    dataset_name: Optional[RouteText] = None
    display_name: Optional[RouteText] = None
    started_at: Optional[RouteText] = None
    ended_at: Optional[RouteText] = None
    final_step: Optional[int] = None
    final_loss: Optional[float] = None
    can_resume: bool = False
    error_message: Optional[RouteText] = None
    # The run's folder name, which is also its checkpoint name for export_model.
    run_folder: Optional[RouteText] = None


class TrainingRunDetail(TrainingRun):
    config: dict[str, TrainingScalar] = {}
    step_history: Optional[list[int]] = None
    loss_history: Optional[list[float]] = None


class CheckpointName(ToolOutput):
    run: RouteText
    # "<run folder>" for the final weights, "<run folder>/<checkpoint>" for an intermediate one.
    name: RouteText
    loss: Optional[float] = None
    base_model: Optional[RouteText] = None
    peft_type: Optional[RouteText] = None
    lora_rank: Optional[int] = None
    is_quantized: bool = False


class TrainingRuns(ToolOutput):
    runs: list[TrainingRun] = []
    total: Optional[int] = None
    run: Optional[TrainingRunDetail] = None
    checkpoints: Optional[list[CheckpointName]] = None


class ExportJobRef(ToolOutput):
    job_id: RouteText
    status: RouteText = "running"


class CancelResult(ToolOutput):
    kind: RouteText
    id: Optional[RouteText] = None
    cancelled: bool
    message: Optional[RouteText] = None
