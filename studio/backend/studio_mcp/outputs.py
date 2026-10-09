# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Typed tool outputs. Several Unsloth Studio routes still return host paths to API-key callers, so a tool never passes route JSON through: it copies named fields into a ToolOutput, which refuses anything undeclared. Free text that came from a route (a status message, an error) is declared ``RouteText`` and scrubbed of paths as a backstop; model-written text and Unsloth Studio URLs are plain ``str`` and left alone."""

from __future__ import annotations

from typing import Annotated, Any, Optional, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, StrictBool, StrictFloat, StrictInt

from hub.utils.host_paths import redact_paths_in_text

RouteText = Annotated[str, AfterValidator(redact_paths_in_text)]


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra = "forbid")


class ChatModel(ToolOutput):
    id: RouteText
    display_name: Optional[RouteText] = None
    # Known for the active model only.
    is_gguf: Optional[bool] = None


class ChatSlot(ToolOutput):
    loaded: list[ChatModel] = []
    loading: list[RouteText] = []


class ImageSlot(ToolOutput):
    loaded: bool = False
    model: Optional[RouteText] = None
    family: Optional[RouteText] = None
    # An image is being generated right now.
    generating: bool = False


class VideoSlot(ToolOutput):
    loaded: bool = False
    model: Optional[RouteText] = None
    # A video is being generated right now.
    generating: bool = False


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
    display_name: Optional[RouteText] = None
    # Models Studio unloaded to make room.
    evicted: list[RouteText] = []


class UnloadResult(ToolOutput):
    kind: RouteText
    model: Optional[RouteText] = None
    unloaded: bool


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


class EmbedResult(ToolOutput):
    model: Optional[RouteText] = None
    dimensions: int
    embeddings: list[list[float]]


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
    # Built from the agent's own address for /mcp.
    url: str
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


class ImageResult(ToolOutput):
    images: list[ImageItem]


class AudioClip(ToolOutput):
    id: RouteText
    role: RouteText = "output"
    url: str
    duration_s: Optional[float] = None
    sample_rate: Optional[int] = None


class AudioResult(ToolOutput):
    model: Optional[RouteText] = None
    group_id: Optional[RouteText] = None
    clips: list[AudioClip] = []
    # False when Studio could not save the clip to Audio history; the audio then comes back inline only.
    saved: bool = True


class TranscriptSegment(ToolOutput):
    start: Optional[float] = None
    end: Optional[float] = None
    # Model-written.
    text: str = ""


class TranscriptResult(ToolOutput):
    # Model-written.
    text: str
    language: Optional[RouteText] = None
    model: Optional[RouteText] = None
    segments: Optional[list[TranscriptSegment]] = None
    saved_to_history: bool = False


class VideoJobRef(ToolOutput):
    id: RouteText
    status: RouteText
    progress: Optional[int] = None
    model: Optional[RouteText] = None
    seconds: Optional[RouteText] = None
    size: Optional[RouteText] = None


class VideoInfo(ToolOutput):
    # The MP4, for the agent or its user to download; the tool never fetches it.
    url: str
    thumbnail_inline: bool = False


class JobSummary(ToolOutput):
    id: RouteText
    status: RouteText
    progress_percent: Optional[float] = None


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
    # Generated rows, as the recipe wrote them.
    data_rows: Optional[list[dict[str, Any]]] = None


class ExportInfo(ToolOutput):
    format: RouteText
    # Relative to Studio's exports folder, or only a name; never a host path.
    output: Optional[RouteText] = None
    phase: RouteText


class JobStatus(ToolOutput):
    kind: RouteText
    id: Optional[RouteText] = None
    status: Optional[RouteText] = None
    progress_percent: Optional[float] = None
    error: Optional[RouteText] = None
    video: Optional[VideoInfo] = None
    recipe: Optional[RecipeInfo] = None
    export: Optional[ExportInfo] = None
    # Recent jobs, when no id was given.
    jobs: Optional[list[JobSummary]] = None


class LocalDataset(ToolOutput):
    id: RouteText
    label: RouteText
    source: RouteText = "local"
    rows: Optional[int] = None


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


class DatasetsResult(ToolOutput):
    local: Optional[list[LocalDataset]] = None
    cached: Optional[list[CachedDataset]] = None
    format: Optional[DatasetFormat] = None
    download: Optional[DatasetDownload] = None


class TrainingStarted(ToolOutput):
    # None when validate_only checked the config without starting anything.
    job_id: Optional[RouteText] = None
    status: RouteText
    message: Optional[RouteText] = None
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
