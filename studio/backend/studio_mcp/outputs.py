# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Typed tool outputs. Several Studio routes still return host paths to API-key callers, so a tool never passes route JSON through: it copies named fields into a ToolOutput, which refuses anything undeclared. Free text that came from a route (a status message, an error) is declared ``RouteText`` and scrubbed of paths as a backstop; model-written text and Studio URLs are plain ``str`` and left alone."""

from __future__ import annotations

from typing import Annotated, Optional, Union

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


class VideoSlot(ToolOutput):
    loaded: bool = False
    model: Optional[RouteText] = None


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
