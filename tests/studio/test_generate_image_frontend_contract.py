# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat Images pill reaches generate_image / generate_video on the models loaded in Unsloth Studio."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "studio" / "frontend" / "src"
BACKEND = ROOT / "studio" / "backend"


def source(path: str) -> str:
    return (FRONTEND / path).read_text(encoding = "utf-8")


def _names(literal: str) -> set[str]:
    return set(re.findall(r'"([^"]+)"', literal))


def test_media_chat_tools_match_backend() -> None:
    backend = (BACKEND / "core" / "inference" / "media_chat_tools.py").read_text(encoding = "utf-8")
    frontend = source("components/assistant-ui/sandbox-files.ts")
    backend_set = re.search(r"^MEDIA_CHAT_TOOLS = frozenset\(\{([^}]*)\}\)", backend, re.M)
    frontend_set = re.search(
        r"export const MEDIA_CHAT_TOOLS[^=]*= new Set\(\[([^\]]*)\]\)", frontend
    )
    assert backend_set and frontend_set
    assert (
        _names(backend_set.group(1))
        == _names(frontend_set.group(1))
        == {
            "generate_image",
            "generate_video",
        }
    )
    assert "...MEDIA_CHAT_TOOLS," in frontend
    # The fold exemption reads the same set.
    assert "MEDIA_CHAT_TOOLS.has(" in source("components/assistant-ui/tool-fold-exemptions.ts")


def test_thread_registers_both_media_cards() -> None:
    thread = source("components/assistant-ui/thread.tsx")
    assert "generate_image: ImageGenerationToolUIConfirmable," in thread
    assert "generate_video: VideoGenerationToolUIConfirmable," in thread
    assert "const VideoGenerationToolUIConfirmable = withToolConfirmation(" in thread
    assert "image_generation: ImageGenerationToolUIConfirmable," in thread


def test_adapter_sends_local_media_tools_on_every_path() -> None:
    adapter = source("features/chat/api/chat-adapter.ts")
    assert adapter.count("...localMediaTools,") == 3
    assert adapter.count("localMediaTools.length > 0 ||") == 2
    assert "localMediaTools.length === 0" in adapter


def test_local_results_have_no_edit_and_video_plays_from_the_sandbox() -> None:
    card = source("components/assistant-ui/tool-ui-image-generation.tsx")
    # Edit stays on the hosted figure only; the sandbox figure has Download alone.
    sandbox_figure = card[
        card.index("function SandboxMediaFigure(") : card.index(
            "function GeneratedImagePlaceholder"
        )
    ]
    assert "Edit" not in sandbox_figure and "openOverlay" not in sandbox_figure
    assert "useSandboxImage<HTMLDivElement>(url)" in sandbox_figure
    assert "<video" in sandbox_figure and 'preload="metadata"' in sandbox_figure
    video = card[card.index("const VideoGenerationToolUIImpl") :]
    assert 'kind="video"' in video and "sandboxFilePath(" in video
    assert "export function useSandboxImage<T extends Element = HTMLImageElement>(" in source(
        "components/assistant-ui/use-sandbox-image.ts"
    )


def test_details_sheet_labels_media_tools() -> None:
    sheet = source("components/assistant-ui/message-response-details-sheet.tsx")
    assert 'generate_image: "Images",' in sheet
    assert 'generate_video: "Video",' in sheet
    assert 'normalized === "generate_video"' in sheet
