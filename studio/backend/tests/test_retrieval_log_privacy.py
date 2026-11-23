# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
from pathlib import Path

import pytest
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.inference import tools
from core.rag import conversation_archive, tool
from storage import rag_db
from utils.debug_log_reader import read_tail


@pytest.mark.parametrize("kind", ["conversation", "documents"])
def test_successful_retrieval_logs_counts_without_user_text(monkeypatch, tmp_path, kind):
    query = "Summarize the confidential Marigold acquisition notes."
    conversation = [{"role": "user", "content": query}]
    found = ("Retrieved passage", [{"documentId": "fixture", "text": "Retrieved passage"}])
    path = tmp_path / "server.log"
    with path.open("w", encoding = "utf-8") as stream:
        logger = structlog.wrap_logger(
            structlog.PrintLogger(stream),
            processors = [
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class = structlog.stdlib.BoundLogger,
        )
        monkeypatch.setattr(tools, "logger", logger)
        if kind == "conversation":
            monkeypatch.setattr(conversation_archive, "enabled", lambda: True)
            monkeypatch.setattr(conversation_archive, "recall", lambda *a, **kw: found)
            built = tools.build_conversation_recall(conversation, "fixture")
            expected = "Conversation recall: 1 earlier passage(s)"
        else:
            monkeypatch.setattr(rag_db, "rag_available", lambda: True)
            monkeypatch.setattr(tool, "search_for_autoinject", lambda **kw: found)
            built = tools._build_rag_autoinject(
                conversation, {"autoinject": True, "kb_id": "fixture"}
            )
            expected = "RAG auto-inject: 1 passage(s)"

    assert built is not None
    args = built["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args)["query"] == query
    raw = path.read_text(encoding = "utf-8")
    assert expected in raw
    assert query not in raw
    assert "Marigold" not in raw
    copied_lines = "\n".join(read_tail(path).lines)
    assert expected in copied_lines
    assert "Marigold" not in copied_lines
