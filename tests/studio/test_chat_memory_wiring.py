"""Regression checks for persisted chat-memory thread ID wiring."""

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ADAPTER_SRC = (WORKSPACE / "studio/frontend/src/features/chat/api/chat-adapter.ts").read_text()


def test_memory_scope_uses_remote_id_after_first_save():
    start = ADAPTER_SRC.index("await ThreadAutosaveHandle.awaitFirstSave(resolvedThreadId);")
    end = ADAPTER_SRC.index("// ── Audio model path", start)
    scope_block = ADAPTER_SRC[start:end]

    assert "const memoryThreadId = memoryRuntime.activeThreadId || undefined;" in scope_block
    assert "ownsThread(memoryThreadId)" in scope_block
    assert "threadId: memoryThreadId" in scope_block
