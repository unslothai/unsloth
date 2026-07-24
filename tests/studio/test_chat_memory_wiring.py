"""Regression checks for persisted chat-memory thread ID wiring."""

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ADAPTER_SRC = (WORKSPACE / "studio/frontend/src/features/chat/api/chat-adapter.ts").read_text()


def test_memory_scope_uses_remote_id_after_first_save():
    start = ADAPTER_SRC.index("await ThreadAutosaveHandle.awaitFirstSave(resolvedThreadId, null);")
    end = ADAPTER_SRC.index("// ── Audio model path", start)
    scope_block = ADAPTER_SRC[start:end]

    assert "const memoryThreadId = memoryRuntime.activeThreadId || undefined;" in scope_block
    assert "ownsThread(memoryThreadId)" in scope_block
    assert "threadId: memoryThreadId" in scope_block
    assert "timeoutMs === null" in ADAPTER_SRC


def test_memory_capture_uses_persisted_memory_scope_thread_id():
    start = ADAPTER_SRC.index("scheduleMemoryCapture({", ADAPTER_SRC.index("const memoryScope ="))
    end = ADAPTER_SRC.index("buildPayload:", start)
    schedule_block = ADAPTER_SRC[start:end]

    assert "threadId: memoryScope.thread_id" in schedule_block
    assert "threadId: resolvedThreadId" not in schedule_block
