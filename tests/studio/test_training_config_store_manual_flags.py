from pathlib import Path


STORE = (
    Path(__file__).resolve().parents[2]
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "training"
    / "stores"
    / "training-config-store.ts"
)


def test_apply_config_patch_marks_completion_intent_manual():
    source = STORE.read_text(encoding = "utf-8")
    block_start = source.index("applyConfigPatch: (config: BackendModelConfig) => {")
    block_end = source.index("},", block_start)
    block = source[block_start:block_end]

    assert "patch.trainOnCompletions !== undefined" in block
    assert "{ trainOnCompletionsManuallySet: true }" in block


def test_apply_config_patch_marks_learning_rate_intent_manual():
    source = STORE.read_text(encoding = "utf-8")
    block_start = source.index("applyConfigPatch: (config: BackendModelConfig) => {")
    block_end = source.index("},", block_start)
    block = source[block_start:block_end]

    assert "patch.learningRate !== undefined" in block
    assert "{ learningRateManuallySet: true }" in block


def test_audio_and_vision_completion_guards_are_outside_manual_default_gate():
    source = STORE.read_text(encoding = "utf-8")

    model_block_start = source.index("const isAudio = !!modelDetails.is_audio;")
    model_block_end = source.index("// Use backend model_type when available", model_block_start)
    model_block = source[model_block_start:model_block_end]
    assert "if (!trainOnCompletionsManuallySet)" not in model_block
    assert "patch.trainOnCompletions = false;" in model_block

    dataset_block_start = source.index("const updates: Record<string, unknown> = {")
    dataset_block_end = source.index("set(updates);", dataset_block_start)
    dataset_block = source[dataset_block_start:dataset_block_end]
    assert "!get().trainOnCompletionsManuallySet" not in dataset_block
    assert "updates.trainOnCompletions = false;" in dataset_block
