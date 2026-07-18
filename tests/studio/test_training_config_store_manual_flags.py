from pathlib import Path


STORE = Path(__file__).resolve().parents[2] / "studio" / "frontend" / "src" / "features" / "training" / "stores" / "training-config-store.ts"


def test_apply_config_patch_marks_completion_intent_manual():
    source = STORE.read_text(encoding = "utf-8")
    block_start = source.index("applyConfigPatch: (config: BackendModelConfig) => {")
    block_end = source.index("},", block_start)
    block = source[block_start:block_end]

    assert "patch.trainOnCompletions !== undefined" in block
    assert "{ trainOnCompletionsManuallySet: true }" in block
