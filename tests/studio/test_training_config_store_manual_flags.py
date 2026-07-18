from pathlib import Path


STORE = Path(__file__).resolve().parents[2] / "studio" / "frontend" / "src" / "features" / "training" / "stores" / "training-config-store.ts"


def test_v12_migration_seeds_context_length_manual_intent():
    source = STORE.read_text(encoding = "utf-8")
    block_start = source.index("if (version < 13) {")
    block_end = source.index("return s as unknown as TrainingConfigStore;", block_start)
    block = source[block_start:block_end]

    assert 'typeof s.contextLengthManuallySet !== "boolean"' in block
    assert "Number.isFinite(s.contextLength)" in block
    assert "s.contextLength !== DEFAULT_HYPERPARAMS.contextLength" in block
