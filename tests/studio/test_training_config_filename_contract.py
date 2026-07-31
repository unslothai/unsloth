from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FILENAME_HELPER = REPO / "studio/frontend/src/features/studio/wizard/training-config-file.ts"
NATIVE_FILE_DIALOGS = REPO / "studio/src-tauri/src/native_file_dialogs.rs"
NATIVE_FILES = REPO / "studio/frontend/src/lib/native-files.ts"
CONFIG_ACTIONS = REPO / "studio/frontend/src/features/studio/wizard/config-actions.tsx"


def test_training_config_filename_is_bounded_by_utf8_bytes():
    source = FILENAME_HELPER.read_text(encoding = "utf-8")

    assert "const FILENAME_SEGMENT_MAX_BYTES = 64;" in source
    assert "for (const character of value)" in source
    assert "character.codePointAt(0)" in source
    assert "bytes + characterBytes > maxBytes" in source
    assert ".slice(0, 64)" not in source
    assert ').replace(TRAILING_WINDOWS_FILENAME_PATTERN, "");' in source


def test_tauri_native_save_dialog_recognizes_yaml_configs():
    source = NATIVE_FILE_DIALOGS.read_text(encoding = "utf-8")

    assert 'Some("yaml") | Some("yml") => ("YAML", vec!["yaml", "yml"])' in source
    assert "fn training_configs_use_a_yaml_save_filter()" in source


def test_tauri_load_dialog_reads_bounded_yaml_configs():
    native = NATIVE_FILE_DIALOGS.read_text(encoding = "utf-8")
    helper = NATIVE_FILES.read_text(encoding = "utf-8")
    actions = CONFIG_ACTIONS.read_text(encoding = "utf-8")

    assert 'TRAINING_CONFIG_EXTENSIONS: &[&str] = &["yaml", "yml"]' in native
    assert "MAX_TRAINING_CONFIG_BYTES" in native
    assert "pick_native_training_config" in native
    assert 'invoke<NativeImportedTextFile | null>("pick_native_training_config")' in helper
    assert "pickNativeTrainingConfig" in actions
    assert "if (!isTauri)" in actions
    assert 'accept=".yaml,.yml"' in actions
