from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FILENAME_HELPER = (
    REPO
    / "studio/frontend/src/features/studio/wizard/training-config-file.ts"
)
NATIVE_FILE_DIALOGS = REPO / "studio/src-tauri/src/native_file_dialogs.rs"


def test_training_config_filename_is_bounded_by_utf8_bytes():
    source = FILENAME_HELPER.read_text(encoding = "utf-8")

    assert "const FILENAME_SEGMENT_MAX_BYTES = 64;" in source
    assert "for (const character of value)" in source
    assert "character.codePointAt(0)" in source
    assert "bytes + characterBytes > maxBytes" in source
    assert ".slice(0, 64)" not in source
    assert (
        ").replace(TRAILING_WINDOWS_FILENAME_PATTERN, \"\");"
        in source
    )


def test_tauri_native_save_dialog_recognizes_yaml_configs():
    source = NATIVE_FILE_DIALOGS.read_text(encoding = "utf-8")

    assert (
        'Some("yaml") | Some("yml") => ("YAML", vec!["yaml", "yml"])'
        in source
    )
    assert "fn training_configs_use_a_yaml_save_filter()" in source
