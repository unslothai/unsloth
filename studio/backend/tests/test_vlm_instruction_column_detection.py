from hub.utils.dataset_format import check_dataset_format


def _vlm_row(**extra):
    return {"image": "sample.jpg", "text": "A caption", **extra}


def test_detects_non_empty_explicit_instruction_column():
    result = check_dataset_format([_vlm_row(instruction = "Describe it")], is_vlm = True)
    assert result["detected_instruction_column"] == "instruction"


def test_ignores_empty_first_row_instruction_column():
    result = check_dataset_format([_vlm_row(instruction = "   ")], is_vlm = True)
    assert result["detected_instruction_column"] is None


def test_vlm_without_instruction_column_reports_none():
    result = check_dataset_format([_vlm_row()], is_vlm = True)
    assert result["detected_instruction_column"] is None
