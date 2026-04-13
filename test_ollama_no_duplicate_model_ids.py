"""Test that no model ID appears in multiple Ollama template tuples (would cause wrong template)."""
import os, sys, re
sys.path.insert(0, os.path.dirname(__file__))

OLLAMA_FILE = os.path.join(os.path.dirname(__file__), "unsloth", "ollama_template_mappers.py")


def _load_ollama_mapper():
    text = open(OLLAMA_FILE).read()
    start = text.find("OLLAMA_TEMPLATE_TO_MODEL_MAPPER = {")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: break
    g = {}
    exec(text[start:i+1], g)
    return g["OLLAMA_TEMPLATE_TO_MODEL_MAPPER"]


def test_no_model_id_in_multiple_ollama_templates():
    mapper = _load_ollama_mapper()
    seen = {}
    conflicts = []
    for template, models in mapper.items():
        for model in models:
            if model in seen and seen[model] != template:
                conflicts.append(f"{model}: in both '{seen[model]}' and '{template}'")
            seen[model] = template
    assert not conflicts, "Model ID conflicts in OLLAMA_TEMPLATE_TO_MODEL_MAPPER:\n" + "\n".join(conflicts)


def test_gemma4_ids_only_in_gemma4():
    """All Gemma-4 model IDs must be in the 'gemma4' template only."""
    mapper = _load_ollama_mapper()
    for template, models in mapper.items():
        if template == "gemma4":
            continue
        for model in models:
            if "gemma-4-" in model and "medgemma" not in model:
                assert False, f"{model} is in template '{template}' instead of 'gemma4'"


def test_no_empty_tuples():
    mapper = _load_ollama_mapper()
    for template, models in mapper.items():
        assert len(models) > 0, f"Empty tuple for Ollama template '{template}'"
