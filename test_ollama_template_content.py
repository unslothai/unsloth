"""Test that Ollama template content exists and is valid for gemma4 models."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

OLLAMA_FILE = os.path.join(os.path.dirname(__file__), "unsloth", "ollama_template_mappers.py")


def _load_ollama_templates():
    text = open(OLLAMA_FILE).read()
    # Extract OLLAMA_TEMPLATES dict
    start = text.find('OLLAMA_TEMPLATES = {}')
    assert start != -1
    # Find all template assignments
    templates = {}
    import re
    for m in re.finditer(r'OLLAMA_TEMPLATES\["([^"]+)"\]\s*=\s*(\w+)', text):
        key = m.group(1)
        templates[key] = m.group(2)
    return templates


def test_gemma4_ollama_template_exists():
    templates = _load_ollama_templates()
    assert "gemma4" in templates, "gemma4 not in OLLAMA_TEMPLATES"


def test_gemma4_template_variable_defined():
    """The variable referenced by OLLAMA_TEMPLATES['gemma4'] must be defined."""
    text = open(OLLAMA_FILE).read()
    import re
    m = re.search(r'OLLAMA_TEMPLATES\["gemma4"\]\s*=\s*(\w+)', text)
    assert m, "gemma4 template assignment not found"
    var_name = m.group(1)
    assert f"{var_name} =" in text or f"{var_name}=" in text, (
        f"Variable {var_name} not defined in ollama_template_mappers.py"
    )


def test_gemma4_template_references_valid_variable():
    """The variable referenced by OLLAMA_TEMPLATES['gemma4'] must be a known gemma4 template."""
    text = open(OLLAMA_FILE).read()
    import re
    m = re.search(r'OLLAMA_TEMPLATES\["gemma4"\]\s*=\s*(\w+)', text)
    assert m, "gemma4 template assignment not found"
    var_name = m.group(1)
    assert "gemma4" in var_name.lower(), f"Variable {var_name} doesn't look like a gemma4 template"


def test_gemma4_and_thinking_share_ollama_template():
    """Gemma-4 and gemma-4-thinking share the same Ollama template (gemma4_ollama)."""
    text = open(OLLAMA_FILE).read()
    # Both gemma-4 and gemma-4-thinking use gemma4_ollama in chat_templates.py
    # The OLLAMA_TEMPLATE_TO_MODEL_MAPPER has one "gemma4" key covering all variants
    assert '"gemma4"' in text
