from ._deepseek import register_deepseek_models as _register_deepseek_models
from ._gemma import register_gemma_models as _register_gemma_models
from ._llama import register_llama_models as _register_llama_models
from ._mistral import register_mistral_models as _register_mistral_models
from ._phi import register_phi_models as _register_phi_models
from ._qwen import register_qwen_models as _register_qwen_models


def register_models():    
    _register_deepseek_models()
    _register_gemma_models()
    _register_llama_models()
    _register_mistral_models()
    _register_phi_models()
    _register_qwen_models()

