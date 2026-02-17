import sys
import types
from importlib.machinery import ModuleSpec

def create_mock_module(name):
    module = types.ModuleType(name)
    module.__file__ = f'{name}.py'
    module.__path__ = []
    module.__spec__ = ModuleSpec(name, None)
    module.__package__ = name.rsplit('.', 1)[0] if '.' in name else name
    return module

def _setup_unsloth_zoo_mocks():
    zoo = create_mock_module('unsloth_zoo')
    sys.modules['unsloth_zoo'] = zoo
    
    zoo_vision = create_mock_module('unsloth_zoo.vision_utils')
    zoo_vision.process_vision_info = lambda *args, **kwargs: None
    zoo_vision._process_vision_info_and_tokenize = lambda *args, **kwargs: None
    zoo_vision.get_png_file_from_url = lambda *args, **kwargs: None
    sys.modules['unsloth_zoo.vision_utils'] = zoo_vision
    
    zoo_compiler = create_mock_module('unsloth_zoo.compiler')
    zoo_compiler.get_compile_dynamic = lambda *args, **kwargs: lambda f: f
    sys.modules['unsloth_zoo.compiler'] = zoo_compiler
    
    zoo_training_utils = create_mock_module('unsloth_zoo.training_utils')
    sys.modules['unsloth_zoo.training_utils'] = zoo_training_utils
    
    zoo_loss_utils = create_mock_module('unsloth_zoo.loss_utils')
    sys.modules['unsloth_zoo.loss_utils'] = zoo_loss_utils
    
    zoo_gradient_checkpointing = create_mock_module('unsloth_zoo.gradient_checkpointing')
    sys.modules['unsloth_zoo.gradient_checkpointing'] = zoo_gradient_checkpointing
    
    zoo_patching_utils = create_mock_module('unsloth_zoo.patching_utils')
    sys.modules['unsloth_zoo.patching_utils'] = zoo_patching_utils
    
    zoo_llama_causal_model = create_mock_module('unsloth_zoo.llama_causal_model')
    sys.modules['unsloth_zoo.llama_causal_model'] = zoo_llama_causal_model
    
    zoo_mistral_causal_model = create_mock_module('unsloth_zoo.mistral_causal_model')
    sys.modules['unsloth_zoo.mistral_causal_model'] = zoo_mistral_causal_model
    
    zoo_logging_utils = create_mock_module('unsloth_zoo.logging_utils')
    sys.modules['unsloth_zoo.logging_utils'] = zoo_logging_utils
    
    zoo_peft_utils = create_mock_module('unsloth_zoo.peft_utils')
    sys.modules['unsloth_zoo.peft_utils'] = zoo_peft_utils
    
    zoo_temporary_patches = create_mock_module('unsloth_zoo.temporary_patches')
    sys.modules['unsloth_zoo.temporary_patches'] = zoo_temporary_patches
    
    zoo_token_utils = create_mock_module('unsloth_zoo.token_utils')
    sys.modules['unsloth_zoo.token_utils'] = zoo_token_utils

_setup_unsloth_zoo_mocks()
