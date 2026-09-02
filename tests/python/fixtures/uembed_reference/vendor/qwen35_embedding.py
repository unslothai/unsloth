import os
import json
import torch
import torch.nn.functional as F
import unicodedata
import numpy as np
import logging

from PIL import Image
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from transformers import AutoModel, AutoConfig, AutoProcessor
from qwen_vl_utils.vision_process import process_vision_info
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel, Qwen3_5Model, Qwen3_5Config
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from tokenizers import processors

logger = logging.getLogger(__name__)
import torch.distributed as dist
# Constants for configuration
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS


@dataclass
class Qwen3_5ForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None

class Qwen3_5ForEmbedding(Qwen3_5PreTrainedModel):
    """Qwen3.5 模型的 Embedding 封装，使用 AutoModel 加载。"""
    config_class = AutoConfig
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.post_init()

    def enable_bidirectional_attention(self):
        self.config.is_causal = False
        if hasattr(self.config, 'text_config'):
            self.config.text_config.is_causal = False
        self.model.language_model.config.is_causal = False
        for layer in self.model.language_model.layers:
            if getattr(layer, 'layer_type', None) == "full_attention":
                layer.self_attn.is_causal = False
        logger.info("Bidirectional attention enabled")

    def get_video_features(self, pixel_values_videos: torch.FloatTensor,
                           video_grid_thw: Optional[torch.LongTensor] = None, **kwargs):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_image_features(self, pixel_values: torch.FloatTensor,
                           image_grid_thw: Optional[torch.LongTensor] = None, **kwargs):
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3_5ForEmbeddingOutput]:
        self.model.rope_deltas = None  # this is set for inference cache
        # print(f"Rank {dist.get_rank()}, device: {torch.cuda.current_device()}, value_device: {next(self.model.parameters()).device}")
        # print(f"Rank {dist.get_rank()}, inputs shape: {input_ids.shape}, inputs device: {input_ids.device}")
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3_5ForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def sample_frames(frames: List[Union[str, Image.Image]], max_segments: int) -> List[Union[str, Image.Image]]:
    duration = len(frames)
    if duration <= max_segments:
        return frames
    frame_id_array = np.linspace(0, duration - 1, max_segments, dtype=int)
    return [frames[idx] for idx in frame_id_array.tolist()]


def is_image_path(path: str) -> bool:
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    if path.startswith(('http://', 'https://')):
        parsed_url = urlparse(path)
        clean_path = parsed_url.path
    else:
        clean_path = path
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    if isinstance(video, list) and len(video) > 0:
        first_elem = video[0]
        if isinstance(first_elem, Image.Image):
            return True
        if isinstance(first_elem, str):
            return is_image_path(first_elem)
    return False


class Qwen35Embedder:
    """Embedder for Qwen3.5 model with sparse and dense embedding support."""

    # Supported pooling methods: pooling_string -> (pooling_type, is_sparse)
    _POOLING_METHODS = {
        'last.normal': ('dense', False),
        'splade.last': ('sparse', True),
        'splade.max': ('sparse', True),
    }

    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = "last.normal",
        normalize: bool = True,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        attn_type: Optional[str] = None,
        **kwargs
    ):
        self.pooling = pooling
        self.normalize = normalize
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction
        self.attn_type = attn_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model using AutoModel with trust_remote_code
        self.model = Qwen3_5ForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        ).to(self.device)
        self.model.eval()

        if self.attn_type == 'bi':
            self.model.enable_bidirectional_attention()

        # Load processor
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side='right', trust_remote_code=True
        )

        # Load sparse config and weights
        self._load_sparse_config(model_name_or_path)
        self.update_processor()

    def update_processor(self):
        self.tokenizer = self.processor.tokenizer

        if self.num_eos_tokens > 0:
            # eos_token = self.tokenizer.eos_token
            # eos_token = self.tokenizer.eos_token
            # eos_id = self.tokenizer.eos_token_id

            eos_token = '<|endoftext|>'
            eos_id = self.tokenizer.convert_tokens_to_ids(eos_token)

            eos_suffix = " " + eos_token
            eos_single = eos_suffix * self.num_eos_tokens
            eos_pair = eos_suffix * self.num_eos_tokens

            multi_eos_tokens = " ".join([eos_token] * self.num_eos_tokens)
            template = processors.TemplateProcessing(
                single=f"$A {multi_eos_tokens}",
                pair="$A" + eos_pair + " $B" + eos_pair,
                special_tokens=[(eos_token, eos_id)]
            )
            # self.original_post_processor = self.tokenizer.backend_tokenizer.post_processor
            self.processor.tokenizer._tokenizer.post_processor = template
        self.tokenizer.padding_side = "right"

    def _load_sparse_config(self, model_name_or_path: str):
        """加载 sparse 配置和权重。"""
        self.num_eos_tokens = 0  # default: disabled
        self.sparse_lm_heads = None
        self.sparse_bias = None
        print(f"Loading sparse info from {model_name_or_path}")
        sparse_info_path = os.path.join(model_name_or_path, "sparse_info.json")
        sparse_weights_path = os.path.join(model_name_or_path, "sparse_weights.pt")

        if os.path.exists(sparse_info_path) and os.path.exists(sparse_weights_path):
            try:
                with open(sparse_info_path, 'r', encoding='utf-8') as f:
                    sparse_info = json.load(f)
                self.num_eos_tokens = sparse_info.get("num_eos_tokens", 0)

                sparse_weights = torch.load(sparse_weights_path, map_location='cpu')
                self.sparse_lm_heads = torch.nn.ParameterList([
                    torch.nn.Parameter(head, requires_grad=False)
                    for head in sparse_weights["sparse_lm_heads"]
                ]).to(self.device)
                self.sparse_bias = torch.nn.ParameterList([
                    torch.nn.Parameter(bias, requires_grad=False)
                    for bias in sparse_weights["sparse_bias"]
                ]).to(self.device)
                logger.info(f"Loaded sparse config: num_eos_tokens={self.num_eos_tokens}")
            except Exception as e:
                logger.warning(f"Failed to load sparse config: {e}")
                self.num_eos_tokens = 0
        else:
            logger.info("No sparse config found, sparse embedding disabled")

    def _pooling_dense_last_normal(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Dense pooling: last.normal"""
        last_indices = (attention_mask.cumsum(dim=1) * attention_mask).argmax(dim=1)
        target_indices = last_indices - self.num_eos_tokens
        # target_indices = last_indices
        batch_size = hidden_state.shape[0]
        batch_indices = torch.arange(batch_size, device=hidden_state.device)
        return hidden_state[batch_indices, target_indices]

    def _pooling_sparse_splade_last(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Sparse pooling: splade.last"""
        if self.num_eos_tokens == 0:
            raise ValueError(
                "Sparse pooling 'splade.last' requires num_eos_tokens > 0, but got 0. "
                "Please ensure sparse_info.json and sparse_weights.pt exist and are valid."
            )
        if self.sparse_lm_heads is None:
            raise ValueError(
                "Sparse pooling 'splade.last' requires sparse_lm_heads, but not loaded. "
                "Please ensure sparse_weights.pt contains valid 'sparse_lm_heads' and 'sparse_bias'."
            )

        # Ensure sparse weights are on the same device as hidden_state
        device = hidden_state.device
        if self.sparse_lm_heads[0].device != device:
            self.sparse_lm_heads = self.sparse_lm_heads.to(device)
            self.sparse_bias = self.sparse_bias.to(device)

        last_indices = (attention_mask.cumsum(dim=1) * attention_mask).argmax(dim=1)
        batch_size = hidden_state.shape[0]
        batch_indices = torch.arange(batch_size, device=device)

        all_logits = []
        for i in range(self.num_eos_tokens):
            offset = (self.num_eos_tokens - 1) - i
            target_indices = last_indices - offset
            h_i = hidden_state[batch_indices, target_indices]
            logits_i = F.linear(h_i, self.sparse_lm_heads[i], self.sparse_bias[i])
            all_logits.append(logits_i)

        logits = torch.cat(all_logits, dim=-1)
        return torch.log1p(F.relu(logits))

    def _pooling_sparse_splade_max(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Sparse pooling: splade.max — max over all token positions."""
        if self.sparse_lm_heads is None:
            raise ValueError(
                "Sparse pooling 'splade.max' requires sparse_lm_heads, but not loaded. "
                "Please ensure sparse_weights.pt contains valid 'sparse_lm_heads' and 'sparse_bias'."
            )

        device = hidden_state.device
        if self.sparse_lm_heads[0].device != device:
            self.sparse_lm_heads = self.sparse_lm_heads.to(device)
            self.sparse_bias = self.sparse_bias.to(device)

        lm_head = self.sparse_lm_heads[0]
        bias = self.sparse_bias[0]

        logits = F.linear(hidden_state, lm_head, bias)
        weights = torch.log1p(F.relu(logits))

        weights = weights.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(),
            torch.finfo(weights.dtype).min
        )

        sparse_embeddings, _ = weights.max(dim=1)
        return sparse_embeddings

    def _do_pooling(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> tuple:
        """
        根据 pooling 字符串路由到具体的 pooling 方法。

        Returns:
            tuple: (embeddings, is_sparse)
        """
        if self.pooling not in self._POOLING_METHODS:
            raise ValueError(
                f"Unknown pooling method: '{self.pooling}'. "
                f"Supported methods: {list(self._POOLING_METHODS.keys())}"
            )

        pooling_type, is_sparse = self._POOLING_METHODS[self.pooling]
        method_name = f"_pooling_{pooling_type}_{self.pooling.replace('.', '_')}"

        if not hasattr(self, method_name):
            raise ValueError(f"Pooling method '{self.pooling}' is defined but implementation '{method_name}' not found.")

        pooling_fn = getattr(self, method_name)
        embeddings = pooling_fn(hidden_state, attention_mask)
        return embeddings, is_sparse

    def format_model_input(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Union[List, str]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
                instruction = instruction + '.'

        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content}
        ]

        texts = [text] if isinstance(text, str) else (text or [])
        images = [image] if image and not isinstance(image, list) else (image or [])
        videos = [video] if is_video_input(video) else (video or [])

        if not texts and not images and not videos:
            content.append({'type': 'text', 'text': "NULL"})
            return conversation

        for vid in videos:
            if not vid:
                continue
            if isinstance(vid, list):
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [('file://' + ele if isinstance(ele, str) else ele) for ele in video_content]
                video_kwargs = {'total_pixels': self.total_pixels}
            elif isinstance(vid, str):
                video_content = vid if vid.startswith(('http://', 'https://')) else 'file://' + vid
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")
            if video_content:
                content.append({'type': 'video', 'video': video_content, **video_kwargs})

        for img in images:
            if not img:
                continue
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(('http://', 'https://')) else 'file://' + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")
            if image_content:
                content.append({
                    'type': 'image', 'image': image_content,
                    "min_pixels": self.min_pixels, "max_pixels": self.max_pixels
                })

        for txt in texts:
            if txt:
                content.append({'type': 'text', 'text': txt})

        return conversation

    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True
            )
        except Exception as e:
            logger.error(f"{conversations[0]}")
            logger.error(f"Error in processing vision info: {e}")
            logger.error(conversations)
            images = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}],
                add_generation_prompt=True, tokenize=False
            )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=text, images=images, videos=videos, video_metadata=video_metadata,
            truncation=True, max_length=self.max_length, padding=True,
            do_resize=False, return_tensors='pt', **video_kwargs
        )
        return inputs

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'attention_mask': inputs.get('attention_mask')
        }

    def process(self, inputs: List[Dict[str, Any]], normalize: bool = None) -> torch.Tensor:
        conversations = [self.format_model_input(
            text=ele.get('text'),
            image=ele.get('image'),
            video=ele.get('video'),
            instruction=ele.get('instruction'),
            fps=ele.get('fps'),
            max_frames=ele.get('max_frames')
        ) for ele in inputs]

        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {k: v.to(self.device) for k, v in processed_inputs.items()}

        outputs = self.forward(processed_inputs)
        hidden_state = outputs['last_hidden_state']
        attention_mask = outputs['attention_mask']

        embeddings, is_sparse = self._do_pooling(hidden_state, attention_mask)

        # Normalize for dense embeddings
        if not is_sparse:
            if normalize is None:
                normalize = self.normalize
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
