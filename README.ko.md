<h1 align="center" style="margin:0;">
  <a href="https://unsloth.ai/docs"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/STUDIO%20WHITE%20LOGO.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/STUDIO%20BLACK%20LOGO.png">
    <img alt="Unsloth 로고" src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/STUDIO%20BLACK%20LOGO.png" height="60" style="max-width:100%;">
  </picture></a>
</h1>
<h3 align="center" style="margin: 0; margin-top: 0;">
통합 로컬 인터페이스로 AI 모델을 실행하고 학습하세요.
</h3>

<p align="center">
  <a href="#-주요-기능">주요 기능</a> •
  <a href="#-빠른-시작">빠른 시작</a> •
  <a href="#-무료-노트북">노트북</a> •
  <a href="https://unsloth.ai/docs">문서</a> •
  <a href="https://discord.com/invite/unsloth">Discord</a>
</p>
 <a href="https://unsloth.ai/docs/new/studio">
<img alt="unsloth studio ui 홈페이지" src="https://raw.githubusercontent.com/unslothai/unsloth/main/studio/frontend/public/studio%20github%20landscape%20colab%20display.png" style="max-width: 100%; margin-bottom: 0;"></a>

Unsloth Studio를 사용하면 텍스트, [오디오](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning), [임베딩](https://unsloth.ai/docs/new/embedding-finetuning), [비전](https://unsloth.ai/docs/basics/vision-fine-tuning) 등 다양한 모달리티의 모델을 실행하고 학습할 수 있습니다. Windows, Linux, macOS에서 사용 가능합니다.

## ⭐ 주요 기능
Unsloth는 추론과 학습 모두를 위한 다양한 핵심 기능을 제공합니다:
### 추론
* **모델 검색 + 다운로드 + 실행** — GGUF, LoRA 어댑터, safetensors 등 지원
* **모델 내보내기**: GGUF, 16-bit safetensors 등 다양한 형식으로 모델 [저장 및 내보내기](https://unsloth.ai/docs/new/studio/export) 지원
* **도구 호출**: [자동 복구 도구 호출](https://unsloth.ai/docs/new/studio/chat#auto-healing-tool-calling) 및 웹 검색 지원
* **[코드 실행](https://unsloth.ai/docs/new/studio/chat#code-execution)**: LLM이 코드를 실행하고 결과를 검증하여 더 정확한 답변 제공
* [추론 파라미터 자동 튜닝](https://unsloth.ai/docs/new/studio/chat#auto-parameter-tuning) 및 채팅 템플릿 커스터마이징
* 이미지, 오디오, PDF, 코드, DOCX 등 다양한 파일을 업로드하여 대화 가능
### 학습
* **500개 이상의 모델**을 최대 **2배 빠르게**, 최대 **70% 적은 VRAM**으로, 정확도 손실 없이 학습
* 전체 파인튜닝, 사전학습, 4-bit, 16-bit, FP8 학습 지원
* **실시간 모니터링**: 학습 과정을 실시간으로 모니터링하고, 손실 및 GPU 사용량 추적, 그래프 커스터마이징
* **데이터 레시피**: **PDF, CSV, DOCX** 등에서 [데이터셋 자동 생성](https://unsloth.ai/docs/new/studio/data-recipe). 시각적 노드 워크플로우로 데이터 편집
* **강화학습**: GRPO, [FP8](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) 등에서 **80% 적은 VRAM**을 사용하는 가장 효율적인 [RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) 라이브러리
* [멀티 GPU](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) 학습 지원, 대규모 업그레이드 예정

## ⚡ 빠른 시작
Unsloth는 두 가지 방식으로 사용할 수 있습니다: 웹 UI인 **[Unsloth Studio](https://unsloth.ai/docs/new/studio/)** 또는 코드 기반 **Unsloth Core**. 각각 요구 사항이 다릅니다.

### Unsloth Studio (웹 UI)
Unsloth Studio는 **Windows, Linux, WSL**, **macOS**에서 작동합니다.

* **CPU:** **채팅 추론만** 지원
* **NVIDIA GPU:** RTX 30/40/50, Blackwell, DGX Spark, DGX Station 등에서 학습 가능
* **macOS:** 현재 채팅만 지원; **MLX 학습**이 곧 출시 예정
* **멀티 GPU:** 현재 사용 가능, 대규모 업그레이드 준비 중

#### Windows, macOS, Linux 또는 WSL:
```
git clone https://github.com/unslothai/unsloth.git
cd unsloth
pip install -e .
unsloth studio setup
unsloth studio -H 0.0.0.0 -p 8888
```
[Docker 이미지](https://hub.docker.com/r/unsloth/unsloth) ```unsloth/unsloth``` 컨테이너를 사용하세요. [Docker 가이드](https://unsloth.ai/docs/get-started/install/docker)를 참고하세요.

### Unsloth Core (코드 기반)
#### Windows, Linux, WSL
```bash
pip install unsloth
```
Windows의 경우, `pip install unsloth`는 PyTorch가 설치되어 있어야 작동합니다. [Windows 가이드](https://unsloth.ai/docs/get-started/install/windows-installation)를 참고하세요.
Unsloth Studio와 동일한 Docker 이미지를 사용할 수 있습니다.

#### AMD, Intel
RTX 50x, B200, 6000 GPU: `pip install unsloth`. 가이드: [Blackwell](https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth) 및 [DGX Spark](https://unsloth.ai/docs/blog/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth). <br>
**AMD** 및 **Intel** GPU에 Unsloth를 설치하려면 [AMD 가이드](https://unsloth.ai/docs/get-started/install/amd) 및 [Intel 가이드](https://unsloth.ai/docs/get-started/install/intel)를 참고하세요.

## ✨ 무료 노트북

무료 노트북으로 모델 학습을 시작하세요. [가이드](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)를 읽어보세요. 데이터셋을 추가하고, 실행한 뒤, 학습된 모델을 배포하세요.

| 모델 | 무료 노트북 | 성능 | 메모리 사용량 |
|-----------|---------|--------|----------|
| **Qwen3.5 (4B)**      | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb)               | 1.5배 빠름 | 60% 절감 |
| **gpt-oss (20B)**      | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)               | 2배 빠름 | 70% 절감 |
| **gpt-oss (20B): GRPO**      | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)               | 2배 빠름 | 80% 절감 |
| **Qwen3: 고급 GRPO**      | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)               | 2배 빠름 | 50% 절감 |
| **Gemma 3 (4B) Vision** | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B)-Vision.ipynb)               | 1.7배 빠름 | 60% 절감 |
| **embeddinggemma (300M)**    | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma_(300M).ipynb)               | 2배 빠름 | 20% 절감 |
| **Mistral Ministral 3 (3B)**      | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_VL_(3B)_Vision.ipynb)               | 1.5배 빠름 | 60% 절감 |
| **Llama 3.1 (8B) Alpaca**      | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)               | 2배 빠름 | 70% 절감 |
| **Llama 3.2 대화형**      | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)               | 2배 빠름 | 70% 절감 |
| **Orpheus-TTS (3B)**     | [▶️ 무료로 시작하기](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)               | 1.5배 빠름 | 50% 절감 |

- 전체 노트북 보기: [Kaggle](https://github.com/unslothai/notebooks?tab=readme-ov-file#-kaggle-notebooks), [GRPO](https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks), [TTS](https://unsloth.ai/docs/get-started/unsloth-notebooks#text-to-speech-tts-notebooks), [임베딩](https://unsloth.ai/docs/new/embedding-finetuning) & [비전](https://unsloth.ai/docs/get-started/unsloth-notebooks#vision-multimodal-notebooks)
- [전체 모델 목록](https://unsloth.ai/docs/get-started/unsloth-model-catalog) 및 [전체 노트북 목록](https://unsloth.ai/docs/get-started/unsloth-notebooks) 보기
- Unsloth 공식 문서는 [여기](https://unsloth.ai/docs)에서 확인하세요

## 🦥 Unsloth 소식
- **Unsloth Studio 소개**: LLM 실행 및 학습을 위한 새로운 웹 UI. [블로그](https://unsloth.ai/docs/new/studio)
- **Qwen3.5** — 0.8B, 2B, 4B, 9B, 27B, 35-A3B, 112B-A10B 지원. [가이드 + 노트북](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- **MoE LLM 12배 빠른 학습**, 35% 적은 VRAM — DeepSeek, GLM, Qwen, gpt-oss. [블로그](https://unsloth.ai/docs/new/faster-moe)
- **임베딩 모델**: ~1.8-3.3배 빠른 임베딩 파인튜닝 지원. [블로그](https://unsloth.ai/docs/new/embedding-finetuning) • [노트북](https://unsloth.ai/docs/get-started/unsloth-notebooks#embedding-models)
- 새로운 배칭 알고리즘으로 다른 설정 대비 **7배 긴 컨텍스트 RL**. [블로그](https://unsloth.ai/docs/new/grpo-long-context)
- 새로운 RoPE & MLP **Triton 커널** & **패딩 프리 + 패킹**: 3배 빠른 학습 & 30% 적은 VRAM. [블로그](https://unsloth.ai/docs/new/3x-faster-training-packing)
- **500K 컨텍스트**: 80GB GPU에서 20B 모델의 500K 이상 컨텍스트 학습 가능. [블로그](https://unsloth.ai/docs/blog/500k-context-length-fine-tuning)
- **FP8 & 비전 RL**: 일반 GPU에서 FP8 및 VLM GRPO 가능. [FP8 블로그](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning) • [비전 RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl)
- **gpt-oss** by OpenAI: [RL 블로그](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning), [Flex Attention](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training) 블로그 및 [가이드](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune)

## 🔗 링크 및 리소스
| 유형                                                                                                                                      | 링크                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| <img width="15" src="https://redditinc.com/hs-fs/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" />  **r/unsloth Reddit**                       | [Reddit 커뮤니티 참여](https://reddit.com/r/unsloth)                          |
| 📚 **문서 & 위키**                                                                                                                        | [문서 보기](https://unsloth.ai/docs)                                           |
| <img width="13" src="https://upload.wikimedia.org/wikipedia/commons/0/09/X_(formerly_Twitter)_logo_late_2025.svg" />  **Twitter (aka X)**      | [X에서 팔로우](https://twitter.com/unslothai)                                  |
| 💾 **설치**                                                                                                                                | [Pip & Docker 설치](https://unsloth.ai/docs/get-started/install)               |
| 🔮 **모델 목록**                                                                                                                          | [Unsloth 카탈로그](https://unsloth.ai/docs/get-started/unsloth-model-catalog)  |
| ✍️ **블로그**                                                                                                                              | [블로그 읽기](https://unsloth.ai/blog)                                         |

### 인용

Unsloth 저장소를 다음과 같이 인용할 수 있습니다:
```bibtex
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {https://github.com/unslothai/unsloth},
  year = {2023}
}
```
🦥 Unsloth로 모델을 학습했다면, 아래 스티커를 자유롭게 사용해 주세요!   <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made with unsloth.png" width="200" align="center" />

### 감사의 말
- Unsloth에서 사용자가 모델을 실행하고 저장할 수 있게 해주는 [llama.cpp 라이브러리](https://github.com/ggml-org/llama.cpp)
- Hugging Face 팀과 [transformers](https://github.com/huggingface/transformers) 및 [TRL](https://github.com/huggingface/trl) 라이브러리
- PyTorch 및 [Torch AO](https://github.com/unslothai/unsloth/pull/3391) 팀의 기여에 감사드립니다
- 그리고 물론, Unsloth에 기여하거나 사용해주신 모든 분들께 감사드립니다!
