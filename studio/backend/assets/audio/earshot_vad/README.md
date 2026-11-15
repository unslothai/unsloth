# Earshot VAD weights

`earshot_vad.safetensors` is the voice activity detector voice mode uses by default: the network
weights and front-end tables of [Earshot](https://github.com/pykeio/earshot) 1.2.2, as packaged by
[audio.cpp](https://github.com/0xShug0/audio.cpp) v0.9.4 (`assets/framework/models/earshot_vad`),
unchanged. The audio runtime runs it on the CPU, from release v0.9.3 on.

Earshot is dual-licensed under MIT OR Apache-2.0; the MIT notice is in `LICENSE-MIT`.
