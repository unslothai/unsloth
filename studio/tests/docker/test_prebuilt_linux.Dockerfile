# tests/docker/test_prebuilt_linux.Dockerfile
FROM unsloth_studio_base_test_img

# Simulate an existing llama.cpp installation
RUN mkdir -p /home/tester/llama.cpp/build/bin
RUN touch /home/tester/llama.cpp/build/bin/llama-server
RUN echo '{"tag": "latest", "published_repo": "ggml-org/llama.cpp"}' \
    > /home/tester/llama.cpp/UNSLOTH_PREBUILT_INFO.json

USER tester
# Set the working directory to the repo root
WORKDIR /home/tester/
