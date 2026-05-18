FROM unsloth_studio_base_test_img
# We mock 'uname' to return Darwin
USER root
RUN mkdir -p /usr/local/bin/mock_bin
RUN echo '#!/bin/bash\nif [[ "$*" == "s" ]]; then echo "Darwin"; else /bin/uname "$@"; fi' \
    > /usr/local/bin/uname
RUN chmod +x /usr/local/bin/uname
USER tester
