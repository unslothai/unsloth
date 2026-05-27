FROM unsloth_studio_base_test_img
# We mock the presence of Microsoft in /proc/version
# by overriding the 'grep' command used in the script
USER root
RUN mkdir -p /usr/local/bin/mock_bin
RUN echo '#!/bin/bash\nif [[ "$*" == *"/proc/version"* ]]; then echo "Linux microsoft-production #1 ..."; else /bin/grep "$@"; fi' \
    > /usr/local/bin/grep
RUN chmod +x /usr/local/bin/grep
USER tester
