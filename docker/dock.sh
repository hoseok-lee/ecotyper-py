#!/usr/bin/env bash
# Run a command in the ecotyper-py container with the project mounted at /work.
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba \
  -v "$HOME/workspace/ecotyper-py:/work" -w /work ecotyper-py:dev \
  bash -c "pip install -q --no-deps --no-build-isolation -e . --target /tmp/pkgs >/dev/null 2>&1; PYTHONPATH=/work/src \$0 \"\$@\"" "$@"
