#!/usr/bin/env bash
# Run a command in the ecotyper-py container with the project mounted at /work.
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba \
  -e ECOTYPER_NO_FMA -e ECOTYPER_LD -e ECOTYPER_R_REF -e ECOTYPER_R_WORK -e ECOTYPER_COLLATION \
  -v "$HOME/workspace/ecotyper-py:/work" -w /work ecotyper-py:dev \
  bash -c "pip install -q --no-deps --no-build-isolation -e . --target /tmp/pkgs >/dev/null 2>&1; PYTHONPATH=/work/src \$0 \"\$@\"" "$@"
