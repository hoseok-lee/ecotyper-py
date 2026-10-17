#!/usr/bin/env bash
# Run a command in the ecotyper-r container with the project mounted at /work.
exec docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e ECOTYPER_R_THREADS="${ECOTYPER_R_THREADS:-}" \
  -v "$HOME/workspace/ecotyper-py:/work" -w /work ecotyper-r:dev "$@"
