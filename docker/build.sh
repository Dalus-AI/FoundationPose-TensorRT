#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t foundationpose-tensorrt:latest \
    "$REPO_ROOT"
