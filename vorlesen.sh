#!/bin/sh
# Launcher: runs the core pipeline in its own venv from anywhere.
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" && exec "$ROOT/envs/core/bin/python" -m vorlesen "$@"
