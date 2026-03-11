#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-3000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
ENGINE="${ENGINE:-sled}"

echo "Markdotabase starting..."
echo "  API:        http://${HOST}:${PORT}"
echo "  Swagger UI: http://${HOST}:${PORT}/swagger-ui/"
echo "  Engine:     ${ENGINE}"
echo "  Log level:  ${LOG_LEVEL}"
echo ""

exec cargo run -- --log-level "${LOG_LEVEL}" --engine "${ENGINE}" api --bind "${HOST}:${PORT}"
