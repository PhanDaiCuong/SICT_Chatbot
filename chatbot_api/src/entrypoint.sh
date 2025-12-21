#!/usr/bin/env bash
set -e

echo "Start run chatbot-rag-api service ...."

exec python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8080
