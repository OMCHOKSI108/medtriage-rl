#!/bin/bash
set -e

echo "[STARTUP] Starting MedTriage environment server..."
python -m uvicorn env_server:app --host 0.0.0.0 --port 7860 &
ENV_PID=$!

echo "[STARTUP] Waiting for env server to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:7860/state > /dev/null 2>&1; then
        echo "[STARTUP] Server ready after ${i}s"
        break
    fi
    sleep 2
done

echo "[STARTUP] Running inference..."
python inference.py

echo "[STARTUP] Inference complete. Keeping server alive..."
wait $ENV_PID
