#!/bin/sh
set -e

uvicorn Api.main:app --host 127.0.0.1 --port 8000 &

# wait for the API so the dashboard's first health check succeeds
for i in $(seq 1 60); do
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null && break
  sleep 1
done

exec streamlit run app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableXsrfProtection false
