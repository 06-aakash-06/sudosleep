#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -f web/dist/index.html ]; then
  npm ci --prefix web
  npm run build --prefix web
fi
if [ -x venv/bin/python ]; then
  exec venv/bin/python -m uvicorn research_api:app --host 127.0.0.1 --port 8000
fi
exec .venv/bin/python -m uvicorn research_api:app --host 127.0.0.1 --port 8000
