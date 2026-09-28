#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying .env.example to .env..."
    cp .env.example .env
    echo "👉 Please edit .env with your LINE and Gemini API keys!"
fi

echo "🚀 Starting LINE Gemini Chatbot on http://localhost:8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
