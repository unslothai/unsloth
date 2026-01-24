#!/bin/bash

# Doctor Calendar Assistant - Start Script

set -e

echo "🏥 Starting Doctor Calendar Assistant..."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if Ollama is running
echo "🦙 Checking Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &> /dev/null &
    sleep 3
fi

# Verify model is available
if ! ollama list | grep -q "personaplex\|qwen2"; then
    echo -e "${YELLOW}Warning: LLM model not found. Pulling qwen2:7b...${NC}"
    ollama pull qwen2:7b
fi

# Start backend
echo ""
echo "🐍 Starting Python backend..."
cd backend
source venv/bin/activate
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo ""
echo "⚛️  Starting Next.js frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo ""
echo -e "${GREEN}✅ Doctor Calendar Assistant is running!${NC}"
echo ""
echo "📱 Open in your browser: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait
