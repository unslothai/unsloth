#!/bin/bash

# Doctor Calendar Assistant - Stop Script

echo "🛑 Stopping Doctor Calendar Assistant..."

# Kill backend
pkill -f "python main.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true

# Kill frontend
pkill -f "next-server" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

echo "✅ All services stopped"
