#!/bin/bash

# Doctor Calendar Assistant - Setup Script
# For Mac Mini M4 (Apple Silicon)

set -e

echo "🏥 Doctor Calendar Assistant - Setup"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Apple Silicon (M1/M2/M4)${NC}"
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p models/piper
mkdir -p config

# 1. Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 2. Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
brew install python@3.11 node ffmpeg || true

# 3. Install Ollama
echo ""
echo "🦙 Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    brew install ollama
fi

# Start Ollama service
echo "🚀 Starting Ollama service..."
ollama serve &> /dev/null &
sleep 3

# Pull PersonaPlex model (or alternative)
echo "📥 Pulling LLM model..."
echo "Note: nvidia/personaplex-7b-v1 may need to be imported manually."
echo "Pulling qwen2:7b as a reliable alternative with Spanish support..."
ollama pull qwen2:7b || true

# Try to create a custom model from PersonaPlex if available
echo ""
echo "Creating PersonaPlex model alias..."
cat > /tmp/Modelfile << 'EOF'
FROM qwen2:7b
SYSTEM """Eres Ana, una asistente médica virtual profesional y amigable que ayuda a los pacientes a agendar citas médicas. Hablas español mexicano de manera natural."""
EOF
ollama create personaplex -f /tmp/Modelfile || true

# 4. Install Piper TTS
echo ""
echo "🔊 Setting up Piper TTS..."
PIPER_DIR="$HOME/.local/piper"
mkdir -p "$PIPER_DIR"

# Download Piper for macOS ARM64
if [[ ! -f "$PIPER_DIR/piper" ]]; then
    echo "Downloading Piper..."
    curl -L "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_macos_aarch64.tar.gz" -o /tmp/piper.tar.gz
    tar -xzf /tmp/piper.tar.gz -C "$PIPER_DIR"
    rm /tmp/piper.tar.gz
fi

# Download Mexican Spanish voice
VOICE_DIR="models/piper"
if [[ ! -f "$VOICE_DIR/es_MX-claude-medium.onnx" ]]; then
    echo "Downloading Mexican Spanish voice..."
    # Note: Using a placeholder URL - actual voice files need to be downloaded from Piper releases
    curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/medium/es_MX-claude-medium.onnx" -o "$VOICE_DIR/es_MX-claude-medium.onnx" || true
    curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/medium/es_MX-claude-medium.onnx.json" -o "$VOICE_DIR/es_MX-claude-medium.onnx.json" || true
fi

# Add Piper to PATH
echo 'export PATH="$HOME/.local/piper:$PATH"' >> ~/.zshrc || true

# 5. Set up Python environment
echo ""
echo "🐍 Setting up Python environment..."
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install faster-whisper for optimized STT
pip install faster-whisper

cd ..

# 6. Set up Node.js frontend
echo ""
echo "⚛️  Setting up Next.js frontend..."
cd frontend
npm install
cd ..

# 7. Google Calendar setup instructions
echo ""
echo -e "${YELLOW}📅 Google Calendar Setup Required${NC}"
echo "=================================="
echo ""
echo "To enable Google Calendar integration:"
echo ""
echo "1. Go to https://console.cloud.google.com/"
echo "2. Create a new project or select existing one"
echo "3. Enable the Google Calendar API"
echo "4. Create OAuth 2.0 credentials (Desktop app type)"
echo "5. Download the credentials JSON file"
echo "6. Save it as: config/google_credentials.json"
echo ""
echo "The first time you run the app, it will open a browser"
echo "window to authorize access to your calendar."
echo ""

# 8. Create example credentials file
if [[ ! -f "config/google_credentials.json" ]]; then
    cat > config/google_credentials.json.example << 'EOF'
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}
EOF
    echo "Created config/google_credentials.json.example"
    echo "Rename it to google_credentials.json after adding your credentials."
fi

# Done!
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "To start the application, run:"
echo "  ./scripts/start.sh"
echo ""
echo "Or start components manually:"
echo "  Backend:  cd backend && source venv/bin/activate && python main.py"
echo "  Frontend: cd frontend && npm run dev"
echo ""
