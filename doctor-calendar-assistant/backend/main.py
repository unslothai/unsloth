"""
Doctor Calendar Assistant - Main FastAPI Server
Handles WebSocket connections for real-time voice communication
"""

import asyncio
import json
import base64
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from voice.stt import SpeechToText
from voice.tts import TextToSpeech
from brain.llm import PersonaPlexBrain
from brain.tools import CalendarTools
from calendar.google_cal import GoogleCalendarClient


# Load configuration
def load_config():
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

# Global instances
stt: SpeechToText = None
tts: TextToSpeech = None
brain: PersonaPlexBrain = None
calendar_client: GoogleCalendarClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global stt, tts, brain, calendar_client

    print("🚀 Initializing Doctor Calendar Assistant...")

    # Initialize components
    print("  📝 Loading Whisper STT...")
    stt = SpeechToText(config["stt"])

    print("  🔊 Loading Piper TTS...")
    tts = TextToSpeech(config["tts"])

    print("  🧠 Connecting to PersonaPlex...")
    calendar_client = GoogleCalendarClient(config["calendar"])
    calendar_tools = CalendarTools(calendar_client)
    brain = PersonaPlexBrain(config["llm"], config["assistant"], calendar_tools)

    print("✅ All systems ready!")
    print(
        f"🌐 Server running at http://{config['server']['host']}:{config['server']['port']}"
    )

    yield

    # Cleanup
    print("👋 Shutting down...")


app = FastAPI(
    title = "Doctor Calendar Assistant",
    description = "AI-powered voice assistant for scheduling medical appointments",
    version = "1.0.0",
    lifespan = lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins = [config["server"]["frontend_url"], "http://localhost:3000"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"📞 Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"📴 Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)


manager = ConnectionManager()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "assistant": config["assistant"]["name"]}


@app.get("/api/appointments")
async def get_appointments():
    """Get upcoming appointments"""
    try:
        appointments = await calendar_client.get_upcoming_appointments(max_results = 10)
        return {"appointments": appointments}
    except Exception as e:
        return JSONResponse(status_code = 500, content = {"error": str(e)})


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for voice communication

    Message types:
    - audio_chunk: Base64 encoded audio data from client
    - start_call: Client initiates call
    - end_call: Client ends call

    Server sends:
    - transcript: User's transcribed speech
    - response: Assistant's text response
    - audio: Base64 encoded audio response
    - status: Connection/processing status
    """
    await manager.connect(websocket, client_id)

    # Audio buffer for accumulating chunks
    audio_buffer = bytearray()
    conversation_history = []
    is_call_active = False

    try:
        # Send initial greeting
        await manager.send_message(
            client_id,
            {"type": "status", "status": "connected", "message": "Conectado con Ana"},
        )

        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "start_call":
                is_call_active = True
                conversation_history = []
                audio_buffer = bytearray()

                # Send greeting
                greeting = (
                    "Hola, soy Ana, tu asistente médica. ¿En qué te puedo ayudar hoy?"
                )

                await manager.send_message(
                    client_id,
                    {"type": "response", "text": greeting, "role": "assistant"},
                )

                # Generate and send audio greeting
                audio_data = await tts.synthesize(greeting)
                await manager.send_message(
                    client_id,
                    {
                        "type": "audio",
                        "data": base64.b64encode(audio_data).decode("utf-8"),
                    },
                )

                conversation_history.append({"role": "assistant", "content": greeting})

            elif message_type == "end_call":
                is_call_active = False
                farewell = "Gracias por llamar. ¡Que te mejores!"

                await manager.send_message(
                    client_id,
                    {"type": "response", "text": farewell, "role": "assistant"},
                )

                audio_data = await tts.synthesize(farewell)
                await manager.send_message(
                    client_id,
                    {
                        "type": "audio",
                        "data": base64.b64encode(audio_data).decode("utf-8"),
                    },
                )

                await manager.send_message(
                    client_id, {"type": "status", "status": "call_ended"}
                )

            elif message_type == "audio_chunk" and is_call_active:
                # Decode and accumulate audio
                chunk = base64.b64decode(data.get("data", ""))
                audio_buffer.extend(chunk)

            elif message_type == "audio_complete" and is_call_active:
                # Process complete audio
                if len(audio_buffer) > 0:
                    await manager.send_message(
                        client_id,
                        {
                            "type": "status",
                            "status": "processing",
                            "message": "Procesando...",
                        },
                    )

                    # Transcribe audio
                    transcript = await stt.transcribe(bytes(audio_buffer))
                    audio_buffer = bytearray()

                    if transcript.strip():
                        await manager.send_message(
                            client_id,
                            {"type": "transcript", "text": transcript, "role": "user"},
                        )

                        conversation_history.append(
                            {"role": "user", "content": transcript}
                        )

                        # Get LLM response
                        await manager.send_message(
                            client_id,
                            {
                                "type": "status",
                                "status": "thinking",
                                "message": "Pensando...",
                            },
                        )

                        response = await brain.process(transcript, conversation_history)

                        await manager.send_message(
                            client_id,
                            {"type": "response", "text": response, "role": "assistant"},
                        )

                        conversation_history.append(
                            {"role": "assistant", "content": response}
                        )

                        # Generate and send audio
                        await manager.send_message(
                            client_id,
                            {
                                "type": "status",
                                "status": "speaking",
                                "message": "Hablando...",
                            },
                        )

                        audio_data = await tts.synthesize(response)
                        await manager.send_message(
                            client_id,
                            {
                                "type": "audio",
                                "data": base64.b64encode(audio_data).decode("utf-8"),
                            },
                        )

                    await manager.send_message(
                        client_id,
                        {
                            "type": "status",
                            "status": "listening",
                            "message": "Escuchando...",
                        },
                    )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        await manager.send_message(client_id, {"type": "error", "message": str(e)})
        manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host = config["server"]["host"],
        port = config["server"]["port"],
        reload = True,
    )
