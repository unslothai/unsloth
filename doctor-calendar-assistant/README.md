# Ana - Asistente Médica Virtual

Un asistente de voz con IA para agendar citas médicas, optimizado para Mac Mini M4 con 16GB de RAM.

![Ana Assistant](https://img.shields.io/badge/Language-Spanish%20(MX)-green)
![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Características

- **Interfaz de voz natural** - Conversación fluida en español mexicano
- **Agendamiento inteligente** - Integración con Google Calendar
- **100% Local** - Tu privacidad está protegida, todo corre en tu Mac
- **UI moderna** - Inspirada en ElevenLabs con Orb y Waveform animados
- **Bajo consumo** - Diseñado para no afectar tu trabajo diario

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    Tu Mac Mini M4                           │
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│   │   Whisper   │    │ PersonaPlex │    │  Piper TTS  │   │
│   │    (STT)    │───▶│     LLM     │───▶│  (Español)  │   │
│   └─────────────┘    └──────┬──────┘    └─────────────┘   │
│                             │                              │
│                      ┌──────▼──────┐                       │
│                      │   Google    │                       │
│                      │  Calendar   │                       │
│                      └─────────────┘                       │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Next.js + ElevenLabs UI                │  │
│   │         (Interfaz web en localhost:3000)            │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Requisitos

- **macOS** con Apple Silicon (M1/M2/M4)
- **16GB RAM** mínimo
- **Python 3.11+**
- **Node.js 18+**
- **Cuenta de Google** (para Calendar)

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repo-url>
cd doctor-calendar-assistant
```

### 2. Ejecutar el script de instalación

```bash
./scripts/setup.sh
```

Este script:
- Instala Homebrew (si no está instalado)
- Instala Ollama y descarga el modelo LLM
- Configura Piper TTS con voz en español mexicano
- Instala dependencias de Python y Node.js

### 3. Configurar Google Calendar

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto
3. Habilita la **Google Calendar API**
4. Crea credenciales OAuth 2.0:
   - Tipo de aplicación: **Aplicación de escritorio**
5. Descarga el archivo JSON de credenciales
6. Guárdalo como `config/google_credentials.json`

### 4. Iniciar la aplicación

```bash
./scripts/start.sh
```

Abre tu navegador en: **http://localhost:3000**

## Uso

1. **Presiona "Llamar a Ana"** para iniciar la conversación
2. **Habla naturalmente** - Ana entiende español mexicano
3. **Pide lo que necesites**:
   - "Necesito una cita para el viernes"
   - "¿Qué horarios hay disponibles mañana?"
   - "Cancela mi cita del lunes"
   - "¿Cuáles son mis próximas citas?"
4. **Presiona el botón rojo** para terminar la llamada

## Ejemplo de Conversación

```
Tú: "Hola Ana, necesito una cita con el doctor"

Ana: "Hola, con gusto te ayudo. ¿Para qué día te gustaría
      la cita?"

Tú: "El viernes si se puede"

Ana: "Perfecto. El viernes tengo disponible a las 10 de
      la mañana, 2 de la tarde y 4 de la tarde.
      ¿Cuál te queda mejor?"

Tú: "A las 2 está bien"

Ana: "Listo, agendé tu cita para el viernes a las 2 de
      la tarde. ¿Necesitas algo más?"
```

## Estructura del Proyecto

```
doctor-calendar-assistant/
├── backend/                 # Python FastAPI server
│   ├── main.py             # Entry point
│   ├── voice/              # STT (Whisper) y TTS (Piper)
│   ├── brain/              # LLM y lógica de herramientas
│   └── calendar/           # Integración Google Calendar
│
├── frontend/               # Next.js + React
│   ├── app/               # App Router pages
│   ├── components/        # UI components (Orb, Waveform)
│   └── lib/               # Utilities (WebSocket, Audio)
│
├── config/                # Configuración
│   ├── settings.yaml      # Configuración general
│   └── google_credentials.json  # (creado por ti)
│
├── scripts/               # Scripts de utilidad
│   ├── setup.sh          # Instalación
│   ├── start.sh          # Iniciar todo
│   └── stop.sh           # Detener todo
│
└── models/               # Modelos descargados
    └── piper/            # Voces TTS
```

## Rendimiento Esperado

| Métrica | Valor |
|---------|-------|
| Tiempo de respuesta total | 4-7 segundos |
| Uso de RAM durante llamada | ~8-10 GB |
| Uso de RAM en reposo | ~5 GB |
| Impacto en trabajo diario | Mínimo |

## Personalización

### Cambiar el nombre del asistente

Edita `backend/brain/prompts.py`:

```python
SYSTEM_PROMPT = """Eres María, una asistente médica..."""
```

### Cambiar la zona horaria

Edita `config/settings.yaml`:

```yaml
calendar:
  timezone: "America/Mexico_City"  # Cambiar aquí
```

### Usar un modelo diferente

Edita `config/settings.yaml`:

```yaml
llm:
  model: "llama3.1:8b"  # O cualquier modelo de Ollama
```

## Solución de Problemas

### "Ollama no está corriendo"

```bash
ollama serve
```

### "No se puede conectar al calendario"

1. Verifica que `config/google_credentials.json` existe
2. Elimina `config/google_token.json` y vuelve a autorizar

### "El audio no funciona"

1. Verifica permisos de micrófono en Preferencias del Sistema
2. Asegúrate de usar Chrome o Safari (Firefox puede tener problemas)

### "Respuestas muy lentas"

El modelo puede tardar más la primera vez. Considera:
- Usar un modelo más pequeño: `ollama pull qwen2:1.5b`
- Cerrar aplicaciones que consuman mucha RAM

## Tecnologías

- **LLM**: PersonaPlex-7B / Qwen2-7B via Ollama
- **STT**: Whisper (faster-whisper)
- **TTS**: Piper con voz mexicana
- **Backend**: Python FastAPI + WebSockets
- **Frontend**: Next.js 14 + React + Tailwind
- **UI Components**: Inspirados en ElevenLabs UI

## Licencia

MIT License - Usa este proyecto como quieras.

## Contribuir

¡Las contribuciones son bienvenidas! Abre un issue o PR.

---

Hecho con ❤️ para hacer la vida más fácil al agendar citas médicas.
