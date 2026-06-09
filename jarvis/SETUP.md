# JARVIS — Setup & Usage Guide

## 1. Install dependencies

```bash
cd jarvis
pip install -r requirements.txt
```

For trading (Windows only for MT5):
```bash
pip install MetaTrader5 pandas-ta
```

For video assembly:
```bash
pip install moviepy Pillow
```

For YouTube upload:
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

---

## 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required for core functionality:
- `ANTHROPIC_API_KEY` — get from console.anthropic.com

Optional:
- `ELEVENLABS_API_KEY` — for high-quality voice output
- `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` — for trading
- YouTube OAuth credentials → place at `jarvis/data/youtube_credentials.json`

---

## 3. Launch the server

```bash
python -m jarvis.main server
```

Open **http://localhost:8000** for the full dashboard.

---

## 4. CLI usage

```bash
# Interactive chat
python -m jarvis.main chat

# Run a task autonomously
python -m jarvis.main task "Research the top 5 AI stocks and write a report"

# Voice mode
python -m jarvis.main voice
```

---

## 5. YouTube automation

Ask JARVIS in Task mode:
> "Create a faceless YouTube video about '10 ways AI will change medicine' for the health niche and save everything to the workspace"

Or use the pipeline tool directly:
```json
{ "tool": "yt_run_pipeline", "topic": "...", "niche": "...", "auto_upload": false }
```

To enable uploads: place your Google OAuth credentials at `jarvis/data/youtube_credentials.json`.

---

## 6. MT5 Trading

1. Install MetaTrader 5 on Windows
2. Set env vars: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`
3. Ask JARVIS:
   > "Connect to MT5 and scan EURUSD, XAUUSD, GBPUSD on H1 for opportunities"

   > "Scan my watchlist and execute any high-confluence setups with max 1.5% risk"

   > "What are my open positions and total P&L?"

---

## 7. API endpoints

| Method | Path              | Description                    |
|--------|-------------------|--------------------------------|
| POST   | /chat             | Single-turn chat               |
| POST   | /task             | Autonomous multi-step task     |
| WS     | /ws/{session_id}  | Streaming WebSocket            |
| GET    | /tasks            | List task history              |
| POST   | /memory/search    | Full-text memory search        |
| POST   | /voice/speak      | TTS → audio bytes              |
| POST   | /voice/transcribe | STT → text                     |
| GET    | /tools            | List all registered tools      |
| GET    | /logs             | Event log                      |
| GET    | /docs             | Swagger UI                     |

---

## 8. Adding custom tools (Plugin system)

```python
from jarvis.plugins.base_plugin import BasePlugin
from jarvis.core.tools import ToolDef

class MyPlugin(BasePlugin):
    name = "my_plugin"

    def tools(self):
        return [
            ToolDef(
                name="my_tool",
                description="Does something useful",
                input_schema={...},
                fn=my_async_function,
            )
        ]

# Register in main.py or server.py:
MyPlugin().register(tools)
```
