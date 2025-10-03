# FastAPI Prototype

Minimal FastAPI app that displays a dynamic H1 heading.

## Quickstart (Windows PowerShell)

1. Create or activate a virtual environment (recommended):

```powershell
# In the project folder
# If a venv folder already exists (as in this project), activate it:
. .\venv\Scripts\Activate.ps1

# Or, create a new one named .venv and activate it:
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the server (auto-reload on change):

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open your browser:
- Home: http://127.0.0.1:8000/
- Dynamic H1: http://127.0.0.1:8000/hello/Copilot

## Edit
- Change the dynamic content by editing `app/main.py` in the `/hello/{title}` route.
- The server reloads automatically when files change.

## Stop
- Press `Ctrl+C` in the terminal.

---
This is a prototype; no database or templates are used. For templating later, consider Jinja2 with FastAPI's `Jinja2Templates`.