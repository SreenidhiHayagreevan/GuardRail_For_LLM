#  **GateKeeper — GuardRail_For_LLM Setup Guide**

---

### STEP 1 — Clone the Repo (Main Branch)
```powershell
git clone https://github.com/SreenidhiHayagreevan/GuardRail_For_LLM.git
```

### STEP 2 — Create Virtual Environment
```powershell
py -3.12 -m venv venv
```

### STEP 3 — Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
>  You should see `(venv)` at the start of your terminal line


### STEP 4 — Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### STEP 5 — Install Python Dependencies
```powershell
pip install fastapi uvicorn sentence-transformers torch joblib scikit-learn numpy requests boto3 pyarrow pandas pydantic
```

### STEP 6 — Install Frontend Dependencies
```powershell
npm install
npx update-browserslist-db@latest
```

### STEP 7 — Pull Ollama Model
```powershell
ollama pull qwen2.5
```
**TO RUN THE PROJECT:**

### TERMINAL 1 — Run Ollama
```powershell
ollama serve
```
> If Ollama is already running, skip this step ( to check if ollama is already running: use 'ollama list' command.

### TERMINAL 2 — Run Backend
```powershell
.\venv\Scripts\Activate.ps1
py -3.12 run_api.py
```
> Backend runs at: http://localhost:8000

### TERMINAL 3 — Run Frontend
```powershell
cd Frontend
npm run dev
```
>  Frontend runs at: http://localhost:5173

---

