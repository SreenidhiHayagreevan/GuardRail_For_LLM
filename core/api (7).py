from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from guardrail_implementation import GuardrailConfig, GuardrailSystem

app = FastAPI()

# Allow CORS from Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GuardrailRequest(BaseModel):
    prompt: str
    model: str = None  # optional model key (e.g., "qwen2.5", "phi3")

@app.post('/api/guardrail')
def guardrail_endpoint(req: GuardrailRequest):
    try:
        cfg = GuardrailConfig()
        # If a model was provided in the request, override the default
        if req.model:
            cfg.ollama_model_name = req.model

        system = GuardrailSystem(cfg)
        result = system.generate_with_guardrails(req.prompt)
        # Include the model used in the response metadata for transparency
        result.setdefault("metadata", {})["ollama_model_name"] = cfg.ollama_model_name
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
