import sys
import os

# Add core/ to Python path so relative imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Patch the relative imports in core files
from core import guardrail_implementation
from core import ml_input_guardrail
from core import ollama_client

import uvicorn

if __name__ == "__main__":
    uvicorn.run("core.api:app", host="0.0.0.0", port=8000, reload=False)