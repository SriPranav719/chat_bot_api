from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from chat_utils import stream_ollama_response


app = FastAPI()

chat_history = []

PYTHON_KEYWORDS = [
    "python", "django", "flask", "pandas", "numpy", "fastapi",
    "sqlalchemy", "asyncio", "py", "jupyter", "pytest", "venv", "pip",
    "tkinter", "pyqt", "matplotlib", "sklearn", "machine learning",
    "dataframe", "pyproject", "pydantic", "typing", "decorator", "class",
    "function", "def", "import", "list", "dict", "dictionary", "tuple",
    "set", "loop", "for", "while", "if", "elif", "else", "try", "except",
    "exception", "error", "traceback"
]


def is_python_question(message: str) -> bool:
    lower_msg = message.lower()
    return any(word in lower_msg for word in PYTHON_KEYWORDS)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    user_message = payload.message.strip()

    if not user_message:
        return JSONResponse(
            content={"reply": "Please enter a valid message."},
            status_code=400
        )

    # Block non-Python questions early
    if not is_python_question(user_message):
        return JSONResponse(content={
            "reply": "Sorry, I can only help with Python-related topics."
        })

    # Save user message
    chat_history.append({"role": "user", "text": user_message})

    async def event_generator():
        # stream Ollama's response chunks
        collected_text = ""
        async for chunk in stream_ollama_response(chat_history):
            collected_text += chunk
            yield chunk

        # Save model reply at the end
        chat_history.append({"role": "model", "text": collected_text})

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.get("/history")
def get_history():
    return JSONResponse(content={"history": chat_history})
