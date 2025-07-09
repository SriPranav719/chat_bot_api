import httpx
import traceback
import json

OLLAMA_MODEL = "qwen:1.8b"


SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a Python expert assistant. "
        "Only answer questions related to Python programming, syntax, libraries, or ecosystem. "
        "For any unrelated question, always reply exactly with: "
        "'Sorry, I can only help with Python-related topics.'"
    )
}


async def stream_ollama_response(chat_history):
    """
    Async generator that yields chunks of streamed response content from Ollama.
    """
    url = "http://127.0.0.1:11434/api/chat"

    # Prepare messages for Ollama
    formatted_history = [SYSTEM_PROMPT]
    for msg in chat_history:
        role = msg["role"]
        if role == "model":
            role = "assistant"
        formatted_history.append({
            "role": role,
            "content": msg["text"]
        })

    payload = {
        "model": OLLAMA_MODEL,
        "messages": formatted_history,
        "stream": True
    }

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                # Ollama streams newline-delimited JSON objects
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        error_details = traceback.format_exc()
        print("Error streaming from Ollama:", error_details)
        yield f"\n[ERROR] Sorry, something went wrong while streaming from Ollama: {str(e)}"
