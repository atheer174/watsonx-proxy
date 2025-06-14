from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import os

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


IBM_API_KEY = os.environ["IBM_API_KEY"]
PROJECT_ID = os.environ["PROJECT_ID"]
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/llama-2-70b-chat")

async def call_ibm_watsonx(prompt):
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation"
    headers = {
        "Authorization": f"Bearer {IBM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 200,
            "min_new_tokens": 1,
        },
        "input": prompt,
    }
    async with httpx.AsyncClient() as client:
        res = await client.post(url, json=payload, headers=headers)
        return res.json()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    try:
        ibm_response = await call_ibm_watsonx(prompt)
        generated_text = ibm_response["results"][0]["generated_text"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {
        "id": "chatcmpl-ibm",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "stop"
        }],
        "model": body.get("model", MODEL_ID)
    }
