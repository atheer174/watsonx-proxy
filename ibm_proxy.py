from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import asyncio
import time
import traceback
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IBM_API_KEY = os.environ.get("IBM_API_KEY", "")
PROJECT_ID = os.environ.get("PROJECT_ID", "")
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/llama-3-3-70b-instruct")

async def call_ibm_watsonx(prompt):
    try:
        iam_url = "https://iam.cloud.ibm.com/identity/token"
        token_headers = {"Content-Type": "application/x-www-form-urlencoded"}
        token_data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": IBM_API_KEY
        }

        async with httpx.AsyncClient() as client:
            token_res = await client.post(iam_url, data=token_data, headers=token_headers)
            iam_token = token_res.json()["access_token"]

            url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-05-14"
            headers = {
                "Authorization": f"Bearer {iam_token}",
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

            res = await client.post(url, json=payload, headers=headers)
            return res.json()
    except Exception as e:
        return {"error": str(e)}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        stream = body.get("stream", False)
        messages = body.get("messages", [])
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        if not stream:
            result = await call_ibm_watsonx(prompt)
            if "results" not in result:
                return JSONResponse(status_code=500, content={"error": result})
            full_text = result["results"][0].get("generated_text", "")
            return {
                "id": "chatcmpl-ibm",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25
                }
            }

        # ---- Streaming path ----
        async def event_stream():
            yield {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [{"delta": {"role": "assistant"}}]
            }

            yield {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "Thinking..."}}]
            }

            result = await call_ibm_watsonx(prompt)
            if "results" not in result:
                yield {
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "[Error from Watsonx]"}}]
                }
                yield {
                    "object": "chat.completion.chunk",
                    "choices": [{"finish_reason": "stop"}]
                }
                return

            full_text = result["results"][0].get("generated_text", "")
            for word in full_text.split():
                await asyncio.sleep(0.04)
                yield {
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": word + " "}}]
                }

            yield {
                "object": "chat.completion.chunk",
                "choices": [{"finish_reason": "stop"}]
            }

        return EventSourceResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        error_trace = traceback.format_exc()
        print("EXCEPTION:", error_trace)
        return JSONResponse(status_code=500, content={"error": str(e), "trace": error_trace})
