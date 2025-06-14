from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import time
import asyncio
import traceback
import json

app = FastAPI()

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
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    token_headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
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

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        stream = body.pop("stream", False)

        messages = body.get("messages", [])
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        if not stream:
            ibm_response = await call_ibm_watsonx(prompt)
            full_text = ibm_response["results"][0].get("generated_text", "").strip()
            return {
                "id": "chatcmpl-ibm",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", MODEL_ID),
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

        # --- Streaming Fix with Queue ---
        queue = asyncio.Queue()

        async def produce_chunks():
            try:
                await queue.put(json.dumps({
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "model": body.get("model", MODEL_ID),
                    "choices": [{"delta": {"role": "assistant"}}]
                }) + "\n\n")

                await queue.put(json.dumps({
                    "choices": [{"delta": {"content": "..."}}],
                    "object": "chat.completion.chunk"
                }) + "\n\n")

                ibm_response = await call_ibm_watsonx(prompt)

                if "results" not in ibm_response:
                    await queue.put(json.dumps({
                        "choices": [{"delta": {"content": "[Error from IBM Watsonx]"}}],
                        "object": "chat.completion.chunk"
                    }) + "\n\n")
                    await queue.put(json.dumps({
                        "choices": [{"finish_reason": "stop"}],
                        "object": "chat.completion.chunk"
                    }) + "\n\n")
                    return

                full_text = ibm_response["results"][0].get("generated_text", "").strip()

                for word in full_text.split():
                    await asyncio.sleep(0.04)
                    await queue.put(json.dumps({
                        "choices": [{"delta": {"content": word + " "}}],
                        "object": "chat.completion.chunk"
                    }) + "\n\n")

                await queue.put(json.dumps({
                    "choices": [{"finish_reason": "stop"}],
                    "object": "chat.completion.chunk"
                }) + "\n\n")

            except Exception as e:
                await queue.put(json.dumps({
                    "error": str(e)
                }) + "\n\n")

        async def stream_response():
            task = asyncio.create_task(produce_chunks())
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {chunk}\n\n"
                    if '"finish_reason": "stop"' in chunk:
                        break
                except asyncio.TimeoutError:
                    break

        return EventSourceResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        error_trace = traceback.format_exc()
        print("Exception:", error_trace)
        return JSONResponse(status_code=500, content={"error": str(e), "trace": error_trace})
