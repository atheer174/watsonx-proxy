import json
import os
import time

def main(args):
    messages = args.get("messages", [])
    model = args.get("model", "llama-3-3-70b-instruct")

    if args.get("stream", False):
        # Streaming simulation
        def sse():
            yield f"data: {json.dumps({'id': 'stream-1', 'object': 'chat.completion.chunk', 'model': model, 'choices': [{'delta': {'role': 'assistant'}}]})}\n\n"
            time.sleep(0.1)
            yield f"data: {json.dumps({'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': 'Hello '}}]})}\n\n"
            time.sleep(0.1)
            yield f"data: {json.dumps({'object': 'chat.completion.chunk', 'choices': [{'delta': {'content': 'world!'}}]})}\n\n"
            yield "data: [DONE]\n\n"
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "text/event-stream"},
            "body": sse
        }

    # Normal (non-streaming) fallback
    content = "Hello world!"
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": {
            "id": "chatcmpl-ibm",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        }
    }
