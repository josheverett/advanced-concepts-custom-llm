import asyncio
import datetime
import json
import time
from fastapi import FastAPI
from openai import AsyncOpenAI
import os
from starlette.responses import StreamingResponse
from app.types.vapi import ChatRequest
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# This modification to the Custom LLM sample project demonstrates an issue with streaming chat.completion.chunks when
# the end token is not received within 20 seconds. Note that the Web SDK was used for this test.
#
# All chunks are streamed in real time to the client as expected. When the end token is received, this is reflected in
# the logs, but the logs do not show a subsequent "Voice input" log, nor is there a corresponding "voice-input" event
# in the client.

@app.post("/chat/completions")
async def chat_completion_stream(vapi_payload: ChatRequest): # Note: Ignoring the payload for this demonstration.
    try:
        response1 = await client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "user",
                    "content": "Say: 'Please hold for a moment.'",
                }
            ],
            temperature=0,
            stream=True,
        )

        response2 = await client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "user",
                    "content": "Say: 'Thank you for holding.'",
                }
            ],
            temperature=0,
            stream=True,
        )

        async def event_stream():
            try:
                # Wait for 10 seconds to make sure we exhaust the 20s limit.
                await asyncio.sleep(10)

                async for chunk in response1:
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                # Wait another 10 seconds.
                await asyncio.sleep(10)

                async for chunk in response2:
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                # Wait another 10 seconds.
                await asyncio.sleep(10)

                # Send end token.
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Error during response streaming: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        return StreamingResponse(
            f"data: {json.dumps({'error': str(e)})}\n\n", media_type="text/event-stream"
        )
