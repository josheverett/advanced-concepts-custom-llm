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

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

def get_flush_chunk(id: str, created: int):
    return {
        'id': id,
        'choices': [{
            'delta': {
                'content': "<flush>",
                'function_call': None,
                'refusal': None,
                'role': None,
                'tool_calls': None
            },
            'finish_reason': None,
            'index': 0,
            'logprobs': None
        }],
        'created': created,
        'model': 'gpt-3.5-turbo-0125',
        'object': 'chat.completion.chunk',
        'service_tier': None,
        'system_fingerprint': None,
        'usage': None
    }

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
                    "content": "Say: 'Thank you for continuing to hold.'",
                }
            ],
            temperature=0,
            stream=True,
        )

        response3 = await client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "user",
                    "content": "Say: 'Thank you for your patience. Here is what I found.'",
                }
            ],
            temperature=0,
            stream=True,
        )

        response4 = await client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "user",
                    "content": "Say: 'Have a great day!'",
                }
            ],
            temperature=0,
            stream=True,
        )

        async def event_stream():
            try:
                logger.info("sending chunks for response 1")
                async for chunk in response1:
                    is_last_chunk = chunk.choices[0].finish_reason is not None
                    if is_last_chunk:
                        yield f"data: {json.dumps(get_flush_chunk(chunk.id, chunk.created))}\n\n"
                    logger.info(f"sending chunk for response 1: {chunk.model_dump()}")
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                await asyncio.sleep(10)

                logger.info("sending chunks for response 2")
                async for chunk in response2:
                    is_last_chunk = chunk.choices[0].finish_reason is not None
                    if is_last_chunk:
                        yield f"data: {json.dumps(get_flush_chunk(chunk.id, chunk.created))}\n\n"
                    logger.info(f"sending chunk for response 2: {chunk.model_dump()}")
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                await asyncio.sleep(10)

                logger.info("sending chunks for response 3")
                async for chunk in response3:
                    is_last_chunk = chunk.choices[0].finish_reason is not None
                    if is_last_chunk:
                        yield f"data: {json.dumps(get_flush_chunk(chunk.id, chunk.created))}\n\n"
                    logger.info(f"sending chunk for response 3: {chunk.model_dump()}")
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                await asyncio.sleep(10)

                logger.info("sending chunks for response 4")
                async for chunk in response4:
                    is_last_chunk = chunk.choices[0].finish_reason is not None
                    if is_last_chunk:
                        yield f"data: {json.dumps(get_flush_chunk(chunk.id, chunk.created))}\n\n"
                    logger.info(f"sending chunk for response 4: {chunk.model_dump()}")
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

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
