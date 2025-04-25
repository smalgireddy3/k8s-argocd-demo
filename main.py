from __future__ import annotations

from pydantic import BaseModel

import os
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI,Request
from openai import AzureOpenAI

load_dotenv(find_dotenv())
app = FastAPI()

AZURE_API_KEY = os.getenv("azure_openai_api_key")
AZURE_ENDPOINT = os.getenv("azure_openai_api_endpoint")

print(AZURE_ENDPOINT)


# Pydantic models for request/response
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

static_client: AzureOpenAI = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=os.getenv("AZURE_MODEL_VERSION"),
)
# Simple GET endpoint
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Hello, world!"}

# POST endpoint to Azure OpenAI
@app.post("/ask", response_model=ChatResponse, tags=["Chat"])
async def static_refine(request: Request) -> dict:
    """
    Endpoint for a synchronous call to Azure OpenAI.

    :param request: request body
    :param auth: Authorization blockc
    """
    body = await request.json()
    prompt: str = body.get("prompt", "")
    completion = static_client.chat.completions.create(
        model="NonProdDataScienceOpenAI_GPT-4o-EastUS2",
        messages=[{"role": "user", "content": prompt}],
    )
    content = completion.choices[0].message.content if completion.choices else ""
    return {"response": content}