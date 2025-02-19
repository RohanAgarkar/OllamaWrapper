from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uuid
import asyncio
import httpx
import logging
import configparser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:: Message:> %(message)s')
app = FastAPI()
config = configparser.ConfigParser()
config.read("config.cfg")

PROCESS_QUEUE = [asyncio.Queue() for _ in range(int(config.get("Server", "NO_QUEUE")) or 2)]


class ModelChatRequest(BaseModel):
    model: str = Field(..., description="The model name")
    messages: List[Dict[str, Any]] = Field(..., description="List of messages")
    suffix: Optional[str] = Field(None, description="The text after the model response")
    images: Optional[List[str]] = Field(None, description="List of base64-encoded images for multimodal models")
    
    # Advanced parameters
    format: Optional[str] = Field(None, description="Format of the response (json or JSON schema)")
    options: Optional[dict] = Field(None, description="Additional model parameters such as temperature")
    system: Optional[str] = Field(None, description="System message (overrides what is defined in the Modelfile)")
    template: Optional[str] = Field(None, description="Prompt template to use (overrides what is defined in the Modelfile)")
    stream: Optional[bool] = Field(False, description="If false, response will be returned as a single object")
    raw: Optional[bool] = Field(False, description="If true, no formatting will be applied to the prompt")
    keep_alive: Optional[str] = Field("5m", description="Controls how long the model stays loaded into memory")
    class Config:
        populate_by_name = True
        arbitrary_types_allowed=True

class ModelGenerateRequest(BaseModel):
    model: str = Field(..., description="The model name")
    prompt: str = Field(..., description="The prompt to generate a response for")
    suffix: Optional[str] = Field(None, description="The text after the model response")
    images: Optional[List[str]] = Field(None, description="List of base64-encoded images for multimodal models")
    
    # Advanced parameters
    format: Optional[str] = Field(None, description="Format of the response (json or JSON schema)")
    options: Optional[dict] = Field(None, description="Additional model parameters such as temperature")
    system: Optional[str] = Field(None, description="System message (overrides what is defined in the Modelfile)")
    template: Optional[str] = Field(None, description="Prompt template to use (overrides what is defined in the Modelfile)")
    stream: Optional[bool] = Field(False, description="If false, response will be returned as a single object")
    raw: Optional[bool] = Field(False, description="If true, no formatting will be applied to the prompt")
    keep_alive: Optional[str] = Field("5m", description="Controls how long the model stays loaded into memory")
    class Config:
        populate_by_name = True
        arbitrary_types_allowed=True

class Job(BaseModel):
    jobId: str = Field(default_factory=lambda: uuid.uuid4().hex, alias="id", description="The job ID")
    data: Union[ModelChatRequest, ModelGenerateRequest]
    result: asyncio.Future = Field(default_factory=asyncio.Future)
    target_url: str
    class Config:
        populate_by_name = True
        arbitrary_types_allowed=True
    
async def process_jobs(queue):
    async with httpx.AsyncClient(timeout=100000) as client:
        while True:
            # Retrieve the next job from the assigned queue.
            job = await queue.get()
            # If the job has a dependency, wait for that job's result before proceeding.
            try:
                # Instead of a sleep delay, we now call an external API endpoint.
                # We send the job's data as JSON to the target URL.
                response = await client.post(job.target_url, json=job.data.model_dump())
                response.raise_for_status()
                # Set the job's result to the JSON response from the API.
                job.result.set_result(response.json())
            except Exception as e:
                # If something goes wrong, we capture the exception in the job's future.
                job.result.set_exception(e)
            finally:
                # Mark the job as done in the queue.
                queue.task_done()

@app.on_event("startup")
async def startup_event():
    # Launch three background processors—one per queue.
    for queue in PROCESS_QUEUE:
        asyncio.create_task(process_jobs(queue))


@app.post("/api/chat")
async def chat(chat_request: ModelChatRequest):
    target_url = "http://127.0.0.1:11434/api/chat"
    job = Job(data=chat_request, target_url=target_url)
    logging.info(job)
    selected_queue = min(PROCESS_QUEUE, key=lambda q: (q.qsize(), PROCESS_QUEUE.index(q)))
    await selected_queue.put(job)
    print("\033[94m\033[1m" + f"QUEUE:> {PROCESS_QUEUE}" + "\033[0m")
    result = await job.result
    return result

@app.post("/api/generate")
async def generate(generate_request: ModelGenerateRequest):
    target_url = "http://127.0.0.1:11434/api/generate"
    job = Job(data=generate_request, target_url=target_url)
    logging.info(job)
    selected_queue = min(PROCESS_QUEUE, key=lambda q: (q.qsize(), PROCESS_QUEUE.index(q)))
    await selected_queue.put(job)
    print("\033[94m\033[1m" + f"QUEUE:> {PROCESS_QUEUE}" + "\033[0m")
    result = await job.result
    return result

