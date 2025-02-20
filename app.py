from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uuid
import asyncio
import httpx
import logging
import configparser
from memory import get_gpu_total_memory

config = configparser.ConfigParser()
config.read("config.cfg")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s:: Message:> %(message)s'
)

app = FastAPI()

# Calculate total GPU memory
total_memory = get_gpu_total_memory()[1]

# Check if required memory exceeds available GPU memory
pr_queue_count = int(config.get("Server", "PR_QUEUE"))
n_queue_count = int(config.get("Server", "N_QUEUE"))

required_memory = (pr_queue_count * 2048 + n_queue_count * 1024)
if required_memory > total_memory:
    raise ValueError("The required memory exceeds the total GPU memory available.")

# Initialize priority queues (2GB VRAM each)
PRIORITY_QUEUES = [asyncio.Queue() for _ in range(pr_queue_count)]
# Initialize normal queues (1GB VRAM each)
NORMAL_QUEUES = [asyncio.Queue() for _ in range(n_queue_count)]
# Semaphores for concurrency control
PrioritySemaphore = asyncio.Semaphore(value=pr_queue_count * 2)  # 2 processes per priority queue
NormalSemaphore = asyncio.Semaphore(value=n_queue_count)  # 1 process per normal queue

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
        arbitrary_types_allowed = True

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
        arbitrary_types_allowed = True

class Job(BaseModel):
    jobId: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        alias="id",
        description="The job ID"
    )
    data: Union[ModelChatRequest, ModelGenerateRequest]
    result: asyncio.Future = Field(default_factory=asyncio.Future)
    target_url: str
    status: str = Field(
        default="pending",
        description="Job status: pending, processing, done"
    )
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


async def process_pr_jobs(queue):
    """
    Process jobs in priority queues with concurrency control.
    """
    while True:
        job = await queue.get()
        async with PrioritySemaphore:
            try:
                async with httpx.AsyncClient(timeout=100000) as client:
                    response = await client.post(job.target_url, json=job.data.dict())
                    job.result.set_result(response.json())
            except Exception as e:
                job.result.set_exception(e)
            finally:
                job.status = "done"

async def process_jobs(queue):
    """
    Process jobs in normal queues with concurrency control.
    """
    while True:
        job = await queue.get()
        async with NormalSemaphore:
            try:
                async with httpx.AsyncClient(timeout=100000) as client:
                    response = await client.post(job.target_url, json=job.data.dict())
                    job.result.set_result(response.json())
            except Exception as e:
                job.result.set_exception(e)
            finally:
                job.status = "done"

@app.on_event("startup")
async def startup_event():
    """
    Initialize background tasks for processing queues.
    """
    if PRIORITY_QUEUES:
        for queue in PRIORITY_QUEUES:
            asyncio.create_task(process_pr_jobs(queue))
    if NORMAL_QUEUES:
        for queue in NORMAL_QUEUES:
            asyncio.create_task(process_jobs(queue))

@app.post("/api/chat")
async def chat_endpoint(request: Request, chat_request: ModelChatRequest):
    """
    Handle chat requests and enqueue them for processing.
    """
    target_url = "http://127.0.0.1:11434/api/chat"
    job = Job(data=chat_request, target_url=target_url)
    priority_header = request.headers.get("X-Priority")
    if priority_header == "1":
        logging.info(f"New chat job enqueued. Job ID: {job.jobId}, Priority: High")
        selected_queue = min(PRIORITY_QUEUES, key=lambda q: (q.qsize(), PRIORITY_QUEUES.index(q)))
        await selected_queue.put(job)
    elif priority_header == "0" or priority_header is None:
        logging.info(f"New chat job enqueued. Job ID: {job.jobId}, Priority: Normal")
        selected_queue = min(NORMAL_QUEUES, key=lambda q: (q.qsize(), NORMAL_QUEUES.index(q)))
        await selected_queue.put(job)
    else:
        raise HTTPException(status_code=400, detail="Invalid priority header")
    
    result = await job.result
    logging.info(f"Chat job completed. Job ID: {job.jobId}, Status: {job.status}, Priority: {priority_header}")
    return result

@app.post("/api/generate")
async def generate_endpoint(request: Request, generate_request: ModelGenerateRequest):
    """
    Handle generation requests and enqueue them for processing.
    """
    target_url = "http://127.0.0.1:11434/api/chat"
    job = Job(data=generate_request, target_url=target_url)
    priority_header = request.headers.get("X-Priority")
    if priority_header == "1":
        logging.info(f"New chat job enqueued. Job ID: {job.jobId}, Priority: High")
        selected_queue = min(PRIORITY_QUEUES, key=lambda q: (q.qsize(), PRIORITY_QUEUES.index(q)))
        await selected_queue.put(job)
    elif priority_header == "0" or priority_header is None:
        logging.info(f"New chat job enqueued. Job ID: {job.jobId}, Priority: Normal")
        selected_queue = min(NORMAL_QUEUES, key=lambda q: (q.qsize(), NORMAL_QUEUES.index(q)))
        await selected_queue.put(job)
    else:
        raise HTTPException(status_code=400, detail="Invalid priority header")
    
    result = await job.result

    return result
