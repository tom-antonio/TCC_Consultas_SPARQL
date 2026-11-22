from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import tempfile
import time
import uuid
from typing import Any, Optional

from fastapi.responses import PlainTextResponse, StreamingResponse

from llm_pipeline import LLMPipeline
from pdf import PDFIngestionPipeline

app = FastAPI(
    title="API de Gestão Semântica de Horários - IFG",
    description="Backend para integração com Anything LLM e Apache Jena Fuseki"
)

# Permitir requisições CORS da interface gráfica
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa os pipelines
query_pipeline = LLMPipeline()
ingestion_pipeline = PDFIngestionPipeline()

class QueryRequest(BaseModel):
    question: str

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

def extract_last_user_message(messages: list[ChatMessage]) -> str:
    """Extrai a pergunta mais recente enviada no formato OpenAI."""
    for message in reversed(messages):
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            text_parts = [
                part.get("text", "")
                for part in message.content
                if isinstance(part, dict) and part.get("type") in ("text", None)
            ]
            return "\n".join(text for text in text_parts if text)
    raise HTTPException(status_code=400, detail="Nenhuma mensagem do usuário foi encontrada.")

@app.get("/")
def read_root():
    return {"status": "API Semântica em execução"}

@app.post("/api/chat")
async def process_question(request: QueryRequest):
    """Endpoint para receber perguntas do Anything LLM e retornar a resposta formatada."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")
    
    result = query_pipeline.run(request.question, verbose=False)
    
    if result.get("error") and not result.get("final_answer"):
        raise HTTPException(status_code=500, detail=result["error"])
        
    return {
        "question": result["question"],
        "sparql_generated": result["sparql"],
        "results_json": result["raw_data"],
        "response": result["final_answer"]
    }

@app.post("/api/chat/text", response_class=PlainTextResponse)
async def process_question_text(request: QueryRequest):
    """Retorna diretamente ao AnythingLLM a resposta em texto natural."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

    result = query_pipeline.run(request.question, verbose=False)

    if result.get("error") and not result.get("final_answer"):
        raise HTTPException(status_code=500, detail=result["error"])

    return PlainTextResponse(
        content=result["final_answer"] or "Não foram encontradas informações na ontologia.",
        media_type="text/plain; charset=utf-8"
    )

@app.get("/v1/models")
async def list_models():
    """Anuncia o backend ao provedor OpenAI compatível do AnythingLLM."""
    return {
        "object": "list",
        "data": [
            {
                "id": "semantic-ifg",
                "object": "model",
                "created": 0,
                "owned_by": "ifg"
            }
        ]
    }

def openai_result(request: ChatCompletionRequest) -> dict:
    question = extract_last_user_message(request.messages)
    result = query_pipeline.run(question, verbose=False)

    if result.get("error") and not result.get("final_answer"):
        raise HTTPException(status_code=500, detail=result["error"])

    answer = result["final_answer"] or "Não foi possível obter uma resposta da ontologia."
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    return {
        "completion_id": completion_id,
        "created": created,
        "answer": answer
    }

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatCompletionRequest):
    """Endpoint OpenAI compatível para o provedor Generic do AnythingLLM web."""
    result = openai_result(request)
    base_response = {
        "id": result["completion_id"],
        "object": "chat.completion",
        "created": result["created"],
        "model": "semantic-ifg",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["answer"]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

    if not request.stream:
        return base_response

    def stream_answer():
        chunk = {
            "id": result["completion_id"],
            "object": "chat.completion.chunk",
            "created": result["created"],
            "model": "semantic-ifg",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": result["answer"]},
                    "finish_reason": None
                }
            ]
        }
        import json
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        final_chunk = {
            "id": result["completion_id"],
            "object": "chat.completion.chunk",
            "created": result["created"],
            "model": "semantic-ifg",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_answer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/api/upload-pdf")
async def upload_schedule_pdf(file: UploadFile = File(...)):
    """Endpoint para receber o PDF do horário e povoar a Triplestore Fuseki."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="O arquivo deve ser formato PDF.")
    
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
            
        success = ingestion_pipeline.process_pdf(temp_path)
        
        if success:
            return {"message": "Ontologia povoada com sucesso a partir do PDF fornecido!"}
        else:
            raise HTTPException(status_code=500, detail="Falha ao extrair ou gravar triplas semânticas no Fuseki.")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)