from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import tempfile
import time
import uuid
import json
import threading
from pathlib import Path
from typing import Any, Optional
from SPARQLWrapper import SPARQLWrapper, POST

from fastapi.responses import PlainTextResponse, StreamingResponse

from llm_pipeline import LLMPipeline
from pdf import PDFIngestionPipeline
from config import FUSEKI_QUERY_URL, FUSEKI_ADMIN_PASSWORD, FUSEKI_USER

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

ANYTHINGLLM_UPLOADS_DIR = Path("/anythingllm-storage/direct-uploads")
INGESTION_STATE_FILE = Path("/anythingllm-storage/semantic-ingested.json")
ingestion_status = {
    "last_upload": None,
    "last_status": "waiting",
    "last_error": None,
    "progress_percent": 0,
    "stage": "Aguardando upload",
    "started_at": None,
    "elapsed_seconds": 0,
    "estimated_remaining_seconds": None,
}


def update_ingestion_progress(progress_percent: int, stage: str) -> None:
    started_at = ingestion_status["started_at"]
    elapsed = 0
    estimated_remaining = None
    if started_at is not None:
        elapsed = max(0, int(time.time() - started_at))
        if 0 < progress_percent < 100:
            estimated_remaining = max(
                0,
                int(elapsed * (100 - progress_percent) / progress_percent),
            )
    ingestion_status.update(
        progress_percent=max(0, min(100, progress_percent)),
        stage=stage,
        elapsed_seconds=elapsed,
        estimated_remaining_seconds=estimated_remaining,
    )


def reset_aula_instances() -> None:
    sparql = SPARQLWrapper(FUSEKI_QUERY_URL.rstrip("/") + "/update")
    if FUSEKI_ADMIN_PASSWORD:
        sparql.setCredentials(FUSEKI_USER, FUSEKI_ADMIN_PASSWORD)
    sparql.setMethod(POST)
    sparql.setQuery("""
PREFIX : <http://www.semanticweb.org/luan/ontologies/2026/8/untitled-ontology-6#>
DELETE {
  ?aula ?propriedade ?valor .
}
WHERE {
  ?aula a :Aula ;
        ?propriedade ?valor .
}
""")
    sparql.query()


def load_ingestion_state() -> set:
    if not INGESTION_STATE_FILE.exists():
        return set()
    try:
        data = json.loads(INGESTION_STATE_FILE.read_text(encoding="utf-8"))
        return set(data if isinstance(data, list) else data.get("ids", []))
    except Exception:
        return set()


def save_ingestion_state(ids: set) -> None:
    try:
        INGESTION_STATE_FILE.write_text(
            json.dumps({"ids": sorted(ids)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as error:
        print(f"[X] Falha ao salvar estado de ingestão: {error}")


def seed_existing_uploads() -> set:
    ids = load_ingestion_state()
    if ANYTHINGLLM_UPLOADS_DIR.exists():
        existing_ids = {
            path.stem
            for path in ANYTHINGLLM_UPLOADS_DIR.glob("*.json")
        }
        if not INGESTION_STATE_FILE.exists():
            ids.update(existing_ids)
            save_ingestion_state(ids)
    return ids


def watch_anythingllm_uploads() -> None:
    ingested_ids = seed_existing_uploads()

    while True:
        try:
            if ANYTHINGLLM_UPLOADS_DIR.exists():
                for metadata_path in ANYTHINGLLM_UPLOADS_DIR.glob("*.json"):
                    upload_id = metadata_path.stem
                    if upload_id in ingested_ids:
                        continue

                    try:
                        metadata = json.loads(
                            metadata_path.read_text(encoding="utf-8")
                        )
                        title = metadata.get("title") or metadata_path.name
                        page_content = metadata.get("pageContent", "").strip()
                        if not page_content:
                            ingestion_status.update(
                                last_upload=title,
                                last_status="ignored",
                                last_error="O PDF não possui texto extraível.",
                            )
                            ingested_ids.add(upload_id)
                            save_ingestion_state(ingested_ids)
                            continue

                        ingestion_status.update(
                            last_upload=title,
                            last_status="processing",
                            last_error=None,
                            started_at=time.time(),
                        )
                        update_ingestion_progress(5, "Texto do PDF recebido")
                        success = ingestion_pipeline.process_text(
                            page_content,
                            status_callback=update_ingestion_progress,
                        )
                        if success:
                            ingested_ids.add(upload_id)
                            save_ingestion_state(ingested_ids)
                            ingestion_status.update(
                                last_status="success",
                                last_error=None,
                                progress_percent=100,
                                stage="Ingestão concluída",
                                estimated_remaining_seconds=0,
                                elapsed_seconds=(
                                    int(time.time() - ingestion_status["started_at"])
                                    if ingestion_status["started_at"] is not None
                                    else 0
                                ),
                            )
                        else:
                            ingestion_status.update(
                                last_status="error",
                                last_error="Falha ao gravar as triplas no Fuseki.",
                            )
                    except Exception as error:
                        ingestion_status.update(
                            last_upload=metadata_path.name,
                            last_status="error",
                            last_error=str(error),
                        )
        except Exception as error:
            ingestion_status.update(last_status="error", last_error=str(error))

        time.sleep(3)


@app.on_event("startup")
def start_anythingllm_upload_watcher() -> None:
    watcher = threading.Thread(
        target=watch_anythingllm_uploads,
        name="anythingllm-pdf-watcher",
        daemon=True,
    )
    watcher.start()

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

@app.get("/api/ingestion-status")
def get_ingestion_status():
    """Retorna o estado da última ingestão feita pelo upload do AnythingLLM."""
    if ingestion_status["started_at"] is not None:
        if ingestion_status["last_status"] == "processing":
            elapsed = max(0, int(time.time() - ingestion_status["started_at"]))
            progress = ingestion_status["progress_percent"]
            remaining = ingestion_status["estimated_remaining_seconds"]
            if progress > 0 and progress < 100:
                remaining = max(0, int(elapsed * (100 - progress) / progress))
            ingestion_status["elapsed_seconds"] = elapsed
            ingestion_status["estimated_remaining_seconds"] = remaining
    return ingestion_status


def format_ingestion_progress() -> str:
    get_ingestion_status()
    progress = ingestion_status["progress_percent"]
    stage = ingestion_status["stage"]
    elapsed = ingestion_status["elapsed_seconds"]
    remaining = ingestion_status["estimated_remaining_seconds"]
    remaining_text = (
        f"Aproximadamente {remaining // 60} min {remaining % 60} s restantes"
        if remaining is not None
        else "Calculando tempo restante"
    )
    filled = round(progress / 10)
    bar = "[" + "#" * filled + "-" * (10 - filled) + "]"
    return (
        f"{bar} {progress}% — {stage}\n"
        f"Tempo decorrido: {elapsed // 60} min {elapsed % 60} s\n"
        f"{remaining_text}"
    )

@app.post("/api/reset-ontology")
def reset_ontology():
    """Remove todas as instâncias de Aula ao criar uma nova thread."""
    try:
        reset_aula_instances()
        if ANYTHINGLLM_UPLOADS_DIR.exists():
            ingested_ids = {
                path.stem
                for path in ANYTHINGLLM_UPLOADS_DIR.glob("*.json")
            }
            save_ingestion_state(ingested_ids)
        ingestion_status.update(
            last_upload=None,
            last_status="waiting",
            last_error=None,
            progress_percent=0,
            stage="Aguardando upload",
            started_at=None,
            elapsed_seconds=0,
            estimated_remaining_seconds=None,
        )
        return {"message": "Instâncias de Aula removidas.", "instancias_aula": 0}
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao resetar a ontologia: {str(error)}",
        ) from error

@app.post("/api/chat")
async def process_question(request: QueryRequest):
    """Endpoint para receber perguntas do Anything LLM e retornar a resposta formatada."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

    if ingestion_status["last_status"] == "processing":
        return {
            "question": request.question,
            "sparql_generated": None,
            "results_json": [],
            "response": "O PDF enviado está sendo processado.\n\n" + format_ingestion_progress(),
        }
    
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

    if ingestion_status["last_status"] == "processing":
        return PlainTextResponse(
            content="O PDF enviado está sendo processado.\n\n" + format_ingestion_progress(),
            media_type="text/plain; charset=utf-8",
        )

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
    if ingestion_status["last_status"] == "processing":
        return {
            "completion_id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "answer": "O PDF enviado está sendo processado.\n\n" + format_ingestion_progress(),
        }
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
            return {
                "message": "Ontologia povoada com sucesso a partir do PDF fornecido.",
                "instancias_aula": ingestion_pipeline.get_aula_count(),
                "total_triplas": ingestion_pipeline.get_triple_count(),
            }
        else:
            raise HTTPException(status_code=500, detail="Falha ao extrair ou gravar triplas semânticas no Fuseki.")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[X] Falha no upload do PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao extrair ou gravar triplas semânticas no Fuseki: {str(e)}",
        ) from e
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)