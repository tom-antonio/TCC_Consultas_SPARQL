from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import tempfile

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
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)