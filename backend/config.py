import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente de um arquivo .env (se existir)
load_dotenv()

# URL do endpoint SPARQL do Fuseki (dataset 'TADS_2026_2')
FUSEKI_QUERY_URL = os.getenv("FUSEKI_QUERY_URL", "http://localhost:3030/TADS_2026_2/query")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Chave de API da LLM (OpenAI, Groq, etc.)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "A variável de ambiente GROQ_API_KEY não foi definida. "
        "Configure-a no arquivo .env."
    )

# Prefixo padrão da ontologia
ONTOLOGY_PREFIX = """
PREFIX : <http://www.semanticweb.org/luan/ontologies/2026/8/untitled-ontology-6#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
