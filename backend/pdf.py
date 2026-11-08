import json
import re
import pdfplumber
from groq import Groq
from SPARQLWrapper import SPARQLWrapper, POST
from config import (
    FUSEKI_ADMIN_PASSWORD,
    FUSEKI_QUERY_URL,
    FUSEKI_USER,
    GROQ_API_KEY,
    GROQ_MODEL,
    ONTOLOGY_PREFIX,
)

# Endpoint para atualização de dados no Fuseki (/update)
FUSEKI_UPDATE_URL = FUSEKI_QUERY_URL.rsplit("/", 1)[0] + "/update"

class PDFIngestionPipeline:
    def __init__(self, model_name: str = GROQ_MODEL):
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.model_name = model_name
        self.sparql_update = SPARQLWrapper(FUSEKI_UPDATE_URL)
        if FUSEKI_ADMIN_PASSWORD:
            self.sparql_update.setCredentials(FUSEKI_USER, FUSEKI_ADMIN_PASSWORD)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrai todo o conteúdo textual do arquivo PDF de horários."""
        extracted_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        return extracted_text

    def generate_triples_from_text(self, pdf_text: str) -> str:
        """Instrui a LLM a extrair a grade e gerar triplas RDF Turtle com instâncias de :Aula."""
        system_prompt = f"""
Você é um extrator de dados semânticos especializado na ontologia acadêmica do IFG.
Sua tarefa é analisar o texto do quadro de horários de um semestre e gerar instrução SPARQL INSERT DATA contendo instâncias da classe :Aula.

ESQUEMA DA ONTOLOGIA:
{ONTOLOGY_PREFIX}

REGRAS DE GERAÇÃO:
1. Para cada horário/bloco identificado na grade, crie um indivíduo do tipo :Aula (ex: :aula_AlgorII_Segunda_Bloco1).
2. Conecte o indivíduo de :Aula às instâncias existentes através das propriedades:
   - :temDisciplina -> aponta para o indivíduo da :Disciplina (ex: :disAlgorII)
   - :ministradaPor -> aponta para o indivíduo do :Professor (ex: :profCleuber)
   - :ministraEm -> aponta para o indivíduo da :Sala (ex: :labI)
   - :pertenceATurma -> aponta para a :Turma (ex: :turma_TADS_2Periodo)
   - :ocorreNoDia -> aponta para o :Dia (ex: :diaSeg)
   - :ocorreEmHorario -> aponta para o :Horario (ex: :HorarioBlocoNoiteI)

3. Retorne APENAS a instrução SPARQL INSERT DATA pronta para ser executada, no seguinte formato:
INSERT DATA {{
  :aula_exemplo1 a :Aula ;
      :temDisciplina :disAlgorII ;
      :ministradaPor :profCleuber ;
      :ministraEm :labI ;
      :pertenceATurma :turma_TADS_2Periodo ;
      :ocorreNoDia :diaSeg ;
      :ocorreEmHorario :HorarioBlocoNoiteI .
}}
"""
        response = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Texto extraído do PDF do semestre:\n{pdf_text}"}
            ],
            temperature=0.0
        )
        
        raw_output = response.choices[0].message.content.strip()
        # Limpa marcações markdown de código
        clean_insert = re.sub(r"```(?:sparql)?\s*(.*?)\s*```", r"\1", raw_output, flags=re.DOTALL)
        return clean_insert.strip()

    def populate_fuseki(self, insert_query: str) -> bool:
        """Executa a query INSERT DATA diretamente no Fuseki via endpoint /update."""
        try:
            # Adiciona os prefixos no topo caso a LLM omitiu
            full_query = f"{ONTOLOGY_PREFIX}\n{insert_query}"
            self.sparql_update.setQuery(full_query)
            self.sparql_update.setMethod(POST)
            self.sparql_update.query()
            return True
        except Exception as e:
            print(f"[X] Erro ao inserir dados na Triplestore: {str(e)}")
            return False

    def process_pdf(self, pdf_path: str):
        """Pipeline completo de ingestão: PDF -> Texto -> Triplas -> Fuseki."""
        print(f"[1] Extraindo texto do PDF: {pdf_path}...")
        pdf_text = self.extract_text_from_pdf(pdf_path)
        
        print("[2] Gerando triplas semânticas RDF com a LLM...")
        insert_query = self.generate_triples_from_text(pdf_text)
        print(f"\nSPARQL INSERT Gerado:\n{insert_query}\n")
        
        print("[3] Gravando instâncias no Jena Fuseki...")
        success = self.populate_fuseki(insert_query)
        if success:
            print(" Ontologia povoada com sucesso a partir do PDF!")
        return success