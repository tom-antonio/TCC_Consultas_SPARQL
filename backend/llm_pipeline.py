import json
import re
from groq import Groq
from config import GROQ_API_KEY, FUSEKI_QUERY_URL, GROQ_MODEL
from fuseki_sparql import FusekiSparql
from prompt import build_sparql_prompt, build_response_prompt

class LLMPipeline:
    def __init__(self, model_name: str = GROQ_MODEL):
        """
        Inicializa a pipeline integrando a API do Groq com o Fuseki.
        """
        self.client = Groq(api_key=GROQ_API_KEY)
        self.fuseki = FusekiSparql(endpoint_url=FUSEKI_QUERY_URL)
        self.model_name = model_name

    def _clean_sparql_code(self, raw_text: str) -> str:
        """
        Extrai e limpa apenas o bloco de código SPARQL retornado pela LLM,
        removendo marcações de markdown do tipo ```sparql ... ```.
        """
        # Procura por blocos entre ```sparql e ```
        match = re.search(r"```(?:sparql)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Se não houver marcação markdown, remove apenas possíveis linhas soltas
        cleaned = raw_text.replace("```sparql", "").replace("```", "").strip()
        return cleaned

    def generate_sparql(self, user_question: str) -> str:
        """
        Etapa 1: Envia a pergunta do usuário para o Qwen via Groq e gera a consulta SPARQL.
        """
        prompt = build_sparql_prompt(user_question)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um gerador estrito de SPARQL. Responda APENAS com a consulta SPARQL executável, sem explicações em linguagem natural."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0 # Temperatura zero para maior precisão determinística na sintaxe SPARQL
        )
        
        raw_sparql = response.choices[0].message.content
        return self._clean_sparql_code(raw_sparql)

    def generate_natural_response(self, user_question: str, raw_data: list) -> str:
        """
        Etapa 3: Recebe os dados brutos obtidos no Fuseki e solicita à LLM a resposta final amigável.
        """
        prompt = build_response_prompt(user_question, raw_data)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente acadêmico amigável do IFG Câmpus Jataí."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3 # Temperatura baixa para evitar alucinações de dados
        )
        
        return response.choices[0].message.content.strip()

    def run(self, user_question: str, verbose: bool = True) -> dict:
        """
        Executa o fluxo completo do pipeline Text-to-SPARQL com tratamento de erros.
        """
        result_log = {
            "question": user_question,
            "sparql": None,
            "raw_data": [],
            "final_answer": None,
            "error": None
        }

        try:
            # 1. Tradução Text-to-SPARQL
            if verbose:
                print(f"\n[1] Pergunta recebida: '{user_question}'")
                print("-> Solicitando geração de SPARQL ao Qwen via Groq...")
                
            sparql_query = self.generate_sparql(user_question)
            result_log["sparql"] = sparql_query
            
            if verbose:
                print(f"\n[2] Consulta SPARQL Gerada:\n{sparql_query}")

            # 2. Execução na Triplestore Fuseki
            if verbose:
                print("\n-> Executando consulta no Apache Jena Fuseki...")
                
            fuseki_res = self.fuseki.execute_query(sparql_query)
            
            if not fuseki_res["success"]:
                result_log["error"] = fuseki_res["error"]
                if verbose:
                    print(f"\n[X] Erro no Fuseki: {fuseki_res['error']}")
                result_log["final_answer"] = "Desculpe, ocorreu uma falha ao interpretar a consulta na base semântica."
                return result_log

            # Limpa e formata os bindings RDF em um JSON legível
            clean_data = self.fuseki.format_bindings_to_clean_json(fuseki_res["data"])
            result_log["raw_data"] = clean_data
            
            if verbose:
                print(f"\n[3] Dados Retornados do Fuseki ({len(clean_data)} registro(s)):\n{json.dumps(clean_data, ensure_ascii=False, indent=2)}")

            # 3. Formatação da Resposta Final em Linguagem Natural
            if verbose:
                print("\n-> Formatando resposta em linguagem natural...")
                
            final_answer = self.generate_natural_response(user_question, clean_data)
            result_log["final_answer"] = final_answer
            
            if verbose:
                print(f"\n[4] Resposta Final do Agente:\n{final_answer}\n")
                
            return result_log

        except Exception as e:
            result_log["error"] = str(e)
            result_log["final_answer"] = "Ocorreu um erro interno durante o processamento do pipeline."
            if verbose:
                print(f"\n[X] Exceção no Pipeline: {str(e)}")
            return result_log

# Teste direto do módulo via terminal
if __name__ == "__main__":
    pipeline = LLMPipeline()
    
    # Pergunta de teste para validação da Fase 3
    pergunta_teste = "Quais disciplinas são ofertadas para a turma do 6° período?"
    pipeline.run(pergunta_teste)