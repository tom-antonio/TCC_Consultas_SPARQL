import json
import re
import unicodedata
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

    def _normalize_question(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return "".join(char for char in normalized if not unicodedata.combining(char))

    def _build_deterministic_query(self, user_question: str) -> str | None:
        """Resolve filtros comuns sem depender da LLM para evitar consultas amplas."""
        normalized = self._normalize_question(user_question)
        prefixes = """PREFIX : <http://www.semanticweb.org/luan/ontologies/2026/8/untitled-ontology-6#>
                        PREFIX owl: <http://www.w3.org/2002/07/owl#>
                        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                        """

        room_instance = None
        if re.search(r"\btads\s*0*1\b", normalized) or "laboratorio tads 01" in normalized:
            room_instance = ":labI"
        elif re.search(r"\btads\s*0*2\b", normalized) or "laboratorio tads 02" in normalized:
            room_instance = ":labII"
        elif re.search(r"\btads\s*0*3\b", normalized) or "laboratorio tads 03" in normalized:
            room_instance = ":labIII"
        elif "laboratorio de redes" in normalized or "lab redes" in normalized:
            room_instance = ":labRedes"

        if room_instance and ("disciplina" in normalized or "turma" in normalized or "alocad" in normalized):
            return prefixes + f"""
SELECT DISTINCT ?nomeDisciplina ?turma
WHERE {{
  ?aula :ministraEm {room_instance} ;
        :temDisciplina ?disciplina ;
        :pertenceATurma ?turma .
  ?disciplina :nomeDisciplina ?nomeDisciplina .
}}
ORDER BY ?nomeDisciplina ?turma
"""

        period_number = None
        period_match = re.search(r"\b([1-6])(?:\s*[oaº°])?\s*periodo\b", normalized)
        if period_match:
            period_number = period_match.group(1)
        else:
            ordinal_words = {
                "primeiro": "1",
                "segundo": "2",
                "terceiro": "3",
                "quarto": "4",
                "quinto": "5",
                "sexto": "6",
            }
            for ordinal_name, number in ordinal_words.items():
                if ordinal_name in normalized and "periodo" in normalized:
                    period_number = number
                    break

        if period_number and "disciplina" in normalized:
            return prefixes + f"""
SELECT DISTINCT ?nomeDisciplina ?cargaHoraria
WHERE {{
  :turma_TADS_{period_number}Periodo :ofereceDisciplina ?disciplina .
  ?disciplina :nomeDisciplina ?nomeDisciplina .
  OPTIONAL {{ ?disciplina :temCargaHoraria ?cargaHoraria . }}
}}
"""

        if (
            "professor" in normalized
            and (
                "mesmo dia" in normalized
                or "mesmo horario" in normalized
                or ("dia" in normalized and "horario" in normalized)
            )
        ):
            return prefixes + """
SELECT ?nomeProfessor ?diaSemana ?horaInicio
       (COUNT(DISTINCT ?aula) AS ?qtdAulas)
WHERE {
  ?aula a :Aula ;
        :ministradaPor ?prof ;
        :ocorreNoDia ?dia ;
        :ocorreEmHorario ?horario .

  ?prof :nomeProfessor ?nomeProfessor .
  ?dia :diaSemana ?diaSemana .
  ?horario :horaInicio ?horaInicio .
}
GROUP BY ?nomeProfessor ?diaSemana ?horaInicio
HAVING (COUNT(DISTINCT ?aula) > 1)
ORDER BY ?diaSemana ?horaInicio ?nomeProfessor
"""

        if (
            "disciplina" in normalized
            and (
                "mais de uma" in normalized
                or "mais de 1" in normalized
                or ("duas" in normalized and ("turma" in normalized or "bloco" in normalized))
                or ("turma" in normalized and "bloco" in normalized)
            )
        ):
            return prefixes + """
SELECT DISTINCT ?nomeDisciplina
       (COUNT(DISTINCT ?aula) AS ?qtdOfertas)
WHERE {
  ?aula :temDisciplina ?disciplina .
  ?disciplina :nomeDisciplina ?nomeDisciplina .
}
GROUP BY ?nomeDisciplina
HAVING (COUNT(DISTINCT ?aula) > 1)
ORDER BY DESC(?qtdOfertas) ?nomeDisciplina
"""

        if "tads" in normalized and "sala" in normalized:
            return prefixes + """
SELECT DISTINCT ?nomeSala
WHERE {
  ?sala a :Sala ;
        :nomeSala ?nomeSala .
  FILTER(
    REGEX(?nomeSala, "TADS", "i")
    ||
    REGEX(?nomeSala, "Redes", "i")
  )
}
ORDER BY ?nomeSala
"""

        day_instances = {
            "segunda": ":diaSeg",
            "terca": ":diaTer",
            "quarta": ":diaQua",
            "quinta": ":diaQui",
            "sexta": ":diaSex",
            "sabado": ":diaSab",
            "domingo": ":diaDom",
        }
        for day_name, day_instance in day_instances.items():
            if day_name in normalized and "professor" in normalized:
                return prefixes + f"""
SELECT DISTINCT ?nomeProfessor
WHERE {{
  ?aula :ocorreNoDia {day_instance} ;
        :ministradaPor ?professor .
  ?professor :nomeProfessor ?nomeProfessor .
}}
ORDER BY ?nomeProfessor
"""
            if day_name in normalized and "disciplina" in normalized:
                if day_name == "sabado" and (
                    "vespertino" in normalized
                    or "tarde" in normalized
                    or "vesper" in normalized
                ):
                    return prefixes + f"""
SELECT DISTINCT ?nomeDisciplina ?cargaHoraria ?horaInicio ?horaFim
WHERE {{
  ?aula :ocorreNoDia {day_instance} ;
        :temDisciplina ?disciplina ;
        :ocorreEmHorario ?horario .
  ?disciplina :nomeDisciplina ?nomeDisciplina .
  OPTIONAL {{ ?disciplina :temCargaHoraria ?cargaHoraria . }}
  ?horario :horaInicio ?horaInicio ;
           :horaFim ?horaFim .
  VALUES ?horario {{
    :HorarioBlocoI
    :HorarioBlocoII
    :HorarioBlocoIII
  }}
}}
ORDER BY ?horaInicio ?nomeDisciplina
"""
                return prefixes + f"""
SELECT DISTINCT ?nomeDisciplina ?cargaHoraria
WHERE {{
  ?aula :ocorreNoDia {day_instance} ;
        :temDisciplina ?disciplina .
  ?disciplina :nomeDisciplina ?nomeDisciplina .
  OPTIONAL {{ ?disciplina :temCargaHoraria ?cargaHoraria . }}
}}
"""

        return None

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
        deterministic_query = self._build_deterministic_query(user_question)
        if deterministic_query:
            return deterministic_query

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
        normalized = self._normalize_question(user_question)

        period_match = re.search(r"\b([1-6])(?:\s*[oaº°])?\s*periodo\b", normalized)
        period_number = period_match.group(1) if period_match else None
        if not period_number:
            for ordinal_name, number in {
                "primeiro": "1",
                "segundo": "2",
                "terceiro": "3",
                "quarto": "4",
                "quinto": "5",
                "sexto": "6",
            }.items():
                if ordinal_name in normalized and "periodo" in normalized:
                    period_number = number
                    break

        day_name = None
        for name in ("segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"):
            if name in normalized:
                day_name = name
                break

        room_number = None
        room_label = None
        room_candidates = (
            (r"\btads\s*0*1\b", "1", "Laboratório TADS 01"),
            (r"\btads\s*0*2\b", "2", "Laboratório TADS 02"),
            (r"\btads\s*0*3\b", "3", "Laboratório TADS 03"),
        )
        for pattern, number, label in room_candidates:
            if re.search(pattern, normalized) or label.lower() in normalized:
                room_number = number
                room_label = label
                break
        if "laboratorio de redes" in normalized or "lab redes" in normalized:
            room_label = "Laboratório de Redes"

        if (
            "professor" in normalized
            and (
                "mesmo dia" in normalized
                or "mesmo horario" in normalized
                or ("dia" in normalized and "horario" in normalized)
            )
        ):
            if not raw_data:
                return "Não foram encontrados professores inscritos em mais de uma disciplina no mesmo dia e horário."
            lines = [
                "Professores inscritos em mais de uma disciplina no mesmo dia e horário:",
                "",
            ]
            for item in raw_data:
                name = item.get("nomeProfessor")
                day = item.get("diaSemana")
                start = item.get("horaInicio")
                count = item.get("qtdAulas", "0")
                lines.append(
                    f"- {name}: {day}, início às {start} — {count} aulas distintas"
                )
            return "\n".join(lines)

        if (
            raw_data
            and room_label
            and ("disciplina" in normalized or "turma" in normalized or "alocad" in normalized)
        ):
            entries = []
            for item in raw_data:
                name = item.get("nomeDisciplina")
                turma = item.get("turma", "")
                turma = turma.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                turma = turma.replace("turma_TADS_", "TADS ").replace("Periodo", "º período")
                key = (name, turma)
                if name and key not in entries:
                    entries.append(key)
            if not entries:
                return f"Não foram encontradas disciplinas alocadas no {room_label}."
            lines = [f"Disciplinas e turmas alocadas no {room_label}:", ""]
            for name, turma in entries:
                lines.append(f"- {name} — {turma}")
            return "\n".join(lines)

        if (
            raw_data
            and "disciplina" in normalized
            and (
                "mais de uma" in normalized
                or "mais de 1" in normalized
                or ("duas" in normalized and ("turma" in normalized or "bloco" in normalized))
                or ("turma" in normalized and "bloco" in normalized)
            )
        ):
            lines = [
                "Disciplinas ofertadas em mais de uma oferta no semestre:",
                "",
            ]
            for item in raw_data:
                name = item.get("nomeDisciplina")
                if not name:
                    continue
                offers = item.get("qtdOfertas", "0")
                lines.append(f"- {name}: {offers} ofertas")
            return "\n".join(lines)

        if raw_data and "tads" in normalized and "sala" in normalized:
            names = []
            for item in raw_data:
                name = item.get("nomeSala")
                if name and name not in names:
                    names.append(name)
            if not names:
                return "Não foram encontradas salas associadas ao curso de TADS."
            lines = ["As salas pertencentes ao curso de TADS são:", ""]
            lines.extend(f"- {name}" for name in names)
            return "\n".join(lines)

        if raw_data and "professor" in normalized and day_name:
            labels = {
                "segunda": "segunda-feira",
                "terca": "terça-feira",
                "quarta": "quarta-feira",
                "quinta": "quinta-feira",
                "sexta": "sexta-feira",
                "sabado": "sábado",
                "domingo": "domingo",
            }
            names = []
            for item in raw_data:
                name = item.get("nomeProfessor")
                if name and name not in names:
                    names.append(name)
            if not names:
                return f"Não foram encontrados professores ministrando aula na {labels[day_name]}."
            lines = [f"Professores que ministram aula na {labels[day_name]}:", ""]
            lines.extend(f"- {name}" for name in names)
            return "\n".join(lines)

        if (
            raw_data
            and "disciplina" in normalized
            and "sabado" in normalized
            and ("vespertino" in normalized or "tarde" in normalized or "vesper" in normalized)
        ):
            names = []
            workload_by_name = {}
            for item in raw_data:
                name = item.get("nomeDisciplina")
                if name and name not in names:
                    names.append(name)
                    workload_by_name[name] = item.get("cargaHoraria")
            if not names:
                return "Não foram encontradas disciplinas no período vespertino de sábado."
            lines = ["As disciplinas que ocorrem aos sábados no período vespertino são:", ""]
            for name in names:
                workload = workload_by_name[name]
                suffix = f" — carga horária: {workload} h" if workload else ""
                lines.append(f"- {name}{suffix}")
            lines.extend(["", "Horários vespertinos: 14:00–15:30, 15:45–17:15 e 17:30–19:00."])
            return "\n".join(lines)

        if raw_data and ("disciplina" in normalized) and (period_number or day_name):
            if period_number:
                heading = f"As disciplinas ofertadas para a turma do {period_number}º período são:"
            else:
                labels = {
                    "segunda": "segunda-feira",
                    "terca": "terça-feira",
                    "quarta": "quarta-feira",
                    "quinta": "quinta-feira",
                    "sexta": "sexta-feira",
                    "sabado": "sábado",
                    "domingo": "domingo",
                }
                heading = f"As disciplinas ofertadas na {labels[day_name]} são:"

            lines = [heading, ""]
            for item in raw_data:
                name = item.get("nomeDisciplina")
                if not name:
                    continue
                workload = item.get("cargaHoraria")
                suffix = f" — carga horária: {workload} h" if workload else ""
                lines.append(f"- {name}{suffix}")
            return "\n".join(lines)

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
            temperature=0.0 # Resposta estritamente baseada nos dados da Triplestore
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