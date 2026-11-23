import json
import re
import unicodedata
import pdfplumber
from groq import Groq
from SPARQLWrapper import SPARQLWrapper, POST, GET, JSON
from config import (
    FUSEKI_ADMIN_PASSWORD,
    FUSEKI_QUERY_URL,
    FUSEKI_USER,
    GROQ_API_KEY,
    GROQ_MODEL,
    ONTOLOGY_PREFIX,
)

# Endpoint para atualização de dados no Fuseki (/update)
FUSEKI_UPDATE_URL = FUSEKI_QUERY_URL.rstrip("/") + "/update"

class PDFIngestionPipeline:
    def __init__(self, model_name: str = GROQ_MODEL):
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.model_name = model_name
        self.sparql_update = SPARQLWrapper(FUSEKI_UPDATE_URL)
        if FUSEKI_ADMIN_PASSWORD:
            self.sparql_update.setCredentials(FUSEKI_USER, FUSEKI_ADMIN_PASSWORD)

    def get_triple_count(self) -> int:
        """Retorna a quantidade atual de triplas no dataset."""
        sparql = SPARQLWrapper(FUSEKI_QUERY_URL)
        sparql.setReturnFormat(JSON)
        sparql.setMethod(GET)
        if FUSEKI_ADMIN_PASSWORD:
            sparql.setCredentials(FUSEKI_USER, FUSEKI_ADMIN_PASSWORD)
        sparql.setQuery("SELECT (COUNT(*) AS ?total) WHERE { ?s ?p ?o }")
        binding = sparql.query().convert()["results"]["bindings"][0]
        return int(binding["total"]["value"])

    def get_aula_count(self) -> int:
        """Retorna a quantidade de instâncias de Aula após a ingestão."""
        sparql = SPARQLWrapper(FUSEKI_QUERY_URL)
        sparql.setReturnFormat(JSON)
        sparql.setMethod(GET)
        if FUSEKI_ADMIN_PASSWORD:
            sparql.setCredentials(FUSEKI_USER, FUSEKI_ADMIN_PASSWORD)
        sparql.setQuery(f"""{ONTOLOGY_PREFIX}
SELECT (COUNT(DISTINCT ?aula) AS ?total) WHERE {{
  ?aula a :Aula .
}}
""")
        binding = sparql.query().convert()["results"]["bindings"][0]
        return int(binding["total"]["value"])

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrai todo o conteúdo textual do arquivo PDF de horários."""
        extracted_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        extracted_text = extracted_text.strip()
        if not extracted_text:
            raise ValueError(
                "Não foi possível extrair texto do PDF. "
                "Verifique se o arquivo contém texto selecionável ou não é apenas uma imagem."
            )
        return extracted_text

    def get_resource_inventory(self) -> str:
        """Lista os indivíduos existentes para evitar que a LLM invente IRIs."""
        sparql = SPARQLWrapper(FUSEKI_QUERY_URL)
        sparql.setReturnFormat(JSON)
        sparql.setMethod(GET)
        if FUSEKI_ADMIN_PASSWORD:
            sparql.setCredentials(FUSEKI_USER, FUSEKI_ADMIN_PASSWORD)

        resources = [
            ("Disciplinas", ":Disciplina", ":nomeDisciplina"),
            ("Professores", ":Professor", ":nomeProfessor"),
            ("Salas", ":Sala", ":nomeSala"),
            ("Turmas", ":Turma", None),
        ]
        inventory = []

        for title, class_name, label_property in resources:
            legacy_name_clause = (
                f"OPTIONAL {{ ?recurso {label_property} ?legado . }}\n"
                if label_property
                else ""
            )
            query = f"""{ONTOLOGY_PREFIX}
SELECT DISTINCT ?recurso ?nome WHERE {{
  ?recurso a {class_name} .
  OPTIONAL {{ ?recurso rdfs:label ?label . }}
  {legacy_name_clause}
  BIND(COALESCE(?label, ?legado) AS ?nome)
}}
ORDER BY ?nome
"""
            sparql.setQuery(query)
            bindings = sparql.query().convert()["results"]["bindings"]
            items = []
            for binding in bindings:
                iri = binding["recurso"]["value"].rsplit("#", 1)[-1]
                name = binding.get("nome", {}).get("value")
                items.append(f":{iri}" + (f" ({name})" if name else ""))
            inventory.append(f"{title}: " + ", ".join(items))

        for title, class_name in (
            ("Dias", ":Dia"),
            ("Horarios", ":Horario"),
        ):
            query = f"""{ONTOLOGY_PREFIX}
SELECT DISTINCT ?recurso WHERE {{
  ?recurso a {class_name} .
}}
ORDER BY ?recurso
"""
            sparql.setQuery(query)
            bindings = sparql.query().convert()["results"]["bindings"]
            items = [
                ":" + binding["recurso"]["value"].rsplit("#", 1)[-1]
                for binding in bindings
            ]
            inventory.append(f"{title}: " + ", ".join(items))

        return "\n".join(inventory)

    def _normalize_day_name(self, day_name: str) -> str:
        """Converte nomes por dia para o padrão usado no texto extraído."""
        return {
            "segunda-feira": "Segunda",
            "terça-feira": "Terça",
            "quarta-feira": "Quarta",
            "quinta-feira": "Quinta",
            "sexta-feira": "Sexta",
            "sábado": "Sábado",
        }[day_name]

    def _normalize_resource_name(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return "".join(
            char for char in normalized
            if not unicodedata.combining(char)
        )

    def get_named_disciplines(self) -> list:
        sparql = SPARQLWrapper(FUSEKI_QUERY_URL)
        sparql.setReturnFormat(JSON)
        sparql.setMethod(GET)
        if FUSEKI_ADMIN_PASSWORD:
            sparql.setCredentials(FUSEKI_USER, FUSEKI_ADMIN_PASSWORD)
        sparql.setQuery(f"""{ONTOLOGY_PREFIX}
SELECT DISTINCT ?recurso ?nome
WHERE {{
  ?recurso :nomeDisciplina ?legado .
  OPTIONAL {{ ?recurso rdfs:label ?label . }}
  BIND(COALESCE(?label, ?legado) AS ?nome)
}}
ORDER BY ?nome
""")
        bindings = sparql.query().convert()["results"]["bindings"]
        disciplines = []
        for binding in bindings:
            iri = binding["recurso"]["value"]
            name = binding["nome"]["value"]
            disciplines.append((iri.rsplit("#", 1)[-1], name))
        return disciplines

    def get_turmas_offering(self, local_iri: str) -> list:
        sparql = SPARQLWrapper(FUSEKI_QUERY_URL)
        sparql.setReturnFormat(JSON)
        sparql.setMethod(GET)
        if FUSEKI_ADMIN_PASSWORD:
            sparql.setCredentials(FUSEKI_USER, FUSEKI_ADMIN_PASSWORD)
        sparql.setQuery(f"""{ONTOLOGY_PREFIX}
SELECT DISTINCT ?turma
WHERE {{
  ?turma a :Turma ;
         :ofereceDisciplina :{local_iri} .
}}
ORDER BY ?turma
""")
        bindings = sparql.query().convert()["results"]["bindings"]
        return [
            binding["turma"]["value"].rsplit("#", 1)[-1]
            for binding in bindings
        ]

    def get_room_context(self, block_text: str, discipline_name: str) -> str:
        normalized_block = self._normalize_resource_name(block_text)
        normalized_name = self._normalize_resource_name(discipline_name)
        name_match = re.search(re.escape(normalized_name), normalized_block)
        context = normalized_block
        if name_match:
            context_start = max(0, name_match.start() - 20)
            context_end = min(len(normalized_block), name_match.end() + 100)
            context = normalized_block[context_start:context_end]

        if "tads 03" in context or "tads03" in context:
            return ":labIII"
        if "tads 02" in context or "tads02" in context:
            return ":labII"
        if "tads 01" in context or "tads01" in context:
            return ":labI"
        if "redes" in context:
            return ":labRedes"
        return ""

    def build_generic_saturday_aulas(self, day_text: str) -> str:
        """Cria aulas de sábado por correspondência dinâmica entre texto e nomes da ontologia."""
        disciplines = self.get_named_disciplines()
        time_positions = list(re.finditer(
            r"[0-9]{2}:[0-9]{2}\s*[-–]\s*[0-9]{2}:[0-9]{2}",
            day_text,
        ))
        if not time_positions:
            return ""

        aula_parts = []
        for index, match in enumerate(time_positions):
            start_text = match.group(0)
            start_time = start_text.split()[0].split("-")[0].replace("–", "-")
            if "-" in start_time:
                start_time = start_time.split("-")[0]

            end = (
                time_positions[index + 1].start()
                if index + 1 < len(time_positions)
                else len(day_text)
            )
            block_text = day_text[match.end():end]
            normalized_block = self._normalize_resource_name(block_text)

            horario = None
            match_hour = re.search(r"([0-9]{2}:[0-9]{2})", start_text)
            hour = match_hour.group(1) if match_hour else ""
            if hour == "07:00":
                horario = ":HorarioBlocoManhaI"
            elif hour == "08:45":
                horario = ":HorarioBlocoManhaII"
            elif hour in ("14:00", "19:00"):
                horario = ":HorarioBlocoI" if hour == "14:00" else ":HorarioBlocoNoiteI"
            elif hour in ("15:45", "20:45"):
                horario = ":HorarioBlocoII" if hour == "15:45" else ":HorarioBlocoNoiteII"
            elif hour in ("17:15", "17:30"):
                horario = ":HorarioBlocoIII"

            if horario is None:
                continue

            for local_iri, discipline_name in disciplines:
                normalized_name = self._normalize_resource_name(discipline_name)
                exact_name_pattern = (
                    r"(?<![a-z0-9])"
                    + re.escape(normalized_name)
                    + r"(?![a-z0-9])"
                )
                if not re.search(exact_name_pattern, normalized_block):
                    continue
                safe_name = re.sub(r"[^A-Za-z0-9]", "_", discipline_name)
                aula_name = f":aula_{safe_name}_Sab_{hour.replace(':', '')}"
                room = self.get_room_context(block_text, discipline_name)
                offering_turmas = self.get_turmas_offering(local_iri)
                turma_clause = ""
                if offering_turmas:
                    turma_clause = f"\n      :pertenceATurma :{offering_turmas[0]} ;"
                room_clause = f"\n      :ministraEm {room} ;" if room else ""
                aula_parts.append(
                    f"{aula_name} a :Aula ;\n"
                    f"      :temDisciplina :{local_iri} ;"
                    f"{room_clause}"
                    f"{turma_clause}\n"
                    f"      :ocorreNoDia :diaSab ;\n"
                    f"      :ocorreEmHorario {horario} ."
                )

        return "\n\n".join(aula_parts)

    def generate_triples_from_text(self, pdf_text: str, status_callback=None) -> str:
        """Instrui a LLM a extrair a grade e gerar triplas RDF Turtle com instâncias de :Aula."""
        inventory = self.get_resource_inventory()
        system_prompt = f"""
Você é um extrator de dados semânticos especializado na ontologia acadêmica do IFG.
Sua tarefa é analisar o texto do quadro de horários de um semestre e gerar instrução SPARQL INSERT DATA contendo instâncias da classe :Aula.

ESQUEMA DA ONTOLOGIA:
{ONTOLOGY_PREFIX}

REGRAS DE GERAÇÃO:
1. Para cada horário/bloco identificado na grade, crie um indivíduo do tipo :Aula.
2. Use SOMENTE os indivíduos existentes listados abaixo. Não invente nomes de disciplinas, professores, salas, turmas, dias ou horários.
3. Para nomes encontrados no PDF, escolha o recurso existente cuja forma normalizada corresponda. Se não houver correspondência segura, não inclua essa aula.
4. Use identificadores locais simples em novos indivíduos, por exemplo :aula_TADS2_Segunda_BlocoI.
5. Correlacione a coluna de período do PDF com uma instância de turma existente.
6. Correlacione cada sala indicada no PDF com uma instância de sala existente.
7. Cada bloco de horário do PDF deve gerar uma aula própria.
8. Se o mesmo bloco ocorrer em períodos diferentes, gere uma aula para cada período.
9. Não pule disciplinas vazias. Não agrupe disciplinas diferentes em uma única aula.
10. Antes de finalizar, verifique o trecho do dia: para cada disciplina e período presentes no texto, inclua todas as ocorrências de horário.
11. Quando o trecho estiver marcado como Sábado, inclua também os blocos matutinos 07:00-08:30 e 08:45-10:15 e os blocos vespertinos 14:00-15:30, 15:45-17:15 e 17:30-18:45.

INDIVÍDUOS EXISTENTES:
{inventory}

5. Conecte o indivíduo de :Aula às instâncias existentes através das propriedades:
   - :temDisciplina -> aponta para o indivíduo da :Disciplina
   - :ministradaPor -> aponta para o indivíduo do :Professor
   - :ministraEm -> aponta para o indivíduo da :Sala
   - :pertenceATurma -> aponta para a :Turma
   - :ocorreNoDia -> aponta para o :Dia
   - :ocorreEmHorario -> aponta para o :Horario

3. Retorne APENAS a instrução SPARQL INSERT DATA pronta para ser executada, no seguinte formato:
INSERT DATA {{
  :aula_exemplo1 a :Aula ;
      :temDisciplina :DisciplinaExistente ;
      :ministradaPor :ProfessorExistente ;
      :ministraEm :SalaExistente ;
      :pertenceATurma :TurmaExistente ;
      :ocorreNoDia :diaSeg ;
      :ocorreEmHorario :HorarioBlocoNoiteI .
}}
"""

        day_names = ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado")
        inserts = []

        for day_index, day_name in enumerate(day_names, start=1):
            if status_callback:
                status_callback(
                    15 + int((day_index - 1) * 65 / len(day_names)),
                    f"Extraindo triplas de {day_name}",
                )
            normalized_day = self._normalize_day_name(day_name)
            if day_name == "sábado":
                day_pattern = re.compile(r"(?ims)^Sábado\b.*")
            else:
                day_pattern = re.compile(
                    rf"(?ims)^(?:{normalized_day}|{day_name})\b.*?"
                    rf"(?=^(?:Segunda|Terça|Quarta|Quinta|Sexta|Sábado)\b|\Z)"
                )
            day_match = day_pattern.search(pdf_text)
            day_text = day_match.group(0).strip() if day_match else ""
            if day_name == "sábado" and not day_text:
                saturday_match = re.search(r"(?is)\bSábado\b.*", pdf_text)
                day_text = saturday_match.group(0).strip() if saturday_match else ""
            if not day_text:
                continue

            room_markers = (
                (r"TADS\s*0*1", ":labI"),
                (r"TADS\s*0*2", ":labII"),
                (r"TADS\s*0*3", ":labIII"),
                (r"(LAB\s*)?REDES", ":labRedes"),
            )
            best_inner = ""
            best_coverage = -1

            for attempt in range(3 if day_name == "sábado" else 2):
                correction = ""
                if day_name == "sábado":
                    correction = (
                        "Trate cada linha de horário do trecho de Sábado como uma ocorrência separada "
                        "e inclua todos os blocos presentes, incluindo manhã e tarde."
                        " Sua resposta deve conter pelo menos uma triplas usando :ocorreNoDia :diaSab."
                    )
                if attempt > 0 and best_inner:
                    missing_rooms = []
                    for marker_pattern, room_instance in room_markers:
                        marker_count = len(re.findall(marker_pattern, day_text, flags=re.IGNORECASE))
                        room_count = len(re.findall(
                            rf":ministraEm\s+{re.escape(room_instance)}",
                            best_inner,
                            flags=re.IGNORECASE,
                        ))
                        if marker_count > room_count:
                            missing_rooms.append(room_instance)
                    if missing_rooms:
                        correction = (
                            "Corrija a resposta anterior: o trecho contém ocorrências das salas "
                            + ", ".join(missing_rooms)
                            + ", mas elas não foram todas incluídas."
                        )

                response = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": (
                                f"Extraia SOMENTE as aulas do dia {day_name}.\n"
                                "Gere uma triplas para cada combinação de disciplina, período, docente, sala e horário indicada no trecho.\n"
                                "Não deixe de incluir todas as salas identificadas no trecho.\n"
                                f"{correction}\n"
                                f"Texto completo extraído do PDF:\n{pdf_text}\n\n"
                                f"Trecho que corresponde ao dia solicitado:\n{day_text}"
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=6000,
                )

                choice = response.choices[0]
                raw_output = (choice.message.content or "").strip()
                if not raw_output:
                    continue

                clean_insert = re.sub(
                    r"```(?:sparql)?\s*(.*?)\s*```",
                    r"\1",
                    raw_output,
                    flags=re.DOTALL,
                ).strip()
                insert_match = re.search(
                    r"\bINSERT\s+DATA\s*\{.*",
                    clean_insert,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if not insert_match:
                    continue

                block = insert_match.group(0).strip()
                opening_brace = block.find("{")
                closing_brace = block.rfind("}")
                if opening_brace < 0 or closing_brace <= opening_brace:
                    continue

                inner = block[opening_brace + 1:closing_brace].strip()
                if not inner:
                    continue

                if day_name == "sábado" and not re.search(
                    r":ocorreNoDia\s+:diaSab",
                    inner,
                    flags=re.IGNORECASE,
                ):
                    continue

                coverage = 0
                for marker_pattern, room_instance in room_markers:
                    marker_count = len(re.findall(marker_pattern, day_text, flags=re.IGNORECASE))
                    room_count = len(re.findall(
                        rf":ministraEm\s+{re.escape(room_instance)}",
                        inner,
                        flags=re.IGNORECASE,
                    ))
                    coverage += min(marker_count, room_count)

                if coverage > best_coverage:
                    best_inner = inner
                    best_coverage = coverage

            if day_name == "sábado":
                saturday_fallback = self.build_generic_saturday_aulas(day_text)
                if saturday_fallback:
                    inserts.append(saturday_fallback)
                    continue

            if best_inner:
                if day_name == "sábado":
                    saturday_fallback = self.build_generic_saturday_aulas(day_text)
                    if saturday_fallback:
                        inserts.append(saturday_fallback)
                    else:
                        parts = re.split(r"(?ms)(?=^:[A-Za-zÀ-ÿ0-9_]+\s+a\s+:Aula)", best_inner)
                        non_saturday_parts = [
                            part.strip()
                            for part in parts
                            if part.strip() and ":ocorreNoDia :diaSab" not in part
                        ]
                        if non_saturday_parts:
                            inserts.append("\n\n".join(non_saturday_parts))
                else:
                    inserts.append(best_inner)

        if not inserts:
            raise RuntimeError("A LLM não gerou nenhuma instrução INSERT DATA válida.")
        return "INSERT DATA {\n" + "\n\n".join(inserts) + "\n}"

    def populate_fuseki(self, insert_query: str) -> bool:
        """Executa a query INSERT DATA diretamente no Fuseki via endpoint /update."""
        try:
            # Adiciona os prefixos no topo caso a LLM omitiu
            full_query = f"{ONTOLOGY_PREFIX}\n{insert_query}"
            normalized = insert_query.strip()
            if not normalized.upper().startswith("INSERT DATA"):
                print("[X] A LLM não retornou uma instrução INSERT DATA válida.")
                return False

            replacement_query = f"""{ONTOLOGY_PREFIX}
DELETE {{
  ?aula ?propriedade ?valor .
}}
WHERE {{
  ?aula a :Aula ;
        ?propriedade ?valor .
}};

{full_query}
"""
            self.sparql_update.setQuery(replacement_query)
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
        return self.process_text(pdf_text)

    def process_text(self, pdf_text: str, status_callback=None):
        """Processa texto extraído pelo AnythingLLM ou pelo pdfplumber."""
        print("[2] Gerando triplas semânticas RDF com a LLM...")
        if status_callback:
            status_callback(12, "Preparando o texto extraído")
        insert_query = self.generate_triples_from_text(pdf_text, status_callback)
        print(f"\nSPARQL INSERT Gerado:\n{insert_query}\n")
        
        print("[3] Gravando instâncias no Jena Fuseki...")
        if status_callback:
            status_callback(82, "Validando e preparando triplas")
        success = self.populate_fuseki(insert_query)
        if success:
            print(" Ontologia povoada com sucesso a partir do PDF!")
            if status_callback:
                status_callback(100, "Ingestão concluída")
        return success