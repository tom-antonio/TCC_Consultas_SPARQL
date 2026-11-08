import json
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET
from config import FUSEKI_ADMIN_PASSWORD, FUSEKI_USER

class FusekiSparql:
    def __init__(self, endpoint_url: str = "http://localhost:3030/TADS_2026_2"):
        """
        Inicializa a conexão com o endpoint SPARQL do Apache Jena Fuseki.
        O endpoint pode ser substituído pela variável FUSEKI_QUERY_URL.
        """
        self.endpoint_url = endpoint_url
        self.sparql = SPARQLWrapper(endpoint_url)
        if FUSEKI_ADMIN_PASSWORD:
            self.sparql.setCredentials(FUSEKI_USER, FUSEKI_ADMIN_PASSWORD)

    def execute_query(self, query: str) -> dict:
        """
        Executa uma consulta SPARQL (SELECT) no Fuseki e devolve o resultado em dicionário Python/JSON.
        """
        try:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(JSON)
            self.sparql.setMethod(GET)
            
            results = self.sparql.query().convert()
            return {
                "success": True,
                "data": results["results"]["bindings"],
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "data": [],
                "error": f"Erro ao executar a consulta no Fuseki: {str(e)}"
            }

    def format_bindings_to_clean_json(self, bindings: list) -> list:
        """
        Limpa a resposta padrão do Fuseki/SPARQL, extraindo apenas os valores simplificados.
        """
        clean_results = []
        for row in bindings:
            clean_row = {}
            for var_name, var_data in row.items():
                # Extrai o valor do nó RDF (literal ou URI)
                clean_row[var_name] = var_data.get("value", "")
            clean_results.append(clean_row)
        return clean_results

# Exemplo rápido para teste manual do módulo (executar diretamente no terminal)
if __name__ == "__main__":
    client = FusekiSparql()
    
    # Consulta de teste para verificar se o Fuseki está ativo e a responder
    test_sparql = """
    PREFIX : <http://www.semanticweb.org/luan/ontologies/2026/8/untitled-ontology-6#>
    SELECT ?professor ?nomeProfessor WHERE {
        ?professor a :Professor ;
                   :nomeProfessor ?nomeProfessor .
    } LIMIT 5
    """
    
    print("Testando conexão com a Triplestore Fuseki...")
    response = client.execute_query(test_sparql)
    
    if response["success"]:
        data_limpa = client.format_bindings_to_clean_json(response["data"])
        print("Conexão estabelecida com sucesso!")
        print(json.dumps(data_limpa, indent=2, ensure_ascii=False))
    else:
        print(response["error"])