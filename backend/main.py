import sys
from llm_pipeline import LLMPipeline

def main():
    pipeline = LLMPipeline()
    
    # Se passar uma pergunta por linha de comando: python main.py "Sua pergunta"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        pipeline.run(question, verbose=True)
    else:
        # Loop interativo no terminal do VS Code
        print("=== Agente Semântico de Horários IFG (Groq + Fuseki) ===")
        print("Digite 'sair' para encerrar.\n")
        
        while True:
            try:
                user_input = input("Pergunte algo: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["sair", "exit", "quit"]:
                    break
                    
                pipeline.run(user_input, verbose=True)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()