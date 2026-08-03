"""
chat.py
Terminal uzerinden RAG sistemini hizlica test etmek icin basit bir sohbet
dongusu. Hicbir is mantigi icermez -- sadece answer_question() cagirir.
"""
from src.llm import answer_question

EXIT_COMMANDS = {"exit", "quit", "q"}


def main() -> None:
    print("=" * 40)
    print("LASC Assistant")
    print("=" * 40)

    while True:
        question = input("\nYou:\n> ").strip()

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("\nGoodbye!")
            break

        result = answer_question(question)

        print("\nAssistant:")
        print(result["answer"])

        print("\nSources:")
        for src in result["sources"]:
            print(f"- {src['file_name']} | Page {src['page']} | Chunk {src['chunk_index']}")


if __name__ == "__main__":
    main()