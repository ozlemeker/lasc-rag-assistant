"""
prompt.py
Soru ve context'ten, LLM'in (Phi-3.5) kullanacagi system/user prompt'larini
uretir. Bu modul LLM'i cagirmaz -- sadece metin uretir.
"""

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer the user's question ONLY using the provided context.

If the answer cannot be found in the context,
respond exactly:

"I don't know."

Do not make up facts.
Do not use outside knowledge.
""".strip()

USER_PROMPT_TEMPLATE = """
Context:
{context}

Question:
{question}

Answer:
""".strip() + "\n"


def build_user_prompt(question: str, context: str) -> str:
    """Context ve soruyu, kullanici mesaji olarak formatlar.
    .format() yerine .replace() kullanilir; context'te { veya } karakteri
    gecerse .format() hata firlatabilir."""
    return (
        USER_PROMPT_TEMPLATE
        .replace("{context}", context)
        .replace("{question}", question)
    )


if __name__ == "__main__":
    from src.retrieval import retrieve

    question = "What is the maximum rocket mass?"
    context = retrieve(question)

    user_prompt = build_user_prompt(question, context)

    print("--- SYSTEM ---")
    print(SYSTEM_PROMPT)
    print("\n--- USER ---")
    print(user_prompt)