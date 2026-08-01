"""
llm.py
Microsoft Foundry Local uzerinde calisan Phi-3.5 Mini modelini kullanarak
RAG context'ine dayali cevap uretir.

Baglanti deseni onceden dogrulanmis calisan koda (context_demo.py) dayanir:
FoundryLocalManager kullanilmaz, servis portu `foundry service status`
komutuyla bulunur ve OpenAI client'i buna gore kurulur.
"""
import re
import subprocess

from openai import OpenAI
from sympy import content

from src.prompt import SYSTEM_PROMPT, build_user_prompt
from src.retrieval import retrieve

MODEL = "Phi-3.5-mini-instruct-generic-cpu:2"

_client: OpenAI | None = None


def _get_foundry_base_url() -> str:
    """Foundry Local servisinin calistigi portu `foundry service status`
    ciktisindan okur ve OpenAI-uyumlu base_url'i dondurur."""
    result = subprocess.run(
        ["foundry", "service", "status"],
        capture_output=True, text=True, check=True,
    )
    match = re.search(r"http://[\d.]+:(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Foundry servisinin portu bulunamadi:\n{result.stdout}")
    port = match.group(1)
    return f"http://127.0.0.1:{port}/v1"


def _get_client() -> OpenAI:
    """OpenAI client'ini ilk cagrida kurar, sonraki cagrilarda ayni
    instance'i dondurur."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=_get_foundry_base_url(),
            api_key="not-needed",
        )
    return _client


def answer_question(question: str) -> str:
    """
    Soruyu retrieve() ile ilgili context'e baglar, SYSTEM_PROMPT ve
    build_user_prompt() ile mesaj listesi olusturur, Phi-3.5 Mini'ye
    gonderir ve modelin cevabini string olarak dondurur.
    """
    context = retrieve(question)
    user_prompt = build_user_prompt(question, context)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content

    if content is None:
        raise RuntimeError("Model returned an empty response.")

    return content


if __name__ == "__main__":
    question = "What is the maximum rocket mass?"
    answer = answer_question(question)

    print("Question:")
    print(question)
    print("\nAnswer:")
    print(answer)