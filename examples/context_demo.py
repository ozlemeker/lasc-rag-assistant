import re
import subprocess
from openai import OpenAI

def get_foundry_base_url():
    result = subprocess.run(
        ["foundry", "service", "status"],
        capture_output=True, text=True, check=True
    )
    match = re.search(r"http://[\d.]+:(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Foundry servisinin portu bulunamadı:\n{result.stdout}")
    port = match.group(1)
    return f"http://127.0.0.1:{port}/v1"

client = OpenAI(
    base_url=get_foundry_base_url(),
    api_key="not-needed"
)

MODEL = "Phi-3.5-mini-instruct-generic-cpu:2"


def ask_model(question, context=None):
    messages = []

    if context:
        messages.append({
            "role": "system",
            "content": f"You must answer ONLY using the following context.\n\nContext:\n{context}"
        })

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

context = """
Microsoft Fabric is Microsoft's unified analytics platform.

It combines data engineering,
data science,
data warehousing,
real-time analytics,
and business intelligence
into a single platform.
"""

question = "What is Microsoft Fabric?"

#withoıut context
print("=" * 60)
print("WITHOUT CONTEXT")
print("=" * 60)

answer = ask_model(question)

print(answer)


#with context
print("\n")
print("=" * 60)
print("WITH CONTEXT")
print("=" * 60)

answer = ask_model(question, context)

print(answer)