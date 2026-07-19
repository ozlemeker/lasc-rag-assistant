from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:51842/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Phi-3.5-mini-instruct-generic-cpu:2",
    messages=[{"role": "user", "content": "Say only: Hello World"}]
)

print(response.choices[0].message.content)