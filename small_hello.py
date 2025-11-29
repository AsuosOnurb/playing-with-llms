from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # any non-empty string; Ollama ignores it
)

resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Say hi in one short sentence, then a joke"}],
)

print(resp.choices[0].message.content)
