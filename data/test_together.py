from together import Together

# Initialize the Together client (uses TOGETHER_API_KEY from env)
client = Together()

# Make a minimal test request to the model
resp = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
    max_tokens=64,        # keep it short to reduce latency
    temperature=0.1       # deterministic-ish output
)

print(resp.choices[0].message.content)
