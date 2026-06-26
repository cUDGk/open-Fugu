from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key="local-frontdoor-key",
)

response = client.chat.completions.create(
    model="local-moe-fugu:deep",
    messages=[
        {
            "role": "user",
            "content": "Ryzen 9 7940HS / 64GB RAMでMoEモデル3体を逐次切替するFugu風構成を短く設計して。",
        }
    ],
)

print(response.choices[0].message.content)
print(response.system_fingerprint)
