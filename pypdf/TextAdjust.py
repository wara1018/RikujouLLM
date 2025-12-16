from openai import OpenAI

textname = '1_本校の目的'

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="dummy_api_key" # api_keyはダミーなので何でもよい
)

MODEL_NAME = "openai/gpt-oss-20b"

# 最初のシステムプロンプト（会話の前提や制約。内容は増やせます。）
messages = [
    {
        "role": "system",
        "content": (
            "あなたはテキスト整形の専門家です。ユーザーから与えられたテキストを整形してください。与えられたテキストはPDFからOCRで抽出したテキストのため、矢印などの記号がなくなっている場合がありますが、適宜補完してください。また、整形したテキスト以外は応答に含めないでください。"
        )
    }
]

while True:
    user_input = input("YOU > ").strip()
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )

    reply = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": reply})
    
    f = open(f'{textname}.txt','w', encoding='UTF-8')
    f.writelines(reply)
    f.close


    
