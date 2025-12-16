# main.py
import os
import json
import base64
import requests
from typing import Optional, List, Dict

# pythonフレームワーク
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# LLM(gpt-oss-120b)用のライブラリ
from openai import OpenAI

# LlamaIndex(RAG構築フレームワーク)
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === 設定 ========================
speakerUuid = "58adbc32-a00a-11f0-ac61-7e5b44f22354"  # MYCOEIROINKのspeakerinfoで確認
styleId = 1043917874  # 該当UUIDフォルダのmeta.jsonに記載

client = OpenAI(
    base_url=os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1"),
    api_key="dummy_api_key",
)
model_name = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")

TTS_BASE_URL = os.getenv("MYCOEIROINK_URL", "http://127.0.0.1:50032")
KNOWLEDGE_JSON = os.getenv("KNOWLEDGE_JSON", "faqs.json")

rag_index: Optional[VectorStoreIndex] = None
rag_top_k_default = 3  # モデルが考慮する選択肢の候補(1～100)
rag_snippet_chars = 360  # スニペット長
max_tokens = 320  # 出力上限

llm_temperature = 0.2
client_timeout = 600  # 応答タイムアウト秒数
# ==================================

# 既定ペルソナ（必要なら system_prompt で上書き可能）
default_persona = (
    "あなたは米子工業高等専門学校の総合工学科・電気電子コースに所属する5年生です。"
    "名前は井東佳希(いとうよしき)です。"
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-LLM-Text-B64"],
)

# /static を配信
app.mount("/static", StaticFiles(directory="static"), name="static")


# json形式のナレッジを読み込み
def build_rag_index(json_path: str) -> Optional[VectorStoreIndex]:
    if not os.path.exists(json_path):
        print(f"[RAG] knowledge json not found: {json_path}")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("[RAG] failed to read json:", e)
        return None

    # 辞書リストとしてfaqsに格納する
    faqs: List[Dict[str, str]] = data.get("faqs", [])
    if not faqs:
        print("[RAG] faqs list empty")
        return None

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    documents: List[Document] = []
    # jsonからデータを抽出
    for i, item in enumerate(faqs, start=1):
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        # 検索対象の本文
        body = f"Q: {q}\nA: {a}"
        # メタデータ
        meta = {"source": os.path.basename(json_path), "faq_id": i, "question": q}
        documents.append(Document(text=body, metadata=meta))

    index = VectorStoreIndex.from_documents(documents)
    print(f"[RAG] index built with {len(documents)} docs from {json_path}")
    return index


# RAGでコンテキストを取得
def get_rag_context(query: str, top_k: int = rag_top_k_default) -> str:
    global rag_index
    if rag_index is None:
        return ""
    hits = rag_index.as_retriever(similarity_top_k=top_k).retrieve(query)
    lines = []
    for rank, hit in enumerate(hits, start=1):
        content = hit.node.get_content().strip()
        snippet = content[:rag_snippet_chars]
        src = hit.node.metadata.get("source", "")
        fid = hit.node.metadata.get("faq_id", "")
        # 見やすく整形
        lines.append(f"[{rank}] (source={src}#{fid})\n{snippet}")
    return "\n\n".join(lines)  # コンテキストを改行間隔で連結


def make_rag_system_instruction(context_text: str) -> str:
    if not context_text:
        return ""
    return (
        "以下は米子高専(米子工業高等専門学校)の理念・制度・学内規程・運用に関するナレッジベース（FAQ JSON）からの検索結果です。\n"
        "あなたの回答は、まずこの参照情報に厳密に基づき、事実に忠実に要点をまとめて文章にしてください。\n"
        "不明点がある場合は、推測せずに『その質問はわかりません』と述べた上で、関連しそうな情報を参照情報から補足してください。\n"
        "もし参照情報が長すぎる場合には3,4文程度に要約して文章として回答してください。\n"
        "箇条書きのような書き方を絶対に回答に含めないでください。\n"
        "「参照情報に」のような書き方を絶対に回答に含めないでください。\n"
        "コンテキスト情報に無い情報は絶対に回答に含めないでください。\n"
        "コンテキスト情報の内容を丸投げするのではなく、絶対にきちんとした文章にして回答してください。\n"
        "質問の答えを知らない場合は、誤った情報を共有しないでください。\n"
        f"【参照情報（上位候補）】\n{context_text}\n"
    )


class ChatIn(BaseModel):
    message: str
    system_prompt: Optional[str] = None


def call_llm_with(user_text: str, system_prompt: Optional[str] = None) -> str:
    convo: List[Dict[str, str]] = []

    # 1) ペルソナ/system（指定がなければ既定ペルソナ）
    primary_system = system_prompt or default_persona
    if primary_system:
        convo.append({"role": "system", "content": primary_system})

    # 2) RAG system instruction（ある場合）
    rag_context = get_rag_context(user_text, top_k=rag_top_k_default)
    rag_sys = make_rag_system_instruction(rag_context)
    if rag_sys:
        convo.append({"role": "system", "content": rag_sys})

    # 3) user
    convo.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model=model_name,
        messages=convo,
        timeout=client_timeout,
        max_tokens=max_tokens,
        temperature=llm_temperature,
    )
    reply = (resp.choices[0].message.content or "").strip()
    return reply


def generate_my_voice(text: str) -> bytes:
    text += 'ーー'
    query = {
        "speakerUuid": speakerUuid,
        "styleId": styleId,
        "text": text,
        "speedScale": 1.0,
        "volumeScale": 1.0,
        "prosodyDetail": [],
        "pitchScale": 0.0,
        "intonationScale": 1.0,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.5,
        "outputSamplingRate": 24000,
    }
    r = requests.post(f"{TTS_BASE_URL}/v1/synthesis", json=query, timeout=60)
    if r.status_code >= 400:
        print("TTS /v1/synthesis error:", r.status_code, r.text[:1000])
    r.raise_for_status()    
    return r.content


# WAV を直接返すAPI（テキストは base64 でヘッダに入れる）
@app.post("/chat_tts_wav")
def chat_tts_wav(inp: ChatIn):
    answer = call_llm_with(inp.message, system_prompt=inp.system_prompt)
    wav_bytes = generate_my_voice(answer)
    answer_b64 = base64.b64encode(answer.encode("utf-8")).decode("ascii")  # ASCII のみ
    headers = {"X-LLM-Text-B64": answer_b64}
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


# （任意）フィードバック受け取りエンドポイント
class FeedbackIn(BaseModel):
    user_message: str
    assistant_text: str
    rating: str  # "up" / "down" / "none"
    comment: str = ""


@app.post("/feedback")
def feedback(inp: FeedbackIn):
    print(
        f"[FB] rating={inp.rating} user='{inp.user_message[:60]}' asst='{inp.assistant_text[:60]}' comment='{inp.comment[:60]}'"
    )
    return {"status": "ok"}


# RAGインデックスの再読込API
@app.post("/reload_rag")
def reload_rag():
    global rag_index
    rag_index = build_rag_index(KNOWLEDGE_JSON)
    return {"status": "ok", "index_ready": rag_index is not None}


# ルートは static/index.html を返す
@app.get("/")
def root():
    return FileResponse("static/index.html")

# ===========================
@app.on_event("startup")
def _startup():
    # 起動時にRAGインデックス構築
    global rag_index
    rag_index = build_rag_index(KNOWLEDGE_JSON)
    if rag_index is None:
        print("[RAG] WARN: RAG index not ready. Set KNOWLEDGE_JSON or place faqs.json.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)