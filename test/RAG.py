# main.py
# ============================================
# RAG統合版（LlamaIndex + JSON FAQナレッジ）
# 事前準備:
#   pip install "llama-index>=0.10.0" "llama-index-embeddings-huggingface>=0.2.0" "sentence-transformers>=2.6.0"
#   （HuggingFaceのembeddingsを利用します。GPUは不要。）
# 環境変数（任意）:
#   KNOWLEDGE_JSON: 参照するFAQ JSONファイルのパス（デフォルト: ./faqs.json）
#   LMSTUDIO_URL:   LM StudioのOpenAI互換エンドポイント（例: http://127.0.0.1:1234/v1）
#   LMSTUDIO_MODEL: 使用するモデル名（例: openai/gpt-oss-120b）
#   MYCOEIROINK_URL: MyCoeiroInk TTSのエンドポイント（例: http://127.0.0.1:50032）
# ============================================

import os
import json
import base64
import requests
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
import uvicorn

from openai import OpenAI

# === LlamaIndex (RAG) ========================
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# ============================================

# === 設定 ========================
speakerUuid = "58adbc32-a00a-11f0-ac61-7e5b44f22354"  # MYCOEIROINKのspeakerinfoで確認
styleId = 1043917874  # 該当UUIDフォルダのmeta.jsonファイルに記載

client = OpenAI(
    base_url=os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1"),
    api_key="dummy_api_key"
)
model_name = os.getenv("LMSTUDIO_MODEL", "openai/gpt-oss-20b")

TTS_BASE_URL = os.getenv("MYCOEIROINK_URL", "http://127.0.0.1:50032")
KNOWLEDGE_JSON = os.getenv("KNOWLEDGE_JSON", "faqs.json")  # 先ほど出力させたQA（JSON）へのパス
#==================================

# === グローバル（RAG） =======================
RAG_INDEX: Optional[VectorStoreIndex] = None
RAG_TOP_K_DEFAULT = 3
RAG_SNIPPET_CHARS = int(os.getenv("RAG_SNIPPET_CHARS", "360"))
# ============================================

# 先頭の設定付近に追加（環境変数で調整可）
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "320"))    # 出力上限
def getenv_float(name: str, default: float) -> float:
    v = os.getenv(name, None)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        print(f"[WARN] Env {name}='{v}' is not a float. Using default={default}.")
        return default

LLM_TEMPERATURE = getenv_float("LLM_TEMPERATURE", 0.2)  # 衝突しない名前に変更
CLIENT_TIMEOUT = int(os.getenv("CLIENT_TIMEOUT", "600"))  # 120→600 に延長

messages = [
    {
        "role": "system",
        "content": (
            "あなたは米子工業高等専門学校の総合工学科・電気電子コースに所属する5年生です。"
            "名前は井東佳希(いとうよしき)です。あなたの趣味は神社仏閣巡りで、好きな寺社は永平寺、四天王寺です。"
            "あなたは2年生のころまで放送部に所属しており、ドラマやドキュメントを制作していましたが現在は退部し、"
            "コンピュータ同好会と数学同好会に所属しています。"
        )
    }
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-LLM-Text-B64"],
)

# ========= RAG: JSON -> Index 構築 ===========
def build_rag_index_from_json(json_path: str) -> Optional[VectorStoreIndex]:
    """
    JSON形式のFAQナレッジを読み込み、LlamaIndexのVectorStoreIndexを構築する。
    期待フォーマット:
      {
        "faqs": [
          {"question": "...", "answer": "..."},
          ...
        ]
      }
    """
    if not os.path.exists(json_path):
        print(f"[RAG] knowledge json not found: {json_path}")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[RAG] json load error: {e}")
        return None

    faqs: List[Dict[str, str]] = data.get("faqs", [])
    if not faqs:
        print("[RAG] faqs list empty")
        return None

    # Embeddingモデル設定（HuggingFace）
    # 軽量かつ精度バランスの良い all-MiniLM-L6-v2 を使用
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    documents: List[Document] = []
    for i, item in enumerate(faqs, start=1):
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        # 検索対象の本文（QとAを併記）
        body = f"Q: {q}\nA: {a}"
        # メタデータ（後で出典表記などにも使える）
        meta = {"source": os.path.basename(json_path), "faq_id": i, "question": q}
        documents.append(Document(text=body, metadata=meta))

    index = VectorStoreIndex.from_documents(documents)
    print(f"[RAG] index built with {len(documents)} docs from {json_path}")
    return index


def get_rag_context(query: str, top_k: int = RAG_TOP_K_DEFAULT) -> str:
    global RAG_INDEX
    if RAG_INDEX is None:
        return ""
    hits = RAG_INDEX.as_retriever(similarity_top_k=top_k).retrieve(query)
    lines = []
    for rank, hit in enumerate(hits, start=1):
        content = hit.node.get_content().strip()
        # Aの要点だけを抜くならここで簡易抽出してもOK
        snippet = content[:RAG_SNIPPET_CHARS]
        src = hit.node.metadata.get("source", "")
        fid = hit.node.metadata.get("faq_id", "")
        lines.append(f"[{rank}] (source={src}#{fid})\n{snippet}")
    return "\n\n".join(lines)


def make_rag_system_instruction(context_text: str) -> str:
    if not context_text:
        return ""
    return (
        "以下は米子高専(米子工業高等専門学校)の理念・制度・学内規程・運用に関するナレッジベース（FAQ JSON）からの検索結果です。\n"
        "あなたの回答は、まずこの参照情報に厳密に基づき、事実に忠実に要点をまとめて文章にしてください。\n"
        "不明点がある場合は、推測せずに『その質問はわからん』と述べた上で、関連しそうな情報を参照情報から補足してください。\n"
        "もし参照情報が長すぎる場合には2、3文程度に要約して文章として回答してください。"
        "-箇条書きのような書き方を絶対に回答に含めないでください。\n"
        "「参照情報に」のような書き方を絶対に回答に含めないでください。"
        "- コンテキスト情報に無い情報は絶対に回答に含めないでください。\n"
        "- コンテキスト情報の内容を丸投げするのではなく、絶対にきちんとした文章にして回答してください。\n"
        "- 質問の答えを知らない場合は、誤った情報を共有しないでください。\n"
        f"【参照情報（上位候補）】\n{context_text}\n"
    )
# ============================================


class ChatIn(BaseModel):
    message: str
    system_prompt: Optional[str] = None


def call_llm_with(user_text: str, system_prompt: Optional[str] = None) -> str:
    # ステートレス（毎回最小構成）
    convo = []

    # 簡略化したペルソナ（1〜2文）
    base_persona = "あなたは米子工業高等専門学校の学生として、簡潔かつ事実に基づき回答します。"
    convo.append({"role": "system", "content": base_persona})

    # RAG
    rag_context = get_rag_context(user_text, top_k=3)  # 5→3
    rag_sys = make_rag_system_instruction(rag_context)
    if rag_sys:
        convo.insert(0, {"role": "system", "content": rag_sys})

    if system_prompt:
        convo.insert(0, {"role": "system", "content": system_prompt})

    convo.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
    model=model_name,
    messages=convo,
    timeout=CLIENT_TIMEOUT,      # 既存
    max_tokens=MAX_TOKENS,       # 既存
    temperature=LLM_TEMPERATURE, # ここを変更
    )
    reply = (resp.choices[0].message.content or "").strip()
    return reply

def generate_my_voice(text: str) -> bytes:
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


# （任意）フィードバック受け取りエンドポイント（UIから呼ばれる）
class FeedbackIn(BaseModel):
    user_message: str
    assistant_text: str
    rating: str
    comment: str = ""


@app.post("/feedback")
def feedback(inp: FeedbackIn):
    # 必要に応じてログ保存や学習用蓄積を実装
    print(f"[FB] rating={inp.rating} user='{inp.user_message[:60]}' asst='{inp.assistant_text[:60]}'")
    return {"status": "ok"}


# RAGインデックスの再読込API（ナレッジ差し替え時に使用）
@app.post("/reload_rag")
def reload_rag():
    global RAG_INDEX
    RAG_INDEX = build_rag_index_from_json(KNOWLEDGE_JSON)
    return {"status": "ok", "index_ready": RAG_INDEX is not None}


# スマホでも使える最小UI
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse("""
<!doctype html>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body { font-family: system-ui, sans-serif; margin: 1rem; }
  #chat { border: 1px solid #ccc; border-radius: 8px; padding: .5rem; max-height: 60vh; overflow:auto; }
  .msg { margin: .5rem 0; }
  .user { color: #0b5; }
  .assistant { color: #06c; }
  .row { display:flex; gap:.5rem; margin-top:.5rem; }
  button { padding:.5rem 1rem; }
  #player { width:100%; margin-top:.5rem; }
</style>
<h3>Chat + TTS (RAG enabled)</h3>
<div id="chat"></div>
<div class="row">
  <input id="msg" placeholder="メッセージを入力" style="flex:1; padding:.5rem;" />
  <button id="send">送信</button>
</div>
<audio id="player" controls></audio>
<div id="fb" style="display:none; margin-top:.5rem;">
  フィードバック:
  <button id="up">👍</button>
  <button id="down">👎</button>
</div>
<script>
  const chat = document.getElementById('chat');
  const msg = document.getElementById('msg');
  const send = document.getElementById('send');
  const player = document.getElementById('player');
  const fb = document.getElementById('fb');
  const up = document.getElementById('up');
  const down = document.getElementById('down');

  let lastUser = "";
  let lastAssistant = "";

  function addMsg(text, cls) {
    const div = document.createElement('div');
    div.className = 'msg ' + cls;
    div.textContent = (cls === 'user' ? 'あなた: ' : 'アシスタント: ') + text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  function b64ToUtf8(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  }

  async function sendChat() {
    const text = msg.value.trim();
    if (!text) return;
    msg.value = "";
    addMsg(text, 'user');
    lastUser = text;

    const res = await fetch('/chat_tts_wav', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });

    const b64 = res.headers.get('X-LLM-Text-B64') || '';
    const answer = b64 ? b64ToUtf8(b64) : '';
    lastAssistant = answer;
    addMsg(answer, 'assistant');

    const blob = await res.blob();
    player.src = URL.createObjectURL(blob);
    try { await player.play(); } catch(e) {}
    fb.style.display = 'block';
  }

  send.onclick = sendChat;
  msg.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendChat(); });

  async function sendFeedback(rating) {
    if (!lastAssistant) return;
    await fetch('/feedback', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        user_message: lastUser,
        assistant_text: lastAssistant,
        rating, comment: ""
      })
    });
    alert('フィードバックを送信しました。ありがとうございます。');
  }
  up.onclick = () => sendFeedback('up');
  down.onclick = () => sendFeedback('down');
</script>
    """)

# ===========================
@app.on_event("startup")
def _startup():
    # 起動時にRAGインデックス構築
    global RAG_INDEX
    RAG_INDEX = build_rag_index_from_json(KNOWLEDGE_JSON)
    if RAG_INDEX is None:
        print("[RAG] WARN: RAG index not ready. Set KNOWLEDGE_JSON or place faqs.json.")

if __name__ == "__main__":
    # スマホからも使うなら host=0.0.0.0
    uvicorn.run(app, host="0.0.0.0", port=8000)