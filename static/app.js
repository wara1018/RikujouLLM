const chat = document.getElementById('chat');
const msg = document.getElementById('msg');
const send = document.getElementById('send');
const player = document.getElementById('player');

let lastUser = "";
let lastAssistant = "";

function addMsg(text, cls) {
  const role = (cls || '').trim() === 'user' ? 'user' : 'assistant';
  const div = document.createElement('div');
  div.className = 'msg ' + role;

  const safeText = (text ?? '').toString();
  div.textContent = (role === 'user' ? 'あなた：' : 'AI：') + (safeText || '（空）');

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;

  console.log('[addMsg]', { role, text: safeText });
}

function b64ToUtf8(b64) {
  try {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes);
  } catch (e) {
    console.error('b64 decode error', e);
    return '';
  }
}

async function sendChat() {
  const text = msg.value.trim();
  if (!text) return;
  msg.value = "";
  addMsg(text, 'user');
  lastUser = text;

  let res;
  try {
    res = await fetch('/chat_tts_wav', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
  } catch (e) {
    console.error('fetch error', e);
    addMsg(`ネットワークエラー: ${e?.message || e}`, 'assistant');
    return;
  }

  if (!res.ok) {
    addMsg(`エラー: ${res.status}`, 'assistant');
    return;
  }

  // 先にテキスト（ヘッダー＞本文フォールバック）
  const b64 = res.headers.get('X-LLM-Text-B64') || '';
  let answer = b64 ? b64ToUtf8(b64) : '';

  if (!answer) {
    try {
      const clone = res.clone();
      const ct = clone.headers.get('Content-Type') || '';
      if (ct.includes('application/json')) {
        const data = await clone.json();
        if (data && typeof data.text === 'string') answer = data.text;
      }
    } catch (_) { /* JSON でないので無視 */ }
  }

  lastAssistant = answer;
  addMsg(answer, 'assistant');

  // WAV を最後に
  try {
    const blob = await res.blob();
    player.src = URL.createObjectURL(blob);
    try { await player.play(); } catch (e) { /* ユーザー操作待ち等 */ }
  } catch (e) {
    console.error('blob/play error', e);
  }
}

send.onclick = sendChat;
msg.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendChat(); });