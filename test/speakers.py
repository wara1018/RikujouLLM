import requests
import json

BASE_URL = "http://127.0.0.1:50032"  # ドキュメント通り
# 例: VOICEVOX互換APIなら 50021 や /speakers のこともある
# BASE_URL = "http://127.0.0.1:50021"

url = f"{BASE_URL}/v1/speakers"
r = requests.get(url, timeout=5)
r.raise_for_status()
speakers = r.json()

print(json.dumps(speakers, ensure_ascii=False, indent=2))