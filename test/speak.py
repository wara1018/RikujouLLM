import json
import requests
from pydub import AudioSegment, playback


text = "米子高専をぶっ壊す！。"
speaker_id = 1043917874

#音声合成のクエリ作成
response = requests.post(
    "http://localhost:50032/audio_query",
    params={
        "text": text,
        "speaker": speaker_id,
        "core_version": "0.0.0"
    })
query = response.json()

#音声合成wavを生成する
response = requests.post(
    "http://localhots:50032/"
)