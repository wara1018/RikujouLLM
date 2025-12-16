import os
import json
import requests
import simpleaudio
text =  "ぎもん、しつもん、どらえもん。。"

query = {
	"speakerUuid": "58adbc32-a00a-11f0-ac61-7e5b44f22354",
	"styleId": 1043917874,
	"text": text,
	"speedScale": 1.0,
	"volumeScale": 1.0,
	"prosodyDetail": [],
	"pitchScale": 0.0,
	"intonationScale": 1.0,
	"prePhonemeLength": 0.1,
	"postPhonemeLength": 0,
	"outputSamplingRate": 24000,
	"endTrimBuffer": 1.0,
}

#APIから音声合成を実行する
response = requests.post(
    "http://127.0.0.1:50032/v1/synthesis",
    headers={"Content-Type": "application/json"},
    data=json.dumps(query),
)

response.raise_for_status()

with open("audio.wav", "wb") as f_temp:
	f_temp.write(response.content)
 
wav_obj = simpleaudio.WaveObject.from_wave_file("audio.wav")
play_obj = wav_obj.play()
play_obj.wait_done()
 
 