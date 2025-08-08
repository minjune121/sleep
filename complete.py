from flask import Flask, request, jsonify, send_file, render_template
import io
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydub import AudioSegment
from google.cloud import texttospeech, speech
from google.oauth2 import service_account
import numpy as np
import pyaudio
import re
import os
import json
import random

app = Flask(__name__)

# ====== 환경설정 ======
GOOGLE_CREDS_PATH = "C:/Temp/SleepVoice/STTS.json" #구글 API
HUGGINGFACE_MODEL = "minjune121/my-kogpt-finetuned" #만든 모델

#조절절
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = int(RATE / 10)
SILENCE_LIMIT = 1
THRESHOLD = 500

# ====== 인증 및 모델 로딩 ======
credentials = service_account.Credentials.from_service_account_file(GOOGLE_CREDS_PATH)
speech_client = speech.SpeechClient(credentials=credentials)
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

# LLM 모델 로딩
hf_tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
hf_model = AutoModelForCausalLM.from_pretrained(HUGGINGFACE_MODEL)
hf_model.eval()
if torch.cuda.is_available():
    hf_model = hf_model.to("cuda")

# ====== 함수 정의 ======
def record_until_silence():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames, silence_counter = [], 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_data).mean()
        silence_counter = silence_counter + 0.1 if volume < THRESHOLD else 0
        if silence_counter > SILENCE_LIMIT:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b"".join(frames)

def audio_to_wav(audio_bytes, format="webm"):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format)
    audio_segment = audio_segment.set_frame_rate(16000)       # 샘플레이트 16000Hz
    audio_segment = audio_segment.set_sample_width(2)         # 샘플 폭 2바이트 (16비트)
    audio_segment = audio_segment.set_channels(1)             # 모노로 강제
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)
    return buffer



def transcribe_audio(audio_buffer):
    audio = speech.RecognitionAudio(content=audio_buffer.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ko-KR"
    )
    response = speech_client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else ""

def generate_llm_response(prompt):
    input_ids = hf_tokenizer(prompt, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    with torch.no_grad():
        output_ids = hf_model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=hf_tokenizer.eos_token_id
        )
    return hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def split_sentences(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

async def synthesize_tts(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")

async def generate_tts_audio(text):
    sentences = split_sentences(text)
    audio_segments = [await synthesize_tts(s) for s in sentences]
    combined = sum(audio_segments[1:], start=audio_segments[0]) if len(audio_segments) > 1 else audio_segments[0]
    buffer = io.BytesIO()
    combined.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

def get_random_story_from_file(json_filepath):
    with open(json_filepath, "r", encoding="utf-8") as f:
        stories = json.load(f)
    return random.choice(stories)

# ====== 라우터 ======
@app.route("/")
def index():
    return render_template("index.html")

from flask import jsonify
import base64

#새 창에서 음성 대화 할 수 있게 함
@app.route("/chat")
def chat():
    return render_template("chat.html")

#수면 친구 AI
@app.route("/voice", methods=["POST"])
def process_voice():
    if 'audio' not in request.files:
        return "오디오 파일이 없습니다.", 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    wav_buffer = audio_to_wav(audio_bytes, format='webm')

    user_text = transcribe_audio(wav_buffer)
    prompt = f"사용자: {user_text}\n답변:"
    bot_response = generate_llm_response(prompt)
    clean_response = bot_response.split("답변:")[-1].strip()

    tts_audio = asyncio.run(generate_tts_audio(clean_response))
    audio_base64 = base64.b64encode(tts_audio.read()).decode("utf-8")

    return jsonify({
        "user_text": user_text,
        "bot_response": clean_response,
        "audio_base64": audio_base64
    })


#음성 테스트
@app.route("/story-audio")
def stream_story_audio():
    story = get_random_story_from_file("C:/Temp/SleepVoice/book_sleep.json")
    text = story['text']
    audio_buffer = asyncio.run(generate_tts_audio(text))
    return send_file(
        audio_buffer,
        mimetype="audio/mpeg",
        as_attachment=False,
        download_name="story.mp3"
    )

#음성 재생
@app.route("/story")
def story():
    story = get_random_story_from_file("C:/Temp/SleepVoice/book_sleep.json")
    text = story['text']
    title = story['title']

    # TTS 음성 생성 및 저장
    audio_path = os.path.join('static', 'story_audio.mp3')
    audio_buffer = asyncio.run(generate_tts_audio(text))
    with open(audio_path, "wb") as f:
        f.write(audio_buffer.read())

    return render_template("story.html", title=title, text=text, audio_file='story_audio.mp3')


if __name__ == "__main__":
    app.run(debug=True)

