import os
import io
import pyaudio
import numpy as np
from pydub import AudioSegment
from google.cloud import speech
from google.oauth2 import service_account

# ===== 설정 =====
json_file = r"C:\Temp\SleepVoice\STTS.json"  # 서비스 계정 키 경로
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = int(RATE / 10)  # 0.1초 단위
SILENCE_LIMIT = 2  # 무음 지속 시간 (초)
THRESHOLD = 500  # 소리 감지 임계값 (값이 낮을수록 작은 소리도 인식)

# ===== 인증 =====
credentials = service_account.Credentials.from_service_account_file(json_file)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_file
client = speech.SpeechClient(credentials=credentials)

def record_until_silence():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("🎙 말하세요... (무음이 감지되면 자동 종료)")
    frames = []
    silence_counter = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        # numpy 배열로 변환하여 소리 크기 측정
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_data).mean()

        if volume < THRESHOLD:
            silence_counter += 0.1
        else:
            silence_counter = 0  # 소리가 들리면 카운터 초기화

        if silence_counter > SILENCE_LIMIT:
            print("⏹ 무음 감지, 녹음 종료")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b"".join(frames)

def audio_to_wav(audio_bytes):
    audio_segment = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=RATE,
        channels=CHANNELS
    )
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
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        print("🗣 인식된 텍스트:", result.alternatives[0].transcript)

if __name__ == "__main__":
    audio_bytes = record_until_silence()
    wav_buffer = audio_to_wav(audio_bytes)
    transcribe_audio(wav_buffer)
