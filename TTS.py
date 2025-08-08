import asyncio
import io
import re
from google.cloud import texttospeech
from google.oauth2 import service_account
from pydub import AudioSegment

# 🔐 1. 서비스 계정 인증
credentials = service_account.Credentials.from_service_account_file("C:/Temp/SleepVoice/STTS.json")
client = texttospeech.TextToSpeechClient(credentials=credentials)

# 🔠 2. 문장 분할 함수
def split_sentences(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

# 🔊 3. TTS: 한 문장 합성 함수 (비동기)
async def synthesize_tts(text):
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
    speaking_rate=0.5,   # 속도 느리게
    pitch=-20.0            # 음높이 낮게
)


    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return AudioSegment.from_file(io.BytesIO(response.audio_content), format="mp3")

# 📦 4. 여러 문장을 하나의 오디오로 합친 후 메모리로 반환
async def generate_tts_audio(text):
    sentences = split_sentences(text)
    audio_segments = []

    for sentence in sentences:
        audio = await synthesize_tts(sentence)
        audio_segments.append(audio)

    combined = sum(audio_segments)
    buffer = io.BytesIO()
    combined.export(buffer, format="mp3")
    buffer.seek(0)
    return buffer

# ✅ 5. 테스트 실행용 코드 (선택)
if __name__ == "__main__":
    sample_text = "안녕하세요. 잠이 안 오시나요? 좋은 꿈 꾸세요!"
    buffer = asyncio.run(generate_tts_audio(sample_text))
    # 저장 테스트 (옵션)
    with open("test_combined.mp3", "wb") as f:
        f.write(buffer.read())
    print("✅ test_combined.mp3 저장 완료")
