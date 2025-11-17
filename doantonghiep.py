import os
import sys
import json
import wave
import subprocess
import time
import threading
import logging
import gc
import RPi.GPIO as GPIO
import pyaudio
from vosk import Model, KaldiRecognizer
from transformers import MarianMTModel, MarianTokenizer
from piper import PiperVoice

# =============================
# ⚙️ LOGGING SETUP
# =============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================
# ⚙️ CẤU HÌNH GPIO
# =============================
PIN_MODE = 27   # LOW=Việt->Anh, HIGH=Anh->Việt
PIN_BUTTON = 17 # Nút thu âm

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_MODE, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(PIN_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# =============================
# ⚙️ FILE GHI ÂM
# =============================
WAV_FILE = "/home/acer/input.wav"
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = "S16_LE"

# =============================
# 🔹 LOAD MODEL 1 LẦN (VỚI KIỂM TRA)
# =============================
def load_models():
    try:
        logging.info("🔹 Loading Vosk small models (Vi + En)...")
        if not os.path.exists("/home/acer/vosk_models/vosk-model-vn-0.4"):
            raise FileNotFoundError("Vosk Vi model not found")
        vosk_vi = Model("/home/acer/vosk_models/vosk-model-vn-0.4")
        if not os.path.exists("/home/acer/vosk_models/vosk-model-en-us-0.22-lgragh"):
            raise FileNotFoundError("Vosk En model not found")
        vosk_en = Model("/home/acer/vosk_models/vosk-model-en-us-0.22-lgraph")

        logging.info("🔹 Loading MarianMT small models...")
        vi2en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
        vi2en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
        en2vi_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        en2vi_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi")

        logging.info("🔹 Loading Piper TTS voices...")
        if not os.path.exists("/home/acer/piper/en_US-amy-medium.onnx"):
            raise FileNotFoundError("Piper En voice not found")
        voice_en = PiperVoice.load("/home/acer/piper/en_US-amy-medium.onnx")
        if not os.path.exists("/home/acer/piper/vi_VN-vais1000-medium.onnx"):
            raise FileNotFoundError("Piper Vi voice not found")
        voice_vi = PiperVoice.load("/home/acer/piper/vi_VN-vais1000-medium.onnx")

        logging.info("✅ Models loaded successfully!\n")
        return vosk_vi, vosk_en, vi2en_tokenizer, vi2en_model, en2vi_tokenizer, en2vi_model, voice_en, voice_vi
    except Exception as e:
        logging.error(f"❌ Error loading models: {e}")
        sys.exit(1)

# Load models once
vosk_vi, vosk_en, vi2en_tokenizer, vi2en_model, en2vi_tokenizer, en2vi_model, voice_en, voice_vi = load_models()

# =============================
# 🎙️ GHI ÂM (arecord) VỚI THREADING
# =============================
def record_audio():
    try:
        logging.info("🎤 Bắt đầu ghi âm...")
        arecord_cmd = [
            "arecord",
            "-D", "plughw:1,0",  
            "-f", FORMAT,
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
            WAV_FILE
        ]
        proc = subprocess.Popen(arecord_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Dừng khi nút thả
        while GPIO.input(PIN_BUTTON) == GPIO.HIGH:
            time.sleep(0.05)
        proc.terminate()
        proc.wait(timeout=5)  # Timeout để tránh hang
        logging.info("⏹️ Dừng ghi âm.")
    except subprocess.TimeoutExpired:
        proc.kill()
        logging.error("❌ arecord timeout")
    except Exception as e:
        logging.error(f"❌ Error in record_audio: {e}")

# =============================
# 🗣️ NHẬN DẠNG GIỌNG NÓI
# =============================
def recognize_speech(filename, lang="vi"):
    try:
        rec = KaldiRecognizer(vosk_vi if lang=="vi" else vosk_en, SAMPLE_RATE)
        with wave.open(filename, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
                logging.warning("⚠️ WAV phải mono, 16-bit, 16kHz")
                return ""
            text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    text += json.loads(rec.Result()).get("text", " ") + " "
            text += json.loads(rec.FinalResult()).get("text", "")
        text = text.strip()
        logging.info(f"🗣️ Recognized ({lang}): {text}")
        return text
    except Exception as e:
        logging.error(f"❌ Error in recognize_speech: {e}")
        return ""

# =============================
# 🌐 DỊCH NGÔN NGỮ
# =============================
def translate_text(text, direction):
    try:
        if direction=="vi2en":
            inputs = vi2en_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            output = vi2en_model.generate(**inputs)
            result = vi2en_tokenizer.decode(output[0], skip_special_tokens=True)
        else:
            inputs = en2vi_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            output = en2vi_model.generate(**inputs)
            result = en2vi_tokenizer.decode(output[0], skip_special_tokens=True)
        logging.info(f"🌍 Translated ({direction}): {result}")
        gc.collect()  # Giải phóng bộ nhớ
        return result
    except Exception as e:
        logging.error(f"❌ Error in translate_text: {e}")
        return text  # Trả về text gốc nếu lỗi

# =============================
# 🔊 PHÁT ÂM (synth WAV trước)
# =============================
def speak_text(text, lang="en"):
    try:
        voice = voice_en if lang=="en" else voice_vi
        temp_wav = "/home/acer/output.wav"
        # Lưu ra WAV
        voice.synthesize_to_file(text, temp_wav)
        # Phát WAV bằng pyaudio
        wf = wave.open(temp_wav, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()
        p.terminate()
        gc.collect()  # Giải phóng bộ nhớ
    except Exception as e:
        logging.error(f"❌ Error in speak_text: {e}")

# =============================
# 🚀 CHƯƠNG TRÌNH CHÍNH (VỚI THREADING VÀ DEBOUNCE)
# =============================
def main_loop():
    try:
        logging.info("🔧 Ready. Nhấn GPIO17 để thu âm...")
        GPIO.add_event_detect(PIN_BUTTON, GPIO.RISING, callback=handle_button, bouncetime=200)
        while True:
            time.sleep(1)  # Giữ vòng lặp chạy
    except KeyboardInterrupt:
        GPIO.cleanup()
        logging.info("👋 Exiting program.")
    except Exception as e:
        logging.error(f"❌ Error in main loop: {e}")
        GPIO.cleanup()

def handle_button():
    try:
        # Chọn chế độ dịch
        direction = "vi2en" if GPIO.input(PIN_MODE) == GPIO.LOW else "en2vi"
        lang_rec = "vi" if direction=="vi2en" else "en"
        lang_tts = "en" if direction=="vi2en" else "vi"

        # Ghi âm trong thread
        record_thread = threading.Thread(target=record_audio)
        record_thread.start()
        record_thread.join()  # Chờ ghi âm xong

        # Nhận dạng
        if not os.path.exists(WAV_FILE):
            logging.error("❌ WAV file not found")
            return
        text = recognize_speech(WAV_FILE, lang_rec)
        if not text:
            return

        # Dịch
        translated = translate_text(text, direction)

        # Phát âm
        speak_text(translated, lang_tts)
    except Exception as e:
        logging.error(f"❌ Error in handle_button: {e}")

if __name__ == "__main__":
    main_loop()
