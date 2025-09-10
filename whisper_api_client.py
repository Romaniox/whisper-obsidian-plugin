import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import keyboard
import requests
import tempfile
import pyperclip
import time
import os

API_URL = "http://127.0.0.1:6431/transcribe"  # твой локальный API
is_recording = False
recording_data = []
samplerate = 16000  # Whisper любит 16кГц


def callback(indata, frames, time_info, status):
    """Собираем звук в память"""
    if is_recording:
        recording_data.append(indata.copy())


def toggle_recording():
    global is_recording, recording_data
    if not is_recording:
        print("🎙 Начало записи...")
        recording_data = []
        is_recording = True
    else:
        print("⏹ Остановка записи...")
        is_recording = False
        save_and_transcribe()


def save_and_transcribe():
    global recording_data
    if not recording_data:
        return

    # Склеиваем аудиофрагменты
    audio = np.concatenate(recording_data, axis=0)

    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav.write(f.name, samplerate, audio)
        temp_path = f.name

    print("📤 Отправка на сервер...")
    try:
        with open(temp_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            data = {
                "language": "ru",   # 🔹 сюда ставишь нужный язык ("en", "ru", "auto" и т.д.)
                "model": "turbo",   # при желании можно менять
                "prompt": ""        # опционально
            }
            resp = requests.post(API_URL, files=files, data=data)

        resp.raise_for_status()
        text = resp.json()["text"].strip()
        print("✅ Распознано:", text)

        # Вставляем текст на место курсора
        pyperclip.copy(text)
        time.sleep(0.1)
        keyboard.write(text)

    except Exception as e:
        print("❌ Ошибка:", e)
    finally:
        os.unlink(temp_path)


def main():
    print("Нажми Alt+Q для старта/остановки записи. Esc для выхода.")
    keyboard.add_hotkey("alt+q", toggle_recording)

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        keyboard.wait("esc")


if __name__ == "__main__":
    main()
