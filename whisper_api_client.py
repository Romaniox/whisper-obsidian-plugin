import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import keyboard
import requests
import tempfile
import pyperclip
import time
import os

# === Настройки ===
API_URL = "http://127.0.0.1:6431/transcribe"  # твой локальный API
LANGUAGE = "ru"        # "ru", "en", "auto"
MODEL = "turbo"        # Whisper модель: tiny, base, small, medium, large, turbo
USE_CLIPBOARD = True   # True = вставка через Ctrl+V, False = посимвольный ввод
samplerate = 16000     # Whisper любит 16кГц

# === Глобальные переменные ===
is_recording = False
recording_data = []
LANGUAGE = "ru"  # язык по умолчанию


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
    global recording_data, LANGUAGE
    if not recording_data:
        return

    # Склеиваем аудиофрагменты
    audio = np.concatenate(recording_data, axis=0)

    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav.write(f.name, samplerate, audio)
        temp_path = f.name

    print(f"📤 Отправка на сервер (язык: {LANGUAGE})...")
    try:
        with open(temp_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            data = {
                "language": LANGUAGE,
                "model": MODEL,
                "prompt": ""
            }
            resp = requests.post(API_URL, files=files, data=data)

        resp.raise_for_status()
        text = resp.json()["text"].strip()
        print("✅ Распознано:", text)

        # Вставляем текст в активное поле ввода
        if USE_CLIPBOARD:
            old_clipboard = pyperclip.paste()  # сохраняем, что было в буфере
            pyperclip.copy(text)
            time.sleep(0.05)
            keyboard.press_and_release("ctrl+v")
            time.sleep(0.05)
            pyperclip.copy(old_clipboard)  # восстанавливаем
        else:
            keyboard.write(text, delay=0)

    except Exception as e:
        print("❌ Ошибка:", e)
    finally:
        os.unlink(temp_path)


def set_language(lang_code, lang_name):
    global LANGUAGE
    LANGUAGE = lang_code
    print(f"🌐 Язык переключён на {lang_name} ({lang_code})")


def main():
    print("Нажми Alt+Q для старта/остановки записи. alt+Esc для выхода.")
    print("Alt+1 = русский, Alt+2 = английский")

    keyboard.add_hotkey("alt+q", toggle_recording)
    keyboard.add_hotkey("alt+1", lambda: set_language("ru", "Русский"))
    keyboard.add_hotkey("alt+2", lambda: set_language("en", "Английский"))

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        keyboard.wait("alt+esc")


if __name__ == "__main__":
    main()
