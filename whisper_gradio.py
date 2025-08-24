import io
import os
import tempfile
import time
import wave

import ffmpeg
import gradio as gr
import numpy as np
import whisper

# Загрузка модели Whisper (можно выбрать tiny, base, small, medium, large)
model = whisper.load_model("turbo")


# Функция для обработки аудио
def transcribe_audio(audio):
    if audio is None:
        return "Ошибка: аудио не записано. Пожалуйста, запишите фразу через микрофон."

    # Сохранение аудио в временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file_path = temp_file.name
        try:
            # Если audio - это tuple (sample_rate, numpy_array)
            if isinstance(audio, tuple) and len(audio) == 2:
                sample_rate, audio_data = audio

                # Нормализация аудио данных (если они в диапазоне [-1, 1])
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    # Конвертируем в int16 для WAV
                    audio_data = (audio_data * 32767).astype(np.int16)

                # Записываем WAV файл
                with wave.open(temp_file_path, "wb") as wav_file:
                    wav_file.setnchannels(1)  # моно
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())

            elif isinstance(audio, bytes):
                # Если audio - это blob (bytes), записываем его напрямую
                temp_file.write(audio)
                temp_file.flush()
            else:
                # Конвертация аудио в WAV (для filepath)
                (
                    ffmpeg.input(audio)
                    .output(temp_file_path, format="wav")
                    .overwrite_output()
                    .run(quiet=True)
                )

        except Exception as e:
            return f"Ошибка при обработке аудио: {str(e)}"

        # Транскрипция с помощью Whisper
        try:
            result = model.transcribe(temp_file_path, language="ru")
            print(result)
            text = result["text"]
            print(text)
        except Exception as e:
            text = f"Ошибка при транскрипции: {str(e)}"
        finally:
            # Попытка удаления файла с небольшой задержкой
            for _ in range(5):  # Пять попыток
                try:
                    os.unlink(temp_file_path)
                    break
                except PermissionError:
                    time.sleep(0.5)  # Задержка 0.5 секунды перед повторной попыткой
                except Exception as e:
                    print(f"Ошибка при удалении файла: {str(e)}")
                    break

    return text


# Создание интерфейса Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Транскрипция речи с микрофона")
    gr.Markdown(
        "Нажмите на кнопку микрофона, чтобы начать запись, скажите фразу, затем нажмите снова, чтобы остановить. Результат появится ниже."
    )

    # Компонент для записи аудио с микрофона (возвращает tuple: sample_rate, numpy_array)
    audio_input = gr.Audio(sources="microphone", type="filepath")

    # Кнопка для запуска транскрипции
    transcribe_button = gr.Button("Получить текст")

    # Поле для вывода результата
    output_text = gr.Textbox(label="Результат транскрипции")

    # Связка кнопки с функцией транскрипции
    transcribe_button.click(
        fn=transcribe_audio, inputs=audio_input, outputs=output_text
    )

# Запуск приложения
demo.launch()
