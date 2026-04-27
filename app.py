import os
import json
import sys
from faster_whisper import WhisperModel

MODEL_SIZE = "medium"
DATA_PATH = "/data"

def select_from_list(items, title="выберите пункт"):
    print(f"\n--- {title.upper()} ---")
    for i, item in enumerate(items, 1):
        print(f"{i}. {item}")
    
    while True:
        try:
            choice = int(input(f"\nВведите номер (1-{len(items)}): "))
            if 1 <= choice <= len(items):
                return items[choice - 1]
        except (ValueError, IndexError):
            print("Ошибка! Введите число из списка.")

def process_audio(file_path):
    print(f"\n[1/3] Загрузка модели {MODEL_SIZE} на GPU...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

    print(f"[2/3] Обработка аудио: {os.path.basename(file_path)}")

    segments, info = model.transcribe(
        file_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt="Lyrics for the song, karaoke style, music singing."
    )

    print(f"Язык: {info.language} ({info.language_probability:.2%})")
    print(f"[3/3] Результаты:\n" + "-"*30)

    full_data = {
        "file": os.path.basename(file_path),
        "language": info.language,
        "segments": []
    }

    for segment in segments:
        line_text = segment.text.strip()
        timestamp = f"[{int(segment.start // 60):02}:{segment.start % 60:05.2f}]"

        print(f"{timestamp} {line_text}")

        seg_info = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": line_text,
            "words": [
                {"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)}
                for w in segment.words
            ]
        }
        full_data["segments"].append(seg_info)

    json_path = file_path.replace(".m4a", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    
    print("-"*30)
    print(f"Готово! Подробный лог сохранен в: {json_path}")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Критическая ошибка: Папка {DATA_PATH} не найдена. Проверь Docker -v.")
        sys.exit(1)

    langs = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    if not langs:
        print("Папка /data пуста.")
        sys.exit(1)
    
    chosen_lang = select_from_list(langs, "выберите язык (папку)")
    lang_dir = os.path.join(DATA_PATH, chosen_lang)

    songs = sorted([f for f in os.listdir(lang_dir) if f.endswith(".m4a")])
    if not songs:
        print(f"В папке {chosen_lang} нет .m4a файлов.")
        sys.exit(1)

    chosen_song = select_from_list(songs, f"выберите песню ({chosen_lang})")

    try:
        process_audio(os.path.join(lang_dir, chosen_song))
    except Exception as e:
        print(f"Произошла ошибка при обработке: {e}")