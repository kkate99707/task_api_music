import os
from faster_whisper import WhisperModel
from eval import evaluate_text 

DATA_PATH = "/data"
LANGS = ['FR', 'IT', 'RU', 'EN', 'PT', 'ES', 'JP', 'PL']

# Загружаем модель один раз
model = WhisperModel("medium", device="cuda", compute_type="float16")

def run_benchmark():

    stats = {}

    for lang in LANGS:
        lang_dir = os.path.join(DATA_PATH, lang)
        if not os.path.exists(lang_dir):
            continue

        print(f"\n--- [ПРОЦЕСС] Анализ языка: {lang} ---")
        lang_accuracies = []
        audio_files = [f for f in os.listdir(lang_dir) if f.endswith('.m4a')]
        
        for audio_name in audio_files:
            audio_path = os.path.join(lang_dir, audio_name)
            txt_path = os.path.join(lang_dir, audio_name.replace('.m4a', '.txt'))

            segments, _ = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
            hypothesis = " ".join([s.text for s in segments]).strip()

            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    reference = f.read().strip()
                res = evaluate_text(reference, hypothesis)
                acc = res['accuracy']
                status = "OK"
            else:

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(hypothesis)
                acc = 100.0 
                status = "NEW (draft)"

            lang_accuracies.append(acc)
            print(f"  > Файл {audio_name}: Точность {acc:.2f}% [{status}]")

        if lang_accuracies:
            stats[lang] = sum(lang_accuracies) / len(lang_accuracies)

    print("\n" + "="*50)
    print(f"{'ЯЗЫК':<10} | {'СРЕДНЯЯ ТОЧНОСТЬ':<20}")
    print("-" * 50)
    for lang, acc in stats.items():
        print(f"{lang:<10} | {acc:.2f}%")
    print("="*50 + "\n")
    print("Бенчмарк завершен. Можно посмотреть результаты в папке с музыкой")
    print("исправить ошибки и запустить скрипт снова для реальной оценки.")

if __name__ == "__main__":
    run_benchmark()