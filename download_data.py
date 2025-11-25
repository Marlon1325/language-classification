import requests
from prepare_data import normalize_text
import os,time



def text_to_lines(text: str, max_words: int = 10) -> list[str]:
    words = text.split()
    lines = []
    
    for i in range(0, len(words), max_words):
        line = " ".join(words[i:i+max_words])
        lines.append(line)
    
    return lines



def download_data( num_data: int, folder: str = "data"):
    languages = {
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Arabic": "ar",
    "Swedish": "sv",
    "Polish": "pl",
    "Czech": "cs",
    "Hungarian": "hu",
    "Turkish": "tr",
    "Finnish": "fi",
    "Norwegian": "no",
    "Romanian": "ro",
    "Greek": "el",
}
    os.makedirs(folder, exist_ok=True)
    
    for lang, lang_code in languages.items():
        num_lines = 0
        file_path = os.path.join(folder, f"{lang}.txt")
        while num_lines < num_data:
            try:
                url = f"https://{lang_code}.wikipedia.org/api/rest_v1/page/random/summary"
                resp = requests.get(url)
                resp.raise_for_status()
            
                
            except:
                time.sleep(2)
                continue
            data:dict = resp.json()
            text = data.get("extract", "")
            text = normalize_text(text)
            lines = text_to_lines(text, 10)
            with open(file_path, "a", encoding="ascii") as f:
                f.write("\n".join(lines))
                f.write("\n")

            with open(file_path, "r", encoding="ascii") as f:
                num_lines = sum(1 for _ in f)
            print(
                    f"\r{lang}: {num_lines}/{num_data} Zeilen ({ (num_lines / num_data * 100):.2f}%) erreicht.",
                    flush=True,
                    end="\t" * 10
                )



download_data(5000, "data")
download_data(1000, "test")