from unidecode import unidecode
import re
import numpy as np
import string

alphabet = string.ascii_lowercase+" "

def normalize_text(text:str)->str:
    text = unidecode(text)
    # replace . , ? ! with blank space
    text = re.sub(r'[.,?!]', '', text)
    # removes all chars except ASCII-letters and blank spaces
    text = re.sub(r'[^a-zA-Z ]', '', text)

    text = " ".join(word.strip() for word in text.split())
    text = text.lower()
    return text


def split_text(text:str, num_chars:int):
    stop = len(text) // num_chars * num_chars
    return [text[i:i+num_chars] for i in range(0,stop, num_chars)]

def n_chars_to_idx(chars: str) -> int:
    assert "  " not in chars
    base = len(alphabet)
    value = 0
    for c in chars:
        idx = alphabet.index(c)
        value = value * base + idx
    return value

def encode_text(text:str, length: int, dtype:type = np.int32)->tuple[np.ndarray, int]:
    text = normalize_text(text)
    text:list[str] = split_text(text, 2)
    text = text[:length]
    array = np.zeros(length, dtype=dtype)
    for i,x in enumerate(map(n_chars_to_idx,text)):
        array[i] = x
    return array, len(text)


def hotencode_target(language:str, lang_list: np.ndarray, dtype:type = np.float32):
    array: np.ndarray = (lang_list == language)
    return array.astype(dtype)
