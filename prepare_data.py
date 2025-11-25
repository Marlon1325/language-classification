from unidecode import unidecode
import re
import numpy as np
import string

def normalize_text(text:str)->str:
    text = unidecode(text)
    # replace . , ? ! with blank space
    text = re.sub(r'[.,?!]', '', text)
    # removes all chars except ASCII-letters and blank spaces
    text = re.sub(r'[^a-zA-Z ]', '', text)

    text = " ".join(word.strip() for word in text.split())
    text = text.lower()
    return text




def encode_text(text:str, length: int, vocab:str = " "+string.ascii_lowercase, dtype:type = np.int32)->np.ndarray:
    text = text[:length]
    array = np.zeros(length, dtype=dtype)
    for i,char in enumerate(text):
        array[i] = vocab.find(char)
    return array


def hotencode_target(language:str, lang_list: np.ndarray, dtype:type = np.float32):
    array: np.ndarray = (lang_list == language)
    return array.astype(dtype)
