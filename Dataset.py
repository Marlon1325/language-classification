from torch.utils.data import Dataset
import torch
from prepare_data import *
import numpy as np

languages = np.array([
 'Arabic',
 'Chinese',
 'Czech',
 'Dutch',
 'English',
 'Finnish',
 'French',
 'German',
 'Greek',
 'Hungarian',
 'Italian',
 'Norwegian',
 'Polish',
 'Portuguese',
 'Romanian',
 'Russian',
 'Spanish',
 'Swedish',
 'Turkish'
])


def loadData()->tuple[list[str], list[str]]:
    "-> (data, targets)"
    data = []
    targets = []
    for lang in languages:
        with open(f"./data/{lang}.txt", "r", encoding="ascii") as f:
            while (x:= f.readline()) != "":
                x = x.replace("\n","")
                if len(x) == 0:
                    continue
                targets.append(lang)
                data.append(x)
    return data, targets


class Languages_Dataset(Dataset):
    def __init__(self, data, targets):
        assert len(data) == len(targets)
        self.data = tuple(zip(targets, data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx)->tuple[torch.Tensor, torch.Tensor, int]:
        lang, text = self.data[idx]
        text_max_length = 100
        inp = encode_text(text, text_max_length)
        target = hotencode_target(lang, languages)
        inp = torch.from_numpy(inp)
        target = torch.from_numpy(target)
        return inp, target, min(text_max_length, len(text))

    @staticmethod
    def collate_fn(batch):
        inputs, targets, lengths = zip(*batch)  
        
        inputs = torch.stack(inputs)      # (batch_size, seq_len, alphabet_size)
        targets = torch.stack(targets)    # (batch_size, num_classes)
        lengths = torch.tensor(lengths)   # (batch_size)
    
        # sort lengths in descending order (important for pack_padded_sequence)
        lengths, perm_idx = lengths.sort(descending=True)
        inputs = inputs[perm_idx]
        targets = targets[perm_idx]
    
        return inputs, targets, lengths

