import torch
import nltk
from nltk.corpus import words
from torch.utils.data import DataLoader, random_split


import torch
import nltk
from nltk.corpus import words
from torch.utils.data import DataLoader, random_split


class PigLatinDataset(torch.utils.data.Dataset):
    def __init__(self, word_list, max_len=20):
        self.max_len = max_len
        self.data = [(word, self.to_piglatin(word)) for word in word_list]

        # 构建字符集（src 和 tgt 中的所有字符）+ 特殊符号
        # initialize character set from the dataset
        chars = set()
        for src, tgt in self.data:
            chars.update(src)
            chars.update(tgt)

        # 特殊符号
        # pad, bos, eos
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'

        sorted_chars = sorted(list(chars))
        self.char2id = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
        }
        for idx, ch in enumerate(sorted_chars, start=3):
            self.char2id[ch] = idx

        self.id2char = {i: ch for ch, i in self.char2id.items()}
        self.vocab_size = len(self.char2id)

    def to_piglatin(self, word):
        if not word.isalpha(): return word
        word = word.lower()
        vowels = 'aeiou'
        if word[0] in vowels:
            return word + 'yay'
        else:
            for i, c in enumerate(word):
                if c in vowels:
                    return word[i:] + word[:i] + 'ay'
            return word + 'ay'  

    def encode(self, text):
        return [self.char2id.get(ch, self.char2id[self.pad_token]) for ch in text]

    def decode(self, ids):
        return ''.join(self.id2char.get(i, '?') for i in ids)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_ids = self.encode(src_text)[:self.max_len]
        src_ids += [self.char2id[self.pad_token]] * (self.max_len - len(src_ids))

        # tgt_input / tgt_output
        bos_id = self.char2id[self.bos_token]
        eos_id = self.char2id[self.eos_token]

        tgt_encoded = self.encode(tgt_text)
        tgt_encoded = tgt_encoded[:self.max_len - 2]
        tgt_ids = [bos_id] + tgt_encoded + [eos_id]

        tgt_input = tgt_ids[:-1]
        tgt_output = tgt_ids[1:]

        # padding
        pad_id = self.char2id[self.pad_token]
        tgt_input += [pad_id] * (self.max_len - len(tgt_input))
        tgt_output += [pad_id] * (self.max_len - len(tgt_output))

        # src 是模型条件，input 是 decoder 上下文输入，output 是训练监督目标。
        # src: condition input, input: decoder context input, output: training supervision target
        return torch.tensor(src_ids), torch.tensor(tgt_input), torch.tensor(tgt_output)

    def __len__(self):
        return len(self.data)



def dataset_generator(batch_size=32, max_len = 10, seed=42):
    nltk.download('words')
    
    # 1. 获取英文单词列表（全小写、纯字母、长度限制）
    word_list = [w.lower() for w in words.words() if w.isalpha() and len(w) <= max_len]
    
    # 2. 构建数据集对象
    dataset = PigLatinDataset(word_list, max_len=max_len)  

    # 3. 数据集划分
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )

    # 4. 构建 DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size)
    test_loader  = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader

    