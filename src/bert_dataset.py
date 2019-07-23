import random
from pathlib import Path
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

VOCAB_PATH = Path('../byebyejuly/bert-chinese-wwm/vocab.txt')


def convert_one_line(text, max_seq_length=None, tokenizer=None):
    max_seq_length -= 2
    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[:max_seq_length // 2] + tokens_a[-(max_seq_length - max_seq_length // 2):]
    one_token = tokenizer.convert_tokens_to_ids(
        ["[CLS]"] + tokens_a + ["[SEP]"])
    return one_token


class TrainDataset(Dataset):

    def __init__(self, data, vocab_path=VOCAB_PATH, do_lower=True, shuffle=True):
        super(TrainDataset, self).__init__()
        self._a = data['a'].tolist()
        self._b = data['b'].tolist()
        self._label = data['label'].tolist()
        self._shuffle = shuffle
        self._tokenizer = BertTokenizer.from_pretrained(
            vocab_path, cache_dir=None, do_lower_case=do_lower)

    def __len__(self):
        return len(self._a)

    def __getitem__(self, idx):
        a = self._a[idx]
        b = self._b[idx]
        token_a = convert_one_line(a, max_seq_length=256, tokenizer=self._tokenizer)
        token_b = convert_one_line(b, max_seq_length=256, tokenizer=self._tokenizer)
        if self._shuffle and random.random() < 0.5:
            return torch.LongTensor(token_b), torch.LongTensor(token_a), self._label[idx]
        else:
            return torch.LongTensor(token_a), torch.LongTensor(token_b), self._label[idx]


def collate_fn(batch):
    a, b, targets = zip(*batch)
    a = pad_sequence(a, batch_first=True)
    b = pad_sequence(b, batch_first=True)
    targets = torch.FloatTensor(targets)
    return a, b, targets
