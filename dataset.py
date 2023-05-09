import torch
from torchtext.datasets import Multi30k
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re, random
from torch.nn.utils.rnn import pad_sequence

class en_de_dataset(Dataset):
    def __init__(self, split = 'train', pair =('en', 'de'), input_vocab = None, target_vocab = None):
        # special token used for
        special_vocab = ['<pad>', '<sos>', '<eos>', '<unk>']

        sen1_token = []
        sen2_token = []
        sentences_pair = Multi30k(split=(split), language_pair=pair)

        self.length = 0
        self.inputs = []
        self.targets = []

        for input_sen, target_sen in sentences_pair:
            if len(input_sen) > 0 and len(target_sen) > 0:
                self.length += 1
                self.inputs.append(input_sen)
                self.targets.append(target_sen)
                sen1_token += re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', input_sen.lower())
                sen2_token += re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', target_sen.lower())

        if input_vocab is None:
            self.input_vocab = Counter(sen1_token)
            self.input_vocab = special_vocab + [k for k, _ in self.input_vocab.items()]
            self.input_vocab = {token: idx for idx, token in enumerate(self.input_vocab)}
        else:
            self.input_vocab = input_vocab

        if target_vocab is None:
            self.target_vocab = Counter(sen2_token)
            self.target_vocab = special_vocab + [k for k, _ in self.target_vocab.items()]
            self.target_vocab = {token: idx for idx, token in enumerate(self.target_vocab)}
        else:
            self.target_vocab = target_vocab

        self.id_to_word_input = {v: k for k, v in self.input_vocab.items()}
        self.id_to_word_target = {v: k for k, v in self.target_vocab.items()}
        print(f"Input vocab size: {len(self.input_vocab)}")
        print(f"Target vocab size: {len(self.target_vocab)}")

    def random_pairs(self):
        pair_idx = random.randint(0, len(self.inputs))
        return self.inputs[pair_idx], self.targets[pair_idx]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        input_indice = []
        target_indice = [1]
        for token in re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', self.inputs[item].lower()):
            try:
                input_indice.append(self.input_vocab[token])
            except:
                input_indice.append(self.input_vocab['<unk>'])

        for token in re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', self.targets[item].lower()):
            try:
                target_indice.append(self.target_vocab[token])
            except:
                target_indice.append(self.target_vocab['<unk>'])

        target_indice.append(2)
        return torch.LongTensor(input_indice).unsqueeze(1), torch.LongTensor(target_indice).unsqueeze(1)

def collate_fn_padding(batch):
    # Each item in batch (input, target)
    inputs_batch = []
    target_batch = []
    teaching_batch = []
    for input, target in batch:
        inputs_batch.append(input)
        target_batch.append(target[1:])
        teaching_batch.append(target[:-1])

    # Merge list of sentences into a batch with padding
    batch_inputs = pad_sequence(inputs_batch, padding_value=0).permute(1, 0, 2).squeeze(-1)
    batch_target = pad_sequence(target_batch, padding_value=0).permute(1, 0, 2).squeeze(-1) # [seq, batch, unsqueezed]
    batch_teaching = pad_sequence(teaching_batch, padding_value=0).permute(1, 0, 2).squeeze(-1)

    # mask for loss
    # mask = torch.zeros_like(batch_target)
    mask = batch_target != 0
    # mask = mask.bool()

    # Transfer back to [batch, indice]
    return batch_inputs, batch_target, batch_teaching, mask



if __name__ == '__main__':
    batch_size = 2
    trainset = en_de_dataset(split='valid')
    train_loader = DataLoader(
        dataset=trainset,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        collate_fn=collate_fn_padding,
        drop_last=True
    )
    print(len(trainset))

    for idx, (input, target, teaching, masks) in enumerate(train_loader):
        print(input.shape, target.shape)

        for idx, (sen1, sen2, sen3, mask) in enumerate(zip(input, target, teaching, masks)):
            sen1 = [trainset.id_to_word_input[int(word)] for word in sen1]
            sen2 = [trainset.id_to_word_target[int(word)] for word in sen2]
            sen3 = [trainset.id_to_word_target[int(word)] for word in sen3]
            mask = [bool(m) for m in mask]
            print(f'{" ".join(sen1)}\n{" ".join(sen2)}\n{" ".join(sen3)}\n{mask}\n\n')

            if idx >= 50:
                break

    input_str, target_str = trainset.random_pairs()
    print((input_str, target_str))
    input_tokens = [trainset.input_vocab[token] for token in re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', input_str.lower())]
    target_tokens = [trainset.target_vocab[token] for token in re.findall(r'\b[A-Za-zäöüÄÖÜß][A-Za-zäöüÄÖÜß]+\b', target_str.lower())]
    print(input_tokens, target_tokens)

    print(torch.LongTensor([1]).unsqueeze(0).shape)

