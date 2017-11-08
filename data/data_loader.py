import numpy as np
import torch
import torch.utils.data as data

import utils


class NoPuncTextParser(object):
    """
    Parse text without punctuation.
    Used by punctuation prediciton inference.
    """

    def __init__(self, txt_path, in_vocab_path, out_vocab_path):
        """Read txt file, input vocab and output vocab (punc vocab)."""
        self.txt_seqs = open(txt_path, encoding='utf8').readlines()
        self.num_seqs = len(self.txt_seqs)
        self.word2id = utils.load_vocab(in_vocab_path,
                                        extra_word_list=["<UNK>", "<END>"])
        self.punc2id = utils.load_vocab(out_vocab_path,
                                        extra_word_list=[" "])
        self.class2punc = { k : v for (v, k) in self.punc2id.items()}

    def __len__(self):
        """Return number of sentences in txt file."""
        return self.num_seqs

    def __getitem__(self, index):
        """Return input id sequence."""
        txt_seq = self.txt_seqs[index]
        word_id_seq = self.preprocess(txt_seq)
        return word_id_seq, txt_seq

    def preprocess(self, txt_seq):
        """Convert txt sequence to word-id-seq."""
        input = []
        for token in txt_seq.split():
            input.append(self.word2id.get(token, self.word2id["<UNK>"]))
        input.append(self.word2id["<END>"])
        return input


class PuncDataset(data.Dataset):
    """Custom Dataset for punctuation prediction."""

    def __init__(self, txt_path, in_vocab_path, out_vocab_path):
        """Read txt file, input vocab and output vocab (punc vocab)."""
        self.txt_seqs = open(txt_path, encoding='utf8').readlines()
        self.num_seqs = len(self.txt_seqs)
        self.word2id = utils.load_vocab(in_vocab_path,
                                        extra_word_list=["<UNK>", "<END>"])
        self.punc2id = utils.load_vocab(out_vocab_path,
                                        extra_word_list=[" "])

    def __len__(self):
        """Return number of sentences in txt file."""
        return self.num_seqs

    def __getitem__(self, index):
        """Return one Tensor pair of (input id sequence, punc id sequence)."""
        txt_seq = self.txt_seqs[index]
        word_id_seq, punc_id_seq = self.preprocess(txt_seq)
        return word_id_seq, punc_id_seq

    def preprocess(self, txt_seq):
        """Convert txt sequence to word-id-seq and punc-id-seq."""
        input = []
        label = []
        punc = " "
        for token in txt_seq.split():
            if token in self.punc2id:
                punc = token
            else:
                input.append(self.word2id.get(token, self.word2id["<UNK>"]))
                label.append(self.punc2id[punc])
                punc = " "
        input.append(self.word2id["<END>"])
        label.append(self.punc2id[punc])
        # input = torch.Tensor(input)
        # label = torch.Tensor(label)
        input = np.array(input)
        label = np.array(label)
        return input, label


def collate_fn(data):
    """Do padding. Pad labels by -1.
    Args:
        data: list of tuple (input, label)
            - input: torch tensor, variable length
            - label: torch tensor, variable length
    """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)
    # seperate inputs and labels
    input_seqs, label_seqs = zip(*data)
    # padding
    PAD = -1
    lengths = [len(seq) for seq in input_seqs]
    input_padded_seqs = torch.zeros(len(input_seqs), max(lengths)).long()
    label_padded_seqs = torch.zeros(len(input_seqs), max(lengths)).fill_(PAD).long()
    for i, (input, label) in enumerate(zip(input_seqs, label_seqs)):
        end = lengths[i]
        input_padded_seqs[i, :end] = input[:end]
        label_padded_seqs[i, :end] = label[:end]
    return input_padded_seqs, label_padded_seqs, lengths


def get_loader(txt_path, in_vocab_path, out_vocab_path, batch_size=1):
    """Return data loader for custom dataset.
    """
    dataset = PuncDataset(txt_path, in_vocab_path, out_vocab_path)
    # data loader for custome dataset
    # this will return (input_padded_seqs, label_padded_seqs, lengths) for each iteration
    # please see collate_fn for details
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    return data_loader
