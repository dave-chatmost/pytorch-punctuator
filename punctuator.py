import argparse
import io
import sys

import numpy as np
import torch
from torch.autograd import Variable

from data.dataset import NoPuncTextDataset
from various_punctuator import LSTMPPunctuator
from utils import add_punc_to_txt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

parser = argparse.ArgumentParser(description="LSTM punctuation prediction.")
parser.add_argument('--data', default='',
                    help='Text data to be punctuated')
parser.add_argument('--vocab', default='',
                    help='Input vocab. (Don\'t include <UNK> and <END>)')
parser.add_argument('--punc_vocab', default='',
                    help='Output punctuations vocab. (Don\'t include " ")')
parser.add_argument('--model_path', default='',
                    help='Path to model file created by training')
parser.add_argument('--output', default='-',
                    help='Write the punctuated text (default to stdout)')
parser.add_argument('--cuda', action="store_true",
                    help='Use GPU to test model')


def inference(args):
    dataset = NoPuncTextDataset(args.data, args.vocab, args.punc_vocab)
    model = LSTMPPunctuator.load_model(args.model_path, cuda=args.cuda)
    model.eval()
    # Output function
    if args.output == "-":
        output, endline = print, ""
    else:
        ofile = open(args.output, "w", encoding='utf8')
        output, endline = ofile.write, "\n"

    for i, (id_seq, txt_seq) in enumerate(dataset):
        # TODO: fix this caused by torch.mm()
        id_seq = np.expand_dims(id_seq, axis=1)  # data dim (T) -- > (T, 1)
        if args.cuda:
            inputs = Variable(torch.LongTensor(id_seq).cuda(), volatile=True)
        else:
            inputs = Variable(torch.LongTensor(id_seq), volatile=True)
        # forward propagation
        hidden = model.init_hidden(batch_size=1)
        scores, hidden = model(inputs, hidden)
        # convert score to prediction result
        scores = scores.view(-1, model.num_class)
        _, predict = torch.max(scores, 1)
        predict = predict.data.cpu().numpy().tolist()
        # add punctuation to text
        result = add_punc_to_txt(txt_seq, predict, dataset.class2punc)
        # write punctuated text with to output
        output(result + endline)


if __name__ == "__main__":
    args = parser.parse_args()
    inference(args)
