import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import utils
from data.dataset import PuncDataset
from various_punctuator import LSTMPPunctuator
from punc_solver import PuncSolver

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

parser = argparse.ArgumentParser(description="LSTM punctuation prediction training.")
# original data
parser.add_argument('--train_data', default='',
                    help='Training text data path.')
parser.add_argument('--cv_data', default='',
                    help='Cross validation text data path.')
parser.add_argument('--vocab', default='',
                    help='Input vocab. (Don\'t include <UNK> and <END>)')
parser.add_argument('--punc_vocab', default='',
                    help='Output punctuations vocab. (Don\'t include " ")')
# model hyper parameters
parser.add_argument('--vocab_size', default=100000+2, type=int,
                    help='Input vocab size. (Include <UNK> and <END>)')
parser.add_argument('--embedding_size', default=256, type=int,
                    help='Input embedding size.')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='Hidden size of each direction.')
parser.add_argument('--proj_size', default=256, type=int,
                    help='Hidden size of each direction.')
parser.add_argument('--hidden_layers', default=2, type=int,
                    help='Number of BLSTM layers')
parser.add_argument('--num_class', default=5, type=int,
                    help='Number of output classes. (Include blank space " ")')
# training hyper parameters
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--bptt_step', default=20, type=int,
                    help='Step of truncated BPTT')
parser.add_argument('--epochs', default=2, type=int,
                    help='Number of training epochs')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='Initial learning rate (now only support Adam)')
parser.add_argument('--half_lr', dest='half_lr', action='store_true',
                    help='Halving learning rate when get small improvement')
parser.add_argument('--L2', default=0, type=float, help='L2 regularization')
parser.add_argument('--max_norm', default=250, type=int,
                    help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--early_stop', dest='early_stop', action='store_true',
                    help='Early stop training when halving lr but still get'
                    'small improvement')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='Watching training process')
parser.add_argument('--print_freq', default=1000, type=int,
                    help='Frequency of printing training infomation')


def main(args):
    # Data Loader, generate a batch of (inputs, labels) in each iteration
    tr_dataset = PuncDataset(args.train_data, args.vocab, args.punc_vocab)
    cv_dataset = PuncDataset(args.cv_data, args.vocab, args.punc_vocab)
    data = {'tr_dataset': tr_dataset, 'cv_dataset': cv_dataset}
    # Model
    model = LSTMPPunctuator(args.vocab_size, args.embedding_size, args.hidden_size,
                            args.proj_size, args.hidden_layers, args.num_class)
    model = model.cuda()  # Just support GPU now.
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.L2)
    # Log
    print(model)
    print("Number of parameters: %d" % utils.num_param(model))

    solver = PuncSolver(data, model, criterion, optimizer,
                        batch_size=args.batch_size,
                        bptt_step=args.bptt_step,
                        epochs=args.epochs,
                        half_lr=args.half_lr,
                        max_norm=args.max_norm,
                        early_stop=args.early_stop,
                        save_folder=args.save_folder,
                        checkpoint=args.checkpoint,
                        continue_from=args.continue_from,
                        model_path=args.model_path,
                        verbose=args.verbose,
                        print_freq=args.print_freq)
    solver.train()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
