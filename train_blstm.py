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
from data_loader import get_loader
from various_punctuator import BLSTMPunctuator

# make print() work correctly under Chinese
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

parser = argparse.ArgumentParser(description="BLSTM punctuation prediction training.")
# original data
parser.add_argument('--train-data', default='',
                    help='Training text data path.')
parser.add_argument('--cv-data', default='',
                    help='Cross validation text data path.')
parser.add_argument('--vocab', default='',
                    help='Input vocab. (Don\'t include <UNK> and <END>)')
parser.add_argument('--punc-vocab', default='',
                    help='Output punctuations vocab. (Don\'t include " ")')
# model hyper parameters
parser.add_argument('--vocab-size', default=100000+2, type=int,
                    help='Input vocab size. (Include <UNK> and <END>)')
parser.add_argument('--embed-size', default=256, type=int,
                    help='Input embedding size.')
parser.add_argument('--hidden-size', default=512, type=int,
                    help='Hidden size of each direction.')
parser.add_argument('--hidden-layers', default=2, type=int,
                    help='Number of BLSTM layers')
parser.add_argument('--num-class', default=5, type=int,
                    help='Number of output classes. (Include blank space " ")')
# training hyper parameters
parser.add_argument('--batch-size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', default=2, type=int,
                    help='Number of training epochs')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='Initial learning rate (now only support Adam)')
parser.add_argument('--L2', default=0, type=float, help='L2 regularization')
parser.add_argument('--max-norm', default=250, type=int,
                    help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--early-stop', dest='early_stop', action='store_true',
                    help='Early stop training when get small improvement')
# save and load model
parser.add_argument('--save-folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model-path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='Watching training process')
parser.add_argument('--print-freq', default=1000, type=int,
                    help='Frequency of printing training infomation')


def run_one_epoch(data_loader, model, criterion, optimizer, epoch, args,
                  cross_valid=False):
    total_loss = 0.0
    total_acc = 0.0
    total_words = 0
    start = time.time()
    for i, (inputs, labels, lengths) in enumerate(data_loader):
        # 1. mini-batch data
        inputs = Variable(inputs.cuda(), requires_grad=False)
        labels = Variable(labels.cuda(), requires_grad=False)
        # 2. forward and compute loss
        optimizer.zero_grad()
        scores = model(inputs, lengths)
        scores = scores.view(-1, args.num_class)
        loss = criterion(scores, labels.view(-1))
        if not cross_valid:
            # 3. backward
            loss.backward()
            # Clip gradient
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # 4. update
            optimizer.step()

        total_loss += loss.data[0]
        _, predict = torch.max(scores, 1)
        correct = (predict == labels.view(-1)).sum().data[0]
        words = np.sum(lengths)
        acc = 100.0 * correct / words
        total_acc += acc
        total_words =+ words

        if args.verbose and i % args.print_freq == 0:
            print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                  'Perplexity {3:.3f} | Acc {4:.3f} | {5} words/s | '
                  '{6:.1f} ms/batch'.format(
                      epoch + 1, i, total_loss / (i + 1),
                      np.exp(total_loss / (i + 1)),
                      total_acc / (i + 1),
                      int(total_words / (time.time()-start)),
                      1000*(time.time()-start)/(i+1)),
                  flush=True)
        del loss
        del scores
    return total_loss / (i + 1), total_acc / (i + 1)


def main(args):
    # IO -- Data Loader, generate a batch of (inputs, labels) in each iteration
    tr_data_loader = get_loader(args.train_data, args.vocab, args.punc_vocab,
                                batch_size=args.batch_size)
    cv_data_loader = get_loader(args.cv_data, args.vocab, args.punc_vocab,
                                batch_size=args.batch_size)
    # Model
    model = BLSTMPunctuator(args.vocab_size, args.embed_size,
                            args.hidden_size, args.hidden_layers,
                            args.num_class)
    model.cuda()  # Just support GPU now.
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # pad labels by -1
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.L2)
    # Log
    print(model)
    print("Number of parameters: %d" % utils.num_param(model))

    # Restore model information from specified model file
    if args.continue_from:
        print('Loading checkpoint model %s' % args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['state_dict'])
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', 1))
    else:
        start_epoch = 0
    # Create save folder
    save_folder = args.save_folder
    utils.mkdir(save_folder)

    prev_val_loss = float("inf")
    best_val_loss = float("inf")
    halving = False

    # Train model multi-epochs
    for epoch in range(start_epoch, args.epochs):
        ######## Train one epoch
        print('Training...')
        model.train()  # Turn on BatchNorm & Dropout
        start = time.time()
        avg_loss, avg_acc = run_one_epoch(tr_data_loader, model, criterion,
                                          optimizer, epoch, args)
        print('-'*85)
        print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Train Loss {2:.3f} | Train Acc {3:.2f} '.format(
              epoch + 1, time.time() - start, avg_loss, avg_acc))
        print('-'*85)

        ######## Save model at each epoch
        if args.checkpoint:
            file_path = os.path.join(save_folder, 'epoch%d.pth.tar'%(epoch + 1))
            torch.save(BLSTMPunctuator.serialize(model, optimizer, epoch + 1),
                       file_path)
            print('Saving checkpoint model to %s' % file_path)

        ######## Cross validation
        print('Cross validation...')
        model.eval()  # Turn off Batchnorm & Dropout
        start = time.time()
        val_loss, val_acc = run_one_epoch(cv_data_loader, model, criterion,
                                          optimizer, epoch, args,
                                          cross_valid=True)
        print('-'*85)
        print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Valid Loss {2:.3f} | Valid Acc {3:.2f} '.format(
              epoch + 1, time.time() - start, val_loss, val_acc))
        print('-'*85)

        ######## Adjust learning rate, halving
        if val_loss >= prev_val_loss:
            if args.early_stop and halving:
                print("Already start halving learing rate, it still gets too "
                      "small imporvement, stop training early.")
                break
            halving = True
        if halving:
            optim_state = optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = \
                optim_state['param_groups'][0]['lr'] / 2.0
            optimizer.load_state_dict(optim_state)
            print('Learning rate adjusted to: {lr:.6f}'.format(
                  lr=optim_state['param_groups'][0]['lr']))
        prev_val_loss = val_loss

        ######## Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            file_path = os.path.join(save_folder, args.model_path)
            torch.save(BLSTMPunctuator.serialize(model, optimizer, epoch + 1),
                       file_path)
            print("Find better validated model, saving to %s" % file_path)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
