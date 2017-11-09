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
from data.data_loader import PuncDataset
from various_punctuator import LSTMPPunctuator

# make print() work correctly under Chinese
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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


def get_one_batch_data(dataset, feats_utt, targets_utt, new_utt_flags,
                       batch_size, steps, idx):
    new_utt_flags = [0] * batch_size
    for b in range(batch_size):
        if feats_utt[b].shape[0] == 0:
            # print(idx)
            # TODO: make dataset an iterator?
            if idx == len(dataset): return None, None, None, idx, True
            feats, targets = dataset[idx]
            idx += 1
            if feats is not None:
                feats_utt[b] = feats
                targets_utt[b] = targets
                new_utt_flags[b] = 1
    
    # end the training after processing all the frames
    frames_to_go = 0
    for b in range(batch_size):
        frames_to_go += feats_utt[b].shape[0]
    if frames_to_go == 0: return None, None, None, idx, True

    #### START pack the mini-batch data ####
    feat_host = np.zeros((steps, batch_size))
    target_host = np.zeros((steps, batch_size))
    frame_num_utt = [0] * batch_size

    # slice at most 'batch_size' frames
    for b in range(batch_size):
        num_rows = feats_utt[b].shape[0]
        frame_num_utt[b] = min(steps, num_rows)

    # pack the features
    for b in range(batch_size):
        for t in range(frame_num_utt[b]):
            feat_host[t, b] = feats_utt[b][t]

    # pack the targets
    for b in range(batch_size):
        for t in range(frame_num_utt[b]):
            target_host[t, b] = targets_utt[b][t]
    #### END pack data ####

    # remove the data we just packed
    for b in range(batch_size):
        packed_rows = frame_num_utt[b]
        feats_utt[b] = feats_utt[b][packed_rows:]
        targets_utt[b] = targets_utt[b][packed_rows:]
        left_rows = feats_utt[b].shape[0]
        if left_rows < steps:
            feats_utt[b] = np.array([])
        # feats
        # rows = feats_utt[b].shape[0]
        # if rows == frame_num_utt[b]:
        #     feats_utt[b] = np.array([])
        # else:
        #     packed_rows = frame_num_utt[b]
        #     feats_utt[b] = feats_utt[b][packed_rows:]
        #     targets_utt[b] = targets_utt[b][packed_rows:]
    #### END prepare mini-batch data ####
    return feat_host, target_host, new_utt_flags, idx, False


def run_one_epoch(dataset, model, criterion, optimizer, epoch, args,
                  cross_valid=False):
    batch_size = args.batch_size
    steps = args.bptt_step
    feats_utt = [np.array([])] * batch_size  # every element is numpy ndarray
    targets_utt = [np.array([])] * batch_size
    new_utt_flags = [0] * batch_size
    # hidden = model.module.init_hidden(batch_size)
    hidden = model.init_hidden(batch_size)

    i = 0
    total_loss = 0.0
    total_correct = 0.0
    total_words = 0
    start = time.time()

    idx = 0
    while True:
        inputs, targets, new_utt_flags, idx, done = \
            get_one_batch_data(dataset, feats_utt, targets_utt, new_utt_flags,
                               batch_size, steps, idx)
        if done:
            break
        # 1. mini-batch data.
        inputs_tensor = torch.LongTensor(inputs.astype(np.int64)).cuda()
        targets_tensor = torch.LongTensor(targets.astype(np.int64)).cuda()
        inputs = Variable(inputs_tensor, requires_grad=False)
        targets = Variable(targets_tensor, requires_grad=False)
        new_utt_flags = Variable(torch.ByteTensor(new_utt_flags).view((1, -1)),
                                 requires_grad=False)
        # 2. forward and compute loss
        scores, hidden = model(inputs, hidden, new_utt_flags, train=True)
        scores = scores.view(-1, args.num_class)
        loss = criterion(scores, targets.view(-1))
        if not cross_valid:
            optimizer.zero_grad()
            # 3. backward
            loss.backward()
            # Clip gradient
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # 4. update
            optimizer.step()

        _, predict = torch.max(scores, 1)

        total_correct += (predict == targets.view(-1)).sum().data[0]
        total_loss += loss.data[0]
        total_words += batch_size * steps

        i += 1
        if args.verbose and i % args.print_freq == 1:
            print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                  'Perplexity {3:.3f} | Acc {4:.3f} | {5} words/s | '
                  '{6:.1f} ms/batch'.format(
                      epoch + 1, i, total_loss / i,
                      np.exp(total_loss / i),
                      total_correct / total_words * 100.0,
                      int(total_words / (time.time() - start)),
                      1000 * (time.time() - start) / i),
                  flush=True)
        del loss
        del scores
    return total_loss / i, total_correct / total_words * 100.0


def main(args):
    # IO -- Data Loader, generate a batch of (inputs, labels) in each iteration
    tr_dataset = PuncDataset(args.train_data, args.vocab, args.punc_vocab)
    cv_dataset = PuncDataset(args.cv_data, args.vocab, args.punc_vocab)
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
        avg_loss, avg_acc = run_one_epoch(tr_dataset, model, criterion,
                                          optimizer, epoch, args)
        print('-'*85)
        print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Train Loss {2:.3f} | Train Acc {3:.2f} '.format(
              epoch + 1, time.time() - start, avg_loss, avg_acc))
        print('-'*85)

        ######## Save model at each epoch
        if args.checkpoint:
            file_path = os.path.join(save_folder, 'epoch%d.pth.tar'%(epoch + 1))
            torch.save(LSTMPPunctuator.serialize(model, optimizer, epoch + 1),
                       file_path)
            print('Saving checkpoint model to %s' % file_path)

        ######## Cross validation
        print('Cross validation...')
        model.eval()  # Turn off Batchnorm & Dropout
        start = time.time()
        val_loss, val_acc = run_one_epoch(cv_dataset, model, criterion,
                                          optimizer, epoch, args,
                                          cross_valid=True)
        print('-'*85)
        print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Valid Loss {2:.3f} | Valid Acc {3:.2f} '.format(
              epoch + 1, time.time() - start, val_loss, val_acc))
        print('-'*85)

        ######## Adjust learning rate, halving
        if args.half_lr and val_loss >= prev_val_loss:
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
            torch.save(LSTMPPunctuator.serialize(model, optimizer, epoch + 1),
                       file_path)
            print("Find better validated model, saving to %s" % file_path)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
