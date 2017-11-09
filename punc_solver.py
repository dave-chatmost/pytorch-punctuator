import os
import time

import numpy as np
import torch
from torch.autograd import Variable

import utils
from data.truncated_bptt_data_loader import get_one_batch_data


class PuncSolver(object):
    """
    TODO: DOC.
    """

    def __init__(self, data, model, criterion, optimizer, **kwargs):
        """
        Construct a new PuncSolver instance.
        """
        self.tr_dataset = data['tr_dataset']
        self.cv_dataset = data['cv_dataset']
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        # Unpack keyword arguments
        # training hyper parameters
        self.batch_size = kwargs.pop('batch_size', 32)
        self.bptt_step = kwargs.pop('bptt_step', 20)
        self.epochs = kwargs.pop('epochs', 2)
        self.half_lr = kwargs.pop('half_lr', False)
        self.max_norm = kwargs.pop('max_norm', 250)
        self.early_stop = kwargs.pop('early_stop', False)
        # save and load model
        self.save_folder = kwargs.pop('save_folder', 'exp/temp')
        self.checkpoint = kwargs.pop('checkpoint', False)
        self.continue_from = kwargs.pop('continue_from', '')
        self.model_path = kwargs.pop('model_path', 'final.pth.tar')
        # logging
        self.verbose = kwargs.pop('verbose', True)
        self.print_freq = kwargs.pop('print_freq', 10)

        self._reset()


    def _reset(self):
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
        else:
            self.start_epoch = 0
        # Create save folder
        utils.mkdir(self.save_folder)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False


    def train(self):
        # Train model multi-epochs
        for epoch in range(self.start_epoch, self.epochs):
            ######## Train one epoch
            print('Training...')
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            # do not need this parameters any more
            avg_loss, avg_acc = self.run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f} | Train Acc {3:.2f} '.format(
                      epoch + 1, time.time() - start, avg_loss, avg_acc))
            print('-' * 85)

            ######## Save model at each epoch
            if self.checkpoint:
                file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar'%(epoch + 1))
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            ######## Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            start = time.time()
            val_loss, val_acc = self.run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f} | Valid Acc {3:.2f} '.format(
                      epoch + 1, time.time() - start, val_loss, val_acc))
            print('-' * 85)

            ######## Adjust learning rate, halving
            if self.half_lr and val_loss >= self.prev_val_loss:
                if self.early_stop and self.halving:
                    print("Already start halving learing rate, it still gets too "
                          "small imporvement, stop training early.")
                    break
                self.halving = True
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
            self.prev_val_loss = val_loss

            ######## Save the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)


    def run_one_epoch(self, epoch, cross_valid=False):
        dataset = self.cv_dataset if cross_valid else self.tr_dataset
        batch_size = self.batch_size
        steps = self.bptt_step

        feats_utt = [np.array([])] * batch_size  # every element is numpy ndarray
        targets_utt = [np.array([])] * batch_size
        new_utt_flags = [0] * batch_size
        # hidden = model.module.init_hidden(batch_size)  # nn.DataParalle()
        hidden = self.model.init_hidden(batch_size)

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
            scores, hidden = self.model(inputs, hidden, new_utt_flags, train=True)
            scores = scores.view(-1, self.model.num_class)  # make it model.num_class
            loss = self.criterion(scores, targets.view(-1))
            if not cross_valid:
                self.optimizer.zero_grad()
                # 3. backward
                loss.backward()
                # Clip gradient
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.max_norm)
                # 4. update
                self.optimizer.step()

            _, predict = torch.max(scores, 1)

            total_correct += (predict == targets.view(-1)).sum().data[0]
            total_loss += loss.data[0]
            total_words += batch_size * steps

            i += 1
            if self.verbose and i % self.print_freq == 1:
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
