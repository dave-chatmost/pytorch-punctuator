import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from data_loader import get_loader
from various_punctuator import BLSTMPunctuator


# hyper parameters
VOCAB_SIZE = 100000 + 2
EMBED_SIZE = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASS = 5
lr = 0.001
num_epochs = 100
batch_size = 10


# Data Loader, generate a batch of (inputs, labels)
data_loader = get_loader('example/train', 'example/vocab.10W',
                         'example/punc_vocab', batch_size=batch_size)
# model                        
punctuator = BLSTMPunctuator(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASS)
punctuator.cuda()
# loss
criterion = nn.CrossEntropyLoss()
# optimizer to update model's parameters
optimizer = torch.optim.Adam(punctuator.parameters(), lr=lr)

# for loop
# 1. mini-batch data
# 2. forward
# 3. backward
# 4. update
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    total_words = 0
    start = time.time()
    for i, (inputs, labels, lengths) in enumerate(data_loader):
        # 1. mini-batch data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # 2. forward and compute loss
        optimizer.zero_grad()
        scores = punctuator(inputs, lengths)
        scores = scores.view(-1, NUM_CLASS)
        loss = criterion(scores, labels.view(-1))
        # 3. backward
        loss.backward()
        # 4. update
        optimizer.step()

        _, predict = torch.max(scores, 1)
        total_correct += (predict == labels.view(-1)).sum().data[0]
        total_loss += loss.data[0]
        total_words += np.sum(lengths)

        if i % 10 == 0:
            print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                  'Perplexity {3:.3f} | Acc {4:.3f}'.format(
                      epoch + 1, i, total_loss / (i + 1),
                      np.exp(total_loss / (i + 1)), 1.0),
                      flush=True)
    print('Epoch {0} | Total time {1:.1f} s'.format(epoch, time.time()-start))
