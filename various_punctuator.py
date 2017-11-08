import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model import lstmp


class BLSTMPunctuator(nn.Module):
    """
    BLSTM Punctuator.
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,
                 num_classes):
        super(BLSTMPunctuator, self).__init__()
        # hyper parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        # model component
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blstm = nn.LSTM(embedding_size, hidden_size, num_layers,
                             batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        # init weights
        self.init_weights(init_range=0.1)

    def init_hidden(self, batch_size):
        # *2 because bidirectional
        h0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size).cuda())
        return h0, c0

    def forward(self, inputs, inputs_lengths):
        """
        Args:
            inputs: (N, T), padded inputs, batch first
        Returns:
            score: (N, T, C)
        """
        hidden = self.init_hidden(inputs.size(0))
        embedding = self.embedding(inputs)
        # PyTorch padding mechanism
        packed = pack_padded_sequence(embedding, inputs_lengths, batch_first=True)
        packed_outputs, hidden = self.blstm(packed, hidden)
        out, out_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        out = out.contiguous()
        score = self.fc(out.view(out.size(0) * out.size(1), out.size(2)))
        return score.view(out.size(0), out.size(1), score.size(1))  #, hidden

    def init_weights(self, init_range=0.1):
        """ Initialize the weights. """
        for p in self.parameters():
            if p.dim() > 1:
                p.data.uniform_(-init_range, init_range)
            else:
                p.data.fill_(0)

    @staticmethod
    def serialize(model, optimizer, epoch):
        """ To store more information about model. """
        package = {'state_dict': model.state_dict(),
                   'optim_dict': optimizer.state_dict(),
                   'epoch': epoch}
        return package


class LSTMPPunctuator(nn.Module):
    """
    LSTMP Punctuator.
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, proj_size,
                 num_layers, num_classes):
        super(LSTMPPunctuator, self).__init__()
        # hyper parameters
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        # model component
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = lstmp.LSTMP(embedding_size, hidden_size,
                                num_layers=num_layers,
                                use_peepholes=True, proj_size=proj_size)
        self.fc = nn.Linear(proj_size, num_classes)
        # init weights
        self.init_weights(init_range=0.1)

    def forward(self, inputs, hidden, reset_flags=None, train=False):
        """
        Args:
            inputs: (N, T), padded inputs, batch first
        Returns:
            score: (N, T, C)
        """
        if train:
            hidden = self.reset_hidden(hidden, reset_flags)
        embedding = self.embedding(inputs)
        out, hidden = self.lstm(embedding, hidden)
        score = self.fc(out.view(out.size(0) * out.size(1), out.size(2)))
        return score.view(out.size(0), out.size(1), score.size(1)), hidden

    def init_weights(self, init_range=0.1):
        """ Initialize the weights. """
        for p in self.parameters():
            if p.dim() > 1:
                p.data.uniform_(-init_range, init_range)
            else:
                p.data.fill_(0)

    def init_hidden(self, batch_size):
        if self.proj_size is not None:
            h0 = Variable(torch.zeros(self.num_layers, batch_size, self.proj_size).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        return h0, c0

    def reset_hidden(self, hidden, reset_flags):
        """
        Reset the hidden according to the reset_flags.
        Call this at each minibatch in forward().
        Inputs:
        - hidden: Variable, shape = (BPTT steps, N, H)
        - reset_flags: Variable, shape = (N, )
        """
        # detach it from history (pytorch mechanics)
        h = Variable(hidden[0].data)
        c = Variable(hidden[1].data)
        hidden = (h, c)
        for b, flag in enumerate(reset_flags):
            if flag.data[0] == 1:  # data[0] access the data in Variable
                hidden[0][:, b, :].data.fill_(0)
                hidden[1][:, b, :].data.fill_(0)
        return hidden

    @staticmethod
    def serialize(model, optimizer, epoch):
        """ To store more information about model. """
        package = {'state_dict': model.state_dict(),
                   'optim_dict': optimizer.state_dict(),
                   'epoch': epoch}
        return package
