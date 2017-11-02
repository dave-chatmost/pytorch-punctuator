import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


class LSTMPunctuator(nn.Module):
    pass
