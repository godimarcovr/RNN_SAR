# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size)
            , nn.ReLU()
        )
        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 2)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTM(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden

        x = x.view(-1, x.size(2))

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()
        if len(gates.shape) == 1:
            gates = gates.unsqueeze(0)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)

        hy = torch.mul(outgate, F.tanh(cy))

        return hy, (hy, cy)


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="rnn", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if n_layers == 1:
            if self.model == "rnn":
                self.rnn = RNN(hidden_size, hidden_size)
            elif self.model == "lstm":
                self.rnn = LSTM(hidden_size, hidden_size)
        else:
            if self.model == "gru":
                self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
            elif self.model == "lstm":
                self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
            elif self.model == "rnn":
                self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

