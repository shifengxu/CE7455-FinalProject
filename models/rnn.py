import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False, log_fn=print, device='cuda',
                 char_mode='CNN', char_cnt=200, char_emsize=25, char_nhid=200):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.device = device
        self.hid_state = None           # hidden state for RNN model.
        inp_size = ninp + char_emsize   # input size
        self.rnn = self.gen_rnn(rnn_type, inp_size, nhid, nlayers, dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.tie_weights(tie_weights, nhid, ninp)
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        def _log_fn(msg): log_fn(msg) if log_fn else None
        _log_fn(f"RNNModel.rnn_type: {self.rnn_type}")
        _log_fn(f"RNNModel.ntoken  : {self.ntoken}")
        _log_fn(f"RNNModel.ninp    : {ninp}")
        _log_fn(f"RNNModel.nhid    : {self.nhid}")
        _log_fn(f"RNNModel.nlayers : {self.nlayers}")
        _log_fn(f"RNNModel.dropout : {dropout}")
        _log_fn(f"RNNModel.tie_weights: {tie_weights}")

        self.char_mode = char_mode
        self.char_cnt = char_cnt
        self.char_emsize = char_emsize
        self.char_nhid = char_nhid
        self.char_embeds = nn.Embedding(char_cnt, char_emsize)
        wt = self.char_embeds.weight
        bias = np.sqrt(3.0 / wt.size(1))  # init embedding. copied from A1
        nn.init.uniform_(wt, -bias, bias)
        if self.char_mode == 'LSTM':
            self.char_lstm = nn.LSTM(char_emsize, char_nhid, num_layers=1)
            init_lstm(self.char_lstm)
        else:
            self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=char_emsize,
                                       kernel_size=(3, char_emsize), padding=(2, 0))

    @staticmethod
    def gen_rnn(rnn_type, ninp, nhid, nlayers, dropout):
        if rnn_type in ['LSTM', 'GRU']:
            return getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            return nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

    def tie_weights(self, tie_weights, nhid, ninp):
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, inputs, chars3):              # input size  [35, 20]
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try back-propagating all the way to start of the dataset.
        h = self.repackage_hidden(self.hid_state)

        emb = self.drop(self.encoder(inputs))       # return size [35, 20, 200]
        char_embeds = self.get_char_embeds(chars3)
        emb = torch.cat((emb, char_embeds), dim=2)
        output, h = self.rnn(emb, h)                # return size [35, 20, 200]
        self.hid_state = h
        output = self.drop(output)
        decoded = self.decoder(output)              # return size [35, 20, 33278]
        decoded = decoded.view(-1, self.ntoken)     # return size [700, 33278]
        return F.log_softmax(decoded, dim=1)

    def get_char_embeds(self, chars3):
        chars3_masked = handle_chars3(chars3, self.char_mode)
        chars3_masked = chars3_masked.to(self.device)
        if self.char_mode == 'LSTM':
            return None
        else:
            chars_embeds = self.char_embeds(chars3_masked)
            # size: (35, 20, 13, 25). (bptt, bsz, max_word_len, char_emsize)

            d0, d1, d2, d3 = chars_embeds.size()
            chars_embeds = chars_embeds.view(-1, d2, d3)  # (700 13 25)
            chars_embeds = chars_embeds.unsqueeze(1)      # (700 1 13 25)

            # Creating Character level representation using Convolutional Neural Network
            # followed by a Maxpooling Layer
            # size: (700 25 15 1) <= (700 1 13 25)
            out3 = self.char_cnn3(chars_embeds)

            # size: (700 25 1 1) <= (700 25 15 1)
            chars_embeds = nn.functional.max_pool2d(out3, kernel_size=(out3.size(2), 1))

            # size: (700 25) <= (700 25 1 1)
            chars_embeds = chars_embeds.squeeze(3)
            chars_embeds = chars_embeds.squeeze(2)
            chars_embeds = chars_embeds.view(d0, d1, -1)
        return chars_embeds

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            new_hidden = (weight.new_zeros(self.nlayers, bsz, self.nhid),
                          weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            new_hidden = weight.new_zeros(self.nlayers, bsz, self.nhid)

        self.hid_state = new_hidden

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

def handle_chars3(chars3, char_mode='LSTM'):
    if char_mode == 'LSTM':
        chars2_sorted = sorted(chars3, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars3):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj and j not in d and i not in d.values():
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)
        chars3_masked = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars3_masked[i, :chars2_length[i]] = c
        chars3_masked = Variable(torch.LongTensor(chars3_masked))
    elif char_mode == 'CNN':
        max_len = 0
        # find max length (max word size)
        for row in chars3:
            for col in row:
                max_len = max(max_len, len(col))
        # Padding each word to max word size of that sentence
        dim0 = len(chars3)
        dim1 = len(chars3[0])
        chars3_masked = np.zeros((dim0, dim1, max_len), dtype='int')
        for i in range(dim0):
            for j in range(dim1):
                chars3_masked[i, j, :len(chars3[i][j])] = chars3[i][j]
        chars3_masked = Variable(torch.LongTensor(chars3_masked))
    else:
        raise Exception("Unsupported char_mode: " + char_mode)
    return chars3_masked

def init_lstm(input_lstm):
    """
    Initialize lstm

    PyTorch weights parameters:

        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size * hidden_size)`

        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size * hidden_size)`
    """

    # Weights init for forward layer
    for ind in range(0, input_lstm.num_layers):
        # Gets the weights Tensor from our model, for the input-hidden weights in our current layer
        weight = eval('input_lstm.weight_ih_l' + str(ind))

        # Initialize the sampling range
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))

        # Randomly sample from our sampling range using uniform distribution and apply it to our current layer
        nn.init.uniform_(weight, -sampling_range, sampling_range)

        # Similar to above but for the hidden-hidden weights of the current layer
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)

    # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

    # Bias initialization steps

    # We initialize them to zero except for the forget gate bias, which is initialized to 1
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))

            # Initializing to zero
            bias.data.zero_()

            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            # Similar for the hidden-hidden layer
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        # Similar to above, we do for backward layer if we are using a bi-directional LSTM
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
