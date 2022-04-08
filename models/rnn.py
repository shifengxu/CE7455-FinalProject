import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False, log_fn=print):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.hid_state = None  # hidden state for RNN model.
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
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

    def forward(self, inputs):                      # input size  [35, 20]
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try back-propagating all the way to start of the dataset.
        h = self.repackage_hidden(self.hid_state)

        emb = self.drop(self.encoder(inputs))       # return size [35, 20, 200]
        output, h = self.rnn(emb, h)                # return size [35, 20, 200]
        self.hid_state = h
        output = self.drop(output)
        decoded = self.decoder(output)              # return size [35, 20, 33278]
        decoded = decoded.view(-1, self.ntoken)     # return size [700, 33278]
        return F.log_softmax(decoded, dim=1)

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
