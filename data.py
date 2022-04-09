import os
from io import open
import torch

from subwords.subword_model import SubwordModel

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
def _batchify(data: torch.Tensor, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivision of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    batch = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return batch, target

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, train_file_list: [], valid_file_list: [], test_file_list: [],
                 subword_vocab_size=None, log_fn=print):
        self.dictionary = Dictionary()
        self.file_dict = {}  # dict, filename to tokens
        self._log_fn = lambda msg: log_fn(msg) if log_fn else None

        # ----------------------------------------------------------- log file list
        self._log_fn(f"corpus.train_file_list: {len(train_file_list)}")
        for file in train_file_list: self._log_fn(f"        {file}")
        self._log_fn(f"corpus.valid_file_list: {len(valid_file_list)}")
        for file in valid_file_list: self._log_fn(f"        {file}")
        self._log_fn(f"corpus.test_file_list: {len(test_file_list)}")
        [self._log_fn(f"        {file}") for file in test_file_list]

        # ----------------------------------------------------------- subword
        self._log_fn(f"corpus.subword_vocab_size: {subword_vocab_size}")
        if subword_vocab_size:
            self.subword_model = SubwordModel(subword_vocab_size)
            files = train_file_list + valid_file_list
            self._log_fn(f"corpus.subword_model.train({len(files)} files)...")
            for file in files: self._log_fn(f"        {file}")
            self.subword_model.train(files)
            self._log_fn(f"corpus.subword_model.train({len(files)} files)...Done")
        else:
            self.subword_model = None
            self._log_fn(f"corpus.subword_model = None. Corpus will not use subword.")

        # ----------------------------------------------------------- tokenize
        self.char_cnt_dict = {}
        self._log_fn(f"corpus tokenize...")
        for file in train_file_list:
            tokens = self.tokenize(file, self.char_cnt_dict)
            self._log_fn(f"        train tokens: {len(tokens):6d}  {file}")
            self.file_dict[file] = tokens
        for file in valid_file_list:
            tokens = self.tokenize(file, self.char_cnt_dict)
            self._log_fn(f"        valid tokens: {len(tokens):6d}  {file}")
            self.file_dict[file] = tokens
        for file in test_file_list:
            tokens = self.tokenize(file, self.char_cnt_dict)
            self._log_fn(f"        test  tokens: {len(tokens):6d}  {file}")
            self.file_dict[file] = tokens
        self.ntokens = len(self.dictionary)

        # ----------------------------------------------------------- char dict
        # Items are ordered by decreasing frequency
        sorted_items = sorted(self.char_cnt_dict.items(), key=lambda x: (-x[1], x[0]))
        # dict, char to id. And id start from 1.
        self.char_id_dict = {v[0]: i + 1 for i, v in enumerate(sorted_items)}
        self.char_count = len(self.char_id_dict)

        self._log_fn(f"corpus.ntokens      : {self.ntokens}")
        self._log_fn(f"corpus.char_count   : {self.char_count}")
        self._log_fn(f"corpus.char_cnt_dict: {len(self.char_cnt_dict)}")
        self._log_fn(f"corpus.char_id_dict : {len(self.char_id_dict)}")

    def tokenize(self, path, char_cnt_dict: dict = None):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Tokenize file content
        ids_arr = []
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = self._split_line(line)
                ids = []
                for word in words:
                    idx = self.dictionary.add_word(word)  # Add words to the dictionary
                    ids.append(idx)
                    if char_cnt_dict is not None:
                        for c in word:
                            if c in char_cnt_dict:
                                char_cnt_dict[c] += 1
                            else:
                                char_cnt_dict[c] = 1
                        # for char
                # for word
                ids_arr.append(torch.tensor(ids).type(torch.int64))
            # for line
        # with
        res = torch.cat(ids_arr)
        return res

    def _split_line(self, line):
        if self.subword_model:
            words = self.subword_model.encode(line)
        else:
            words = line.split()
        words += ['<eos>']
        return words

    def batchify(self, file_list, batch_size):
        arr = []
        for file in file_list:
            tokens = self.file_dict[file]
            arr.append(tokens)
        tokens_all = torch.cat(arr)
        batched = _batchify(tokens_all, batch_size)
        self._log_fn(f"corpus.batchify({len(file_list)} files, batch_size={batch_size})...")
        self._log_fn(f"        tokens_all: {len(tokens_all):6d}")
        self._log_fn(f"        batched   : {len(batched):6d}")
        [self._log_fn(f"        {f}") for f in file_list]
        return batched

    def word_to_char(self, word_tensor):
        """
        convert word tensor to char matrix.
        Usually, word_tensor is a 2D tensor, such as (35, 20).
        And "35" is bptt, while "20" is batch size.
        :param word_tensor:
        :return:
        """
        res = []
        for _, x in enumerate(word_tensor):
            x_arr = []
            for _, w_idx in enumerate(x):  # word index
                w_str = self.dictionary.idx2word[w_idx]
                c_idx_arr = []
                for c in w_str:
                    if c in self.char_id_dict:
                        c_idx_arr.append(self.char_id_dict[c])
                    else:
                        self._log_fn(f"!!![Warn] Not found char {c} in corpus.char_id_dict.")
                x_arr.append(c_idx_arr)
            # for
            res.append(x_arr)
        # for
        return res
