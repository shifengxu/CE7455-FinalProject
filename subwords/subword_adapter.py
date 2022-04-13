from subwords.subword_model import SubwordModel

class SubwordAdapter(object):
    def __init__(self, file_list, subword_vocab_size, log_fn=print):
        self.file_list = file_list
        self.subword_vocab_size = subword_vocab_size
        self._log_fn = lambda msg: log_fn(msg) if log_fn else None

        # ----------------------------------------------------------- log file list
        self._log_fn(f"SubwordAdapter.file_list: {len(file_list)}")
        for file in file_list: self._log_fn(f"        {file}")

        self._log_fn(f"SubwordAdapter.subword_vocab_size: {subword_vocab_size}")
        self.subword_model = SubwordModel(subword_vocab_size)
        self._log_fn(f"SubwordAdapter.subword_model.train({len(file_list)} files)...")
        self.subword_model.train(file_list)
        self._log_fn(f"SubwordAdapter.subword_model.train({len(file_list)} files)...Done")

        self.subword2Id_dict = {
            "\r\n": 0,      # do not use id 0. So put impossible string for it.
            "\r\n\r\n": 1,  # special purpose for empty result of subword.encode()
        }

    def subword_to_id(self, subword):
        if subword in self.subword2Id_dict:
            return self.subword2Id_dict[subword]
        else:
            cnt = len(self.subword2Id_dict)
            self.subword2Id_dict[subword] = cnt
            return cnt
    # subword_to_id()

    def str_to_subword_idx_arr(self, s):
        id_arr = []
        subwords = self.subword_model.encode(s)
        for sw in subwords:
            i = self.subword_to_id(sw)
            id_arr.append(i)
        # we must return non-empty list.
        # if input string is "/", the subword encode() function may return empty list.
        # And empty subword id list may cause issue when call this function:
        #    torch.nn.utils.rnn.pack_padded_sequence(embeds, length, enforce_sorted=False)
        # because `length` list will have 0.
        # so here we guarantee that id_arr is non-empty
        if len(id_arr) == 0:
            id_arr.append(1)
        return id_arr


    def inputs_to_ids(self, inputs, idx2word_dict):
        """ inputs is 2D integer tensor """
        res = []
        sw_cnt = 0 # subword count
        for _, x in enumerate(inputs):
            x_arr = []
            for _, w_idx in enumerate(x):  # word index
                w_str = idx2word_dict[w_idx]
                subword_idx_arr = self.str_to_subword_idx_arr(w_str)
                x_arr.append(subword_idx_arr)
                sw_cnt += len(subword_idx_arr)
            # for
            res.append(x_arr)
        # for
        return res, sw_cnt

    def __str__(self):
        return f"SubwordAdapter({len(self.file_list)} files, {self.subword_vocab_size})"
# class
