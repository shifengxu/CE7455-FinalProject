from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

class SubwordModel:

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, files):
        """ training your tokenizer on a set of files """
        trainer = BpeTrainer(vocab_size=self.vocab_size,
                             special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                             show_progress=False,
                             )
        self.tokenizer.train(files=files, trainer=trainer)

    def encode(self, sequence):
        output = self.tokenizer.encode(sequence)
        return output.tokens

# class
