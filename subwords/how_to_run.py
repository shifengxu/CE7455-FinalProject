from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# training your tokenizer on a set of files just takes two lines of codes:
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(vocab_size=10000,
                     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
                     )
# tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)
tokenizer.train(files=["../data/adolescent_valid.txt"], trainer=trainer)

# Once your tokenizer is trained, encode any text with just one line:
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
