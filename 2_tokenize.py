import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence


tokenizer_file = 'data/tokenizer.json'


if not os.path.exists(tokenizer_file):
    tokenizer = Tokenizer.from_file(tokenizer_file)
    
    result = tokenizer.encode("This is an example sentence for tokenization.")
    
    print(result.tokens)
    exit()


file = 'data/yt_cmts_230624_en.txt'

# Initialize a tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Initialize a normalizer
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = Whitespace()

# Initialize a trainer, assuming you have some text file `text.txt`
trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])

# Train the tokenizer
tokenizer.train(files=[file], trainer=trainer)

# Tokenize some input text
output = tokenizer.encode("This is an example sentence for tokenization.")

# Print out the tokens
print('vocab size:', tokenizer.get_vocab_size())
print(output.tokens)

# save the tokenizer
tokenizer.save(tokenizer_file)
