import nltk

from .unit import Unit

import numpy as np

class Tokenize(Unit):
    """Process unit for text tokenization."""
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    def transform(self, input_: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input_: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        # return nltk.word_tokenize(input_)
        encoded_input = self.tokenizer.encode(input_, padding=True, truncation=True, return_tensors='pt')
        # return nltk.word_tokenize(input_)
        return np.array(encoded_input.squeeze()[1:]).tolist()
