import numpy as np
from sklearn.decomposition import TruncatedSVD, randomized_svd
from gensim.models.keyedvectors import KeyedVectors
import time
from collections import defaultdict, Counter
import codecs
from nltk.tokenize import word_tokenize
from numba import njit

class Analogy:
    def __init__(self):
        path_to_wikipedia = "wikipedia2text-extracted.txt"
        with open(path_to_wikipedia, "rb") as f:
            wikipedia = f.read().decode().lower()
        tokens = word_tokenize(wikipedia.lower())
        
        self.sorted_words = self.generate_sorted_words(tokens)
        word2code = self.generate_word2code(self.sorted_words)
        self.codes = self.convert_tokens_to_codes(tokens, word2code)
        
    def generate_sorted_words(self, tokens):
        """ 
        Create list of unique words sorted by count in descending order

        Parameters
        ----------
        tokens: list(str)
            A list of tokens (words), e.g., ["the", "cat", "in", "the", "in", "the"]

        Returns
        -------
        list(str)
            A list of unique tokens sorted in descending order, e.g., ["the", "in", cat"]

        """
        # SOLUTION
        counter = Counter(tokens)
        words = [word for word, count in counter.most_common()]
        return words

    def generate_word2code(self, sorted_words):
        """ 
        Create dict that maps a word to its position in the sorted list of words

        Parameters
        ---------
        sorted_words: list(str)
            A list of unique words, e.g., ["b", "c", "a"]

        Returns
        -------
        dict[str, int]
            A dictionary that maps a word to an integer code, e.g., {"b": 0, "c": 1, "a": 2}

        """
        # SOLUTION
        word2code = {w : i for i, w in enumerate(sorted_words)}
        return word2code

    def convert_tokens_to_codes(self, tokens, word2code):
        """ 
        Convert tokens to codes.

        Parameters
        ---------
        tokens: list(str)
            A list of words, e.g., ["b", "c", "a"]
        word2code: dict[str, int]
            A dictionary mapping words to integer codes, e.g., {"b": 0, "c": 1, "a": 2}

        Returns
        -------
        list(int)
            A list of codes corresponding to the input words, e.g., [0, 1, 2].
        """
        # SOLUTION
        return [word2code[token] for token in tokens]

    @staticmethod
    @njit
    def generate_word_by_context(codes, max_vocab_words=1000, max_context_words=1000, context_size=2, weight_by_distance=False):
        """ Create matrix of vocab word by context word (possibly weighted) co-occurrence counts.

            Parameters
            ----------
            codes: list(int)
                A sequence of word codes.
            max_vocab_words: int
                The max number of words to include in vocabulary (will correspond to rows in matrix).
                This is equivalent to the max word code that will be considered/processed as the center word in a window.
            max_context_words: int
                The max number of words to consider as possible context words (will correspond to columns in matrix).
                This is equivalent to the max word code that will be considered/processed when scanning over contexts.
            context_size: int
                The number of words to consider on both sides (i.e., to the left and to the right) of the center word in a window.
            weight_by_distance: bool
                Whether or not the contribution of seeing a context word near a center word should be 
                (down-)weighted by their distance:

                    False --> contribution is 1.0
                    True  --> contribution is 1.0 / (distance between center word position and context word position)

                For example, suppose ["i", "am", "scared", "of", "dogs"] has codes [45, 10, 222, 25, 88]. 

                With weighting False, 
                    X[222, 25], X[222, 10], X[222, 25], and X[222, 88] all get incremented by 1.

                With weighting True, 
                    X[222, 25] += 1.0/2 
                    X[222, 10] += 1.0/1 
                    X[222, 25] += 1.0/1
                    X[222, 88] += 1.0/2

            Returns
            -------
            (max_vocab_words x max_context_words) ndarray
                A matrix where rows are vocab words, columns are context words, and values are
                (possibly weighted) co-occurrence counts.
        """
        X = np.zeros((max_vocab_words, max_context_words))
        for i in range(context_size, len(codes) - context_size):
            center_code = codes[i]
            if center_code < max_vocab_words:
                # left side
                for j in range(1, context_size + 1):
                    context_code = codes[i - j]
                    if context_code < max_context_words:
                        value = 1.0
                        if weight_by_distance:
                            value = 1.0 / j
                        X[center_code, context_code] += value
                # right side
                for j in range(1, context_size + 1):
                    context_code = codes[i + j]
                    if context_code < max_context_words:
                        value = 1.0
                        if weight_by_distance:
                            value = 1.0 / j
                        X[center_code, context_code] += value

        return X
    
    @staticmethod
    def reduce(X, n_components, power=0.0):
        U, Sigma, VT = randomized_svd(X, n_components=n_components)
        # note: TruncatedSVD always multiplies U by Sigma, but can tune results by just using U or raising Sigma to a power
        return U * (Sigma**power)
    
    @staticmethod
    def x_log(X_wiki):
        return np.log10(1 + X_wiki, dtype="float32")

test = Analogy()
max_vocab_words = 1000
max_context_words = 1000
X_wiki = Analogy.generate_word_by_context(test.codes, 
                                  max_vocab_words=max_vocab_words, 
                                  max_context_words=max_context_words, 
                                  context_size=4,
                                  weight_by_distance=True)
my_vectors = Analogy.reduce(Analogy.x_log(X_wiki), n_components=200)

# save in word2vec format (first line has vocab_size and dimension; other lines have word followed by embedding)
with codecs.open("my_vectors_200.txt", "w", "utf-8") as f:
    f.write(str(max_vocab_words) + " " + str(200) + "\n")  
    for i in range(max_vocab_words):
        f.write(test.sorted_words[i] + " " + " ".join([str(x) for x in my_vectors[i,:]]) + "\n")

# load back in
word_vectors = KeyedVectors.load_word2vec_format("my_vectors_200.txt", binary=False)


print(word_vectors.wv.similar_by_word("red"))