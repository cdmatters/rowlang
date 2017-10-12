import nltk

import os
from functools import reduce

NLTK_DIR = "../data/nltk_data"
nltk.data.path.append(NLTK_DIR)

class Tokenizer:
    try:
        punkt = nltk.data.load(NLTK_DIR +'/tokenizers/punkt/english.pickle')
    except LookupError:
        nltk.download('punkt', download_dir=NLTK_DIR)
        punkt = nltk.data.load(NLTK_DIR +'/tokenizers/punkt/english.pickle')
    
    def __init__(self):
        pass

    def tokenize(self, filename):
        with open(filename, "r", encoding="latin_1") as f:
            sentences = self.__class__.punkt.tokenize(f.read())
            split_sentences = [ nltk.word_tokenize(s) for s in sentences ]
            
        return Tokens(filename, sentences=split_sentences)


class Tokens:
    def __init__(self, source, sentences):
        self.source = source
        self.sentences = sentences
        self._words = None

    def __iter__(self):
        for s in self.sentences:
            for word in s:
                yield word

    def __str__(self):
        return '''Tokens: Title: {} Words: {} Sentences: {}
    0 - {}
    1 - {}
    2 - {}
    ... '''.format(self.source, len(self.words), len(self.sentences),
                      self.sentences[0], self.sentences[1], self.sentences[2])
        
    @property
    def words(self):
        if not self._words:
           self._words = list(self)
        return self._words



if __name__ == '__main__':
    hp1 = '../data/hp1.txt'
    t = Tokenizer()
    tks = t.tokenize(hp1)
    print(tks)

    for i, t in enumerate(tks):
        print (i, t)
        if i == 10:
            break
