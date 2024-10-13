import numpy as np
from math import log

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = np.mean([len(doc) for doc in corpus])
        self.doc_freqs = []
        self.idf_values = {}
        self.doc_len = [len(doc) for doc in corpus]
        self.build()

    def build(self):
        """Build the document frequency table and compute IDF values."""
        term_doc_freqs = {}
        for doc in self.corpus:
            term_freqs = {}
            for term in doc:
                if term not in term_freqs:
                    term_freqs[term] = 0
                term_freqs[term] += 1
            self.doc_freqs.append(term_freqs)
            for term in term_freqs:
                if term not in term_doc_freqs:
                    term_doc_freqs[term] = 0
                term_doc_freqs[term] += 1

        # Calculate IDF for each term
        for term, doc_freq in term_doc_freqs.items():
            self.idf_values[term] = log((self.N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def score(self, query, doc_index):
        """Calculates the BM25 score for a single document."""
        score = 0
        doc_len = self.doc_len[doc_index]
        for term in query:
            if term in self.doc_freqs[doc_index]:
                freq = self.doc_freqs[doc_index][term]
                idf = self.idf_values.get(term, 0)
                score += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
        return score

    def rank(self, query):
        """Ranks all documents for the given query."""
        scores = [self.score(query, i) for i in range(self.N)]
        return np.argsort(scores)[::-1]