from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class IRIndex:
    def __init__(self, documents: list[str]):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[a-z0-9]+",
            ngram_range=(1, 2),
        )
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def score(self, query: str) -> list[float]:
        q_vec = self.vectorizer.transform([query])
        scores = (self.doc_vectors @ q_vec.T).toarray().ravel()
        return scores.tolist()
