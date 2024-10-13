import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(text):
    """Lowercase, remove stopwords, and apply stemming."""
    words = text.lower().split()
    return [ps.stem(word) for word in words if word not in stop_words]
