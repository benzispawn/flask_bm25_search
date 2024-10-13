from crypt import methods

from flask import Flask, render_template, request, jsonify
from bm25 import BM25
from preprocessor import preprocess

app = Flask(__name__)

# Load the corpus (from a text file)
with open('data/corpus.txt', 'r') as f:
    original_corpus = [line.strip().split() for line in f]

# Preprocess corpus
corpus = [preprocess(' '.join(doc)) for doc in original_corpus]

# Initialize BM25 with the preprocessed corpus
bm25 = BM25(corpus)
@app.route('/')
def home():  # put application's code here
    return render_template('search.html')

@app.route('/search', methods=['GET'])
def search():
    """Handle search queries and return ranked results."""
    query = request.args.get('q', '').lower().split()  # Convert query to lowercase
    query = preprocess(' '.join(query))
    # print(f"Query: {query}")  # Debugging print

    if query:
        # Get ranked documents based on the query
        ranked_docs = bm25.rank(query)
        # print(f"Ranked docs: {ranked_docs}")  # Debugging print

        # Display the ranked documents along with their BM25 scores
        results = [
            {
                'document': ' '.join(original_corpus[i]),  # Convert document tokens back to a string
                'score': bm25.score(query, i)     # Show the BM25 score
            }
            for i in ranked_docs
        ]
        return jsonify(results)  # Return results as JSON for the frontend
    return jsonify([])



if __name__ == '__main__':
    app.run()
