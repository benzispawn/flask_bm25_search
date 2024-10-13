# Flask BM25 Search Application

This project implements a search engine using the BM25 ranking algorithm with Flask. BM25 is a ranking function commonly used in search engines to score document relevance to a query based on term frequency and inverse document frequency (TF-IDF).

## Features

- **BM25 Ranking**: Ranks documents based on relevance to the search query.
- **Flask Web Application**: Provides a web-based interface for searching documents.
- **Preprocessing**: Implements text preprocessing (e.g., stemming, stopword removal) for better search accuracy, while displaying the original documents.
- **Dynamic Search Results**: Displays the original documents with BM25 scores based on preprocessed versions.

## Requirements

- Python 3.x
- Flask
- NumPy
- NLTK (Natural Language Toolkit)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/flask_bm25_search.git
cd flask_bm25_search
