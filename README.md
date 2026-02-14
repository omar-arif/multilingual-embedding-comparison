# multilingual-embedding-comparison

## Dataset

Tatoeba English--French parallel sentences (80,000 pairs)\
https://tatoeba.org/fr/downloads

------------------------------------------------------------------------

## Setup

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

### 1. Generate Embeddings

Run `embedding.ipynb` (execute relevant cells for your) to create:

-   One-hot embeddings\
-   TF-IDF embeddings\
-   Word2Vec embeddings\
-   FastText embeddings\
-   GloVe embeddings

Output directory:

./embeddings/{ohe, tfidf, w2v, ft, glv}/

------------------------------------------------------------------------

### 2. Align Embeddings

``` bash
python run_muse.py align    # Align embeddings
python run_muse.py eval     # Evaluate alignment
python run_muse.py          # Run both
```

Output directory:

./embeddings/{type}/aligned/
