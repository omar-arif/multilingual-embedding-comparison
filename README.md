# Multilingual Embedding Comparison

Build EN/FR word embeddings from a parallel corpus, align the spaces with MUSE, analyze aligned vs unaligned embeddings, then run a language classifier.

## Dataset

- File: `data/french_english.tsv`
- Columns: `id_en`, `en`, `id_fr`, `fr`
- Source: Tatoeba EN-FR parallel sentences
- Sampling:
  - `embedding.ipynb` samples 20000 sentence pairs (full TSV is larger) to keep OHE/TF-IDF feasible
  - `language_prediction.ipynb` samples 5000 pairs and caps sentence length to 30 tokens

## Install

```bash
pip install -r requirements.txt
```

## Run order (important)

### 1) Generate embeddings (run first)

Open `embedding.ipynb` and run all cells top to bottom.

It generates (EN and FR):
- OHE (1700 dim, reduced vocab)
- TF-IDF (1700 dim, reduced vocab)
- Word2Vec (300 dim)
- FastText (300 dim, also saves a `.model` used later for OOV tests)
- GloVe (300 dim, slowest step due to co-occurrence matrix)

Outputs go under:
- `embeddings/ohe/`, `embeddings/tfidf/`, `embeddings/w2v/`, `embeddings/ft/`, `embeddings/glv/`

### 2) Align embeddings with MUSE (run second)

Run one of these from the project root:

```bash
python run_muse.py
```

This runs BOTH alignment and evaluation by default.

Or run only one stage:

```bash
python run_muse.py align
python run_muse.py eval
```

What the script does (high level):
- Clones MUSE if missing
- Patches MUSE for PyTorch 2.6 loading behavior
- Downloads EN-FR dictionaries (uses bash script if available, otherwise a manual download fallback)
- Aligns each embedding type with supervised Procrustes (5 refinement iterations)
- Exports aligned embeddings and evaluates Precision@k

Outputs (per embedding type):
- `embeddings/<type>/aligned/vectors-en.pth`
- `embeddings/<type>/aligned/vectors-fr.pth`
- `embeddings/<type>/aligned/best_mapping.pth`

### 3) Analyze embeddings (run third)

Open `analysis.ipynb` and run all cells.

It loads:
- Unaligned `.vec` embeddings from Step 1
- Aligned `.pth` embeddings from Step 2

It runs similarity tests, OOV coverage checks, and PCA plots.

### 4) Language prediction (run fourth)

Open `language_prediction.ipynb` and run all cells.

It loads `.vec` embeddings (W2V, FT, GLV), builds padded sentence tensors (max length 30), trains an LSTM classifier, and prints metrics.


---

## Project Structure

```
.
|-- __pycache__/
|-- data/
|   `-- french_english.tsv
|-- embeddings/
|   |-- ft/
|   |   |-- aligned/
|   |   |   |-- best_mapping.pth
|   |   |   |-- params.pkl
|   |   |   |-- train.log
|   |   |   |-- vectors-en.pth
|   |   |   `-- vectors-fr.pth
|   |   |-- english_fasttext.model
|   |   |-- english_fasttext.model.wv.vectors_ngrams.npy
|   |   |-- english_fasttext.vec
|   |   |-- french_fasttext.model
|   |   |-- french_fasttext.model.wv.vectors_ngrams.npy
|   |   `-- french_fasttext.vec
|   |-- glv/
|   |-- ohe/
|   |-- tfidf/
|   `-- w2v/
|-- MUSE/
|-- .gitignore
|-- analysis.ipynb
|-- embedding.ipynb
|-- language_prediction.ipynb
|-- README.md
|-- requirements.txt
|-- run_muse.py
`-- utils.py
```

---

## Notes

- One-hot and TF-IDF use reduced vocabulary (1700 tokens) to avoid memory issues
- Neural embeddings (Word2Vec, FastText, GloVe) use full vocabulary (around 10k tokens)