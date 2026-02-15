# Multilingual Embedding Comparison

## Dataset

Tatoeba English-French parallel sentences (80,000 pairs from full corpus)  
https://tatoeba.org/fr/downloads

---

## Setup

```bash
pip install -r requirements.txt
```

Required libraries: pandas, numpy, scikit-learn, gensim, mittens, torch, faiss-cpu, colorama

---

## Usage

### 1. Generate Embeddings

Run `embedding.ipynb` (all cells) to create embeddings for both languages:

- **One-hot embeddings** (1700-dim, limited vocab for memory)
- **TF-IDF embeddings** (1700-dim, limited vocab for memory)
- **Word2Vec embeddings** (300-dim, neural)
- **FastText embeddings** (300-dim, neural with subword info)
- **GloVe embeddings** (300-dim, co-occurrence matrix)

**Output directory:** `./embeddings/{ohe,tfidf,w2v,ft,glv}/`

#### Runtime (per language):
- One-hot: ~10 seconds
- TF-IDF: ~15 seconds
- Word2Vec: ~30 seconds
- FastText: ~45 seconds
- **GloVe: ~20-30 minutes** (co-occurrence matrix computation)

**Total notebook runtime: ~1 hour**

---

### 2. Align Embeddings with MUSE

Run `run_muse.py` to align English-French embedding spaces using supervised Procrustes alignment:

```bash
python run_muse.py align    # Align embeddings only
python run_muse.py eval     # Evaluate alignment only
python run_muse.py          # Run both alignment and evaluation (default)
```

**What it does:**
1. Clones Facebook MUSE repository
2. Patches code for PyTorch 2.6 compatibility
3. Downloads English-French bilingual dictionaries
4. Learns linear mapping between embedding spaces (5 refinement iterations)
5. Exports aligned embeddings in PyTorch format
6. Evaluates with Precision@1, Precision@5, Precision@10 metrics

**Output directory:** `./embeddings/{type}/aligned/`

#### Files created:
- `vectors-en.pth` - Aligned English embeddings
- `vectors-fr.pth` - Aligned French embeddings
- `best_mapping.pth` - Learned transformation matrix

---

## Project Structure

```
.
├── embedding.ipynb           # Step 1: Generate embeddings
├── run_muse.py              # Step 2: Align embeddings with MUSE
├── analysis.ipynb           # Visualization and analysis
├── requirements.txt         # Python dependencies
├── data/
│   └── french_english.tsv   # Text corpus
├── embeddings/              # Generated embeddings
│   ├── ohe/
│   │   ├── english_onehot.vec
│   │   ├── french_onehot.vec
│   │   └── aligned/         # MUSE output
│   ├── tfidf/
│   ├── w2v/
│   ├── ft/
│   └── glv/
└── MUSE/                    # Auto-cloned by run_muse.py
```

---

## Notes

- One-hot and TF-IDF use reduced vocabulary (1700 words) to avoid memory issues
- Neural embeddings (Word2Vec, FastText, GloVe) use full vocabulary